"""
Точка входу агента — CLI-інтерфейс.

Використання:
    python main.py                   # повний запуск (генерація + навчання + симуляція)
    python main.py --mode train      # тільки навчання
    python main.py --mode simulate   # тільки симуляція (потребує навченої моделі)
    python main.py --steps 50        # кількість кроків симуляції
    python main.py --verbose         # детальний вивід кожного рішення
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.generate_data import generate_dataset
from src.pipeline import run_pipeline
from src.models import FailureClassifier, LSTMPredictor
from src.agent import PredictiveMaintenanceAgent, SensorReading
from src.metrics import print_metrics, agent_performance


def parse_args():
    p = argparse.ArgumentParser(description="Агент прогнозування відмов обладнання — Варіант 15")
    p.add_argument("--mode",    choices=["full", "train", "simulate"], default="full")
    p.add_argument("--steps",   type=int, default=250,  help="Кроки симуляції")
    p.add_argument("--verbose", action="store_true",   help="Детальний вивід")
    p.add_argument("--data",    default="data/sensor_data.csv")
    p.add_argument("--log",     default="agent_log.json")
    return p.parse_args()

def train_phase(data_path: str):
    print("\n[1/3] Генерація даних")
    if not os.path.exists(data_path):
        df = generate_dataset()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"      Збережено: {data_path}")
    else:
        print(f"      Використовується існуючий файл: {data_path}")

    print("\n[2/3] Pipeline обробки даних")
    pdata = run_pipeline(data_path)

    print("\n[3/3] Навчання моделей")

    # Random Forest
    clf = FailureClassifier()
    clf.train(pdata["X_train"], pdata["y_train"], pdata["feature_cols"])

    val_metrics = clf.evaluate(pdata["X_val"],  pdata["y_val"],  label="Validation")
    tst_metrics = clf.evaluate(pdata["X_test"], pdata["y_test"], label="Test")
    print_metrics(tst_metrics, title="Random Forest — тестова вибірка")

    # LSTM
    train_temp = pdata["train"]["temperature"].values
    lstm = LSTMPredictor(window_size=24)
    lstm.train(train_temp, epochs=20)

    test_temp  = pdata["test"]["temperature"].values
    lstm.evaluate_rmse(test_temp)

    # Feature importance
    print("\n  Топ-5 важливих ознак (Random Forest):")
    for name, imp in clf.feature_importance()[:5]:
        bar = "█" * int(imp * 40)
        print(f"    {name:30s} {imp:.3f}  {bar}")

    return clf, lstm, pdata


def simulate_phase(clf, lstm, pdata, steps: int, verbose: bool, log_path: str):
    print(f"\n  Симуляція агента ({steps} кроків × 15 хв)")

    # Ініціалізація агента
    agent = PredictiveMaintenanceAgent(
        classifier   = clf,
        lstm         = lstm,
        scaler       = pdata["scaler"],
        feature_cols = pdata["feature_cols"],
    )

    # Беремо тестову частину датасету
    test_df  = pdata["test"].reset_index(drop=True)
    steps    = min(steps, len(test_df))
    actuals  = []

    print(f"\n  {'Крок':>4}  {'Час':^19}  {'P(fail)':^9}  {'Рівень':^10}  Дія")
    print(f"  {'─'*4}  {'─'*19}  {'─'*9}  {'─'*10}  {'─'*28}")

    critical_steps = []

    for i in range(steps):
        row = test_df.iloc[i]
        reading = SensorReading(
            timestamp     = str(row["timestamp"]),
            temperature   = float(row["temperature"]),
            vibration     = float(row["vibration"]),
            pressure      = float(row["pressure"]),
            current       = float(row["current"]),
            rpm           = float(row["rpm"]),
            runtime_hours = float(row["runtime_hours"]),
        )
        decision = agent.act(reading)
        actuals.append(int(row["failure"]))

        # Мітки для рядка таблиці
        icon = {"NORMAL": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}[decision.risk_level]
        ts_short = decision.timestamp[5:16]  # MM-DD HH:MM
        action_short = decision.action[:90]

        print(f"  {i+1:>4}  {ts_short:^19}  {decision.failure_prob:^9.3f}  "
              f"{icon} {decision.risk_level:^8}  {action_short}")

        if decision.risk_level == "CRITICAL":
            critical_steps.append(i)
            if verbose:
                print(decision)

        # Позначаємо реальний результат для адаптації
        agent.mark_actual_outcome(i, int(row["failure"]))

    # Адаптація після симуляції
    print(f"\n  Адаптація агента")
    agent.adapt()

    # Підсумкові метрики
    print(f"\n  Підсумок роботи агента")
    s = agent.summary()
    print(f"  Всього кроків   : {s['total_steps']}")
    print(f"  Норма           : {s['normal']}")
    print(f"  Попередження    : {s['warnings']}")
    print(f"  Критичні сигнали: {s['critical_alerts']}")
    print(f"  Хибні тривоги   : {s['false_alarms']}")
    print(f"  Пропущені відмови: {s['missed_failures']}")

    perf = agent_performance(agent.decisions, actuals)
    print_metrics(perf, title="Ефективність агента (по критичним рішенням)")

    agent.save_log(log_path)
    plot_results(pdata["test"], agent.decisions, steps)
    return agent


def plot_results(test_df, decisions, steps):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Агент прогнозування відмов — Варіант 15", fontsize=14)

    idx = range(steps)
    temps = test_df["temperature"].values[:steps]
    vibs  = test_df["vibration"].values[:steps]
    probs = [d.failure_prob for d in decisions[:steps]]
    actual = test_df["failure"].values[:steps]

    # 1. Температура
    axes[0].plot(idx, temps, color="#378ADD", linewidth=1.2, label="Температура (°C)")
    axes[0].axhline(85, color="red", linestyle="--", linewidth=1, label="Поріг 85°C")
    axes[0].set_ylabel("°C")
    axes[0].legend(fontsize=9)
    axes[0].set_title("Показники сенсорів")

    # 2. Вібрація
    axes[1].plot(idx, vibs, color="#639922", linewidth=1.2, label="Вібрація (mm/s)")
    axes[1].axhline(12, color="red", linestyle="--", linewidth=1, label="Поріг 12 mm/s")
    axes[1].set_ylabel("mm/s")
    axes[1].legend(fontsize=9)

    # 3. P(failure) + реальні відмови
    axes[2].plot(idx, probs, color="#E24B4A", linewidth=1.5, label="P(failure)")
    axes[2].axhline(0.4, color="orange", linestyle="--", linewidth=1, label="Поріг WARNING")
    axes[2].axhline(0.7, color="red",    linestyle="--", linewidth=1, label="Поріг CRITICAL")
    for i, a in enumerate(actual):
        if a == 1:
            axes[2].axvline(i, color="black", alpha=0.4, linewidth=1.5)
    axes[2].set_ylabel("Ймовірність")
    axes[2].set_xlabel("Крок (×15 хв)")
    axes[2].legend(fontsize=9)
    axes[2].set_title("Рішення агента vs реальні відмови (чорні лінії)")

    plt.tight_layout()
    plt.savefig("agent_results.png", dpi=150, bbox_inches="tight")
    print("[Agent] Графік збережено: agent_results.png")
    plt.show()


def main():
    print("  Агент прогнозування відмов обладнання")
    print("  Predictive Maintenance · Варіант 15")
    print("  Методи: Random Forest + LSTM")
    args = parse_args()

    if args.mode in ("full", "train"):
        clf, lstm, pdata = train_phase(args.data)
    else:
        print("[!] Режим simulate без попереднього навчання — запустіть спочатку --mode train")
        sys.exit(1)

    if args.mode in ("full", "simulate"):
        simulate_phase(clf, lstm, pdata, args.steps, args.verbose, args.log)

    print(f"\n  Готово. Журнал рішень: agent_log.json" + "\n")


if __name__ == "__main__":
    main()
