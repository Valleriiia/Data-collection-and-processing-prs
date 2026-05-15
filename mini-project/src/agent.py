"""
Агент прогнозування відмов обладнання — Варіант 15
Реалізує повний ReAct-цикл:
    Perceive → Remember → Plan → Act → Adapt

Агентна логіка (Decision Policy):
    P(failure) < 0.40  → NORMAL   : моніторинг
    P(failure) < 0.70  → WARNING  : сповіщення + зниження навантаження
    P(failure) >= 0.70 → CRITICAL : аварійний сигнал + зупинка + ТО
"""

import json
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


# Структури даних

@dataclass
class SensorReading:
    timestamp: str
    temperature: float
    vibration: float
    pressure: float
    current: float
    rpm: float
    runtime_hours: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentDecision:
    timestamp: str
    failure_prob: float
    lstm_forecast: list        # прогноз температури на 4 кроки (1 год)
    risk_level: str            # NORMAL / WARNING / CRITICAL
    action: str
    reasoning: str
    actual_failure: Optional[int] = None   # заповнюється після факту

    def __str__(self):
        sep = "─" * 56
        return (
            f"\n{sep}\n"
            f"  Час         : {self.timestamp}\n"
            f"  P(failure)  : {self.failure_prob:.3f}\n"
            f"  Рівень      : {self.risk_level}\n"
            f"  Прогноз t°  : {[f'{v:.1f}' for v in self.lstm_forecast]}\n"
            f"  Дія         : {self.action}\n"
            f"  Обґрунтування: {self.reasoning}\n"
            f"{sep}"
        )


# Головний клас агента
class PredictiveMaintenanceAgent:
    """
    AI-агент для predictive maintenance.

    Параметри
    classifier  : навчений FailureClassifier
    lstm        : навчений LSTMPredictor
    scaler      : sklearn-scaler (StandardScaler)
    feature_cols: список ознак у порядку, якого очікує classifier
    memory_size : розмір ковзного вікна пам'яті (кількість вимірів)
    threshold_warning  : поріг для рівня WARNING
    threshold_critical : поріг для рівня CRITICAL
    """

    def __init__(
        self,
        classifier,
        lstm,
        scaler,
        feature_cols: list,
        memory_size: int = 96,          # 96 × 15 хв = 24 год
        threshold_warning:  float = 0.40,
        threshold_critical: float = 0.70,
    ):
        self.classifier  = classifier
        self.lstm        = lstm
        self.scaler      = scaler
        self.feature_cols = feature_cols
        self.th_warn     = threshold_warning
        self.th_crit     = threshold_critical

        # Пам'ять агента
        self.memory: deque = deque(maxlen=memory_size)   # сирі вимірювання
        self.decisions: list[AgentDecision] = []          # журнал рішень
        self.maintenance_log: list[dict]    = []          # журнал ТО

        # Статистика для адаптації
        self._false_alarms   = 0
        self._missed_failures = 0
        self._total_steps    = 0

        print("[Agent] Ініціалізовано. Пороги: "
              f"warn={self.th_warn}, crit={self.th_crit}")

    # 1. СПРИЙНЯТТЯ

    def perceive(self, reading: SensorReading) -> dict:
        """Отримує один вимір, зберігає в пам'ять."""
        self.memory.append(reading.to_dict())
        return reading.to_dict()

    # 2. ПАМ'ЯТЬ

    def remember(self) -> pd.DataFrame:
        """
        Повертає DataFrame з усіма збереженими вимірами.
        Використовується для обчислення ковзних ознак.
        """
        return pd.DataFrame(list(self.memory))

    def _compute_rolling_features(self, df: pd.DataFrame) -> dict:
        """
        Обчислює rolling-ознаки на основі пам'яті.
        Повертає словник значень для останньої точки.
        """
        features = {}
        window = 12  # 3 год
        for col in ["temperature", "vibration", "pressure", "current"]:
            if col not in df.columns:
                continue
            series = df[col]
            features[col]                   = series.iloc[-1]
            features[f"{col}_mean{window}"] = series.tail(window).mean()
            features[f"{col}_std{window}"]  = series.tail(window).std() if len(series) >= 2 else 0.0
            features[f"{col}_roc"]          = series.iloc[-1] - series.iloc[-2] if len(series) >= 2 else 0.0
        features["rpm"]                = df["rpm"].iloc[-1]         if "rpm"           in df.columns else 2980.0
        features["runtime_hours"]      = df["runtime_hours"].iloc[-1] if "runtime_hours" in df.columns else 0.0
        features["vibration_rms"] = float(
            np.sqrt((df["vibration"].tail(window) ** 2).mean())
        ) if "vibration" in df.columns else 0.0
        return features

    # 3. ПЛАНУВАННЯ

    def plan(self) -> tuple[float, list, str]:
        """
        Обчислює P(failure) через Random Forest.
        Отримує LSTM-прогноз температури на 1 год (4 кроки × 15 хв).
        Повертає: (prob, lstm_forecast, reasoning)
        """
        mem_df = self.remember()
        if len(mem_df) < 2:
            return 0.0, [], "Недостатньо даних в пам'яті."

        feat_dict = self._compute_rolling_features(mem_df)

        # Вектор ознак у порядку feature_cols
        feat_vec = np.array([[feat_dict.get(c, 0.0) for c in self.feature_cols]])

        # Масштабування (передаємо DataFrame зі збереженням назв колонок)
        try:
            feat_df     = pd.DataFrame(feat_vec, columns=self.feature_cols)
            feat_scaled = self.scaler.transform(feat_df)
        except Exception:
            feat_scaled = feat_vec

        prob = float(self.classifier.predict_proba(feat_scaled)[0])

        # LSTM-прогноз температури
        temp_series = mem_df["temperature"].values
        if len(temp_series) >= self.lstm.window_size:
            lstm_forecast = self.lstm.predict_horizon(temp_series, horizon=4)
        else:
            lstm_forecast = [round(temp_series[-1], 1)] * 4

        # Формуємо обґрунтування
        t    = feat_dict.get("temperature", 0)
        v    = feat_dict.get("vibration", 0)
        vrms = feat_dict.get("vibration_rms", 0)
        reasoning_parts = []
        if t > 85:
            reasoning_parts.append(f"температура {t:.1f}°C (норма <85°C)")
        if v > 12:
            reasoning_parts.append(f"вібрація {v:.2f} mm/s (норма <12)")
        if vrms > 10:
            reasoning_parts.append(f"RMS вібрації {vrms:.2f}")
        max_fc = max(lstm_forecast) if lstm_forecast else t
        if max_fc > 90:
            reasoning_parts.append(f"LSTM-прогноз t°={max_fc:.1f}°C через 1 год")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "параметри в нормі"

        return prob, lstm_forecast, reasoning

    # 4. ДІЯ

    def act(self, reading: SensorReading) -> AgentDecision:
        """
        Головний метод агента: сприймає → планує → діє.
        Повертає AgentDecision із рівнем ризику і дією.
        """
        self._total_steps += 1
        self.perceive(reading)
        prob, lstm_forecast, reasoning = self.plan()

        # Decision Policy
        if prob >= self.th_crit:
            risk   = "CRITICAL"
            action = ("🔴 АВАРІЙНИЙ СИГНАЛ: зупинити обладнання, "
                      "викликати бригаду ТО негайно.")
        elif prob >= self.th_warn:
            risk   = "WARNING"
            action = ("🟡 ПОПЕРЕДЖЕННЯ: сповістити оператора, "
                      "знизити навантаження на 20%, запланувати ТО.")
        else:
            risk   = "NORMAL"
            action = "🟢 Норма: продовжувати моніторинг."

        decision = AgentDecision(
            timestamp      = reading.timestamp,
            failure_prob   = round(prob, 4),
            lstm_forecast  = lstm_forecast,
            risk_level     = risk,
            action         = action,
            reasoning      = reasoning,
        )
        self.decisions.append(decision)
        return decision

    # 5. АДАПТАЦІЯ

    def adapt(self):
        """
        Порівнює прогнози з реальними результатами (post-hoc).
        Коригує пороги на основі false alarms і missed failures.

        Викликається після того, як стали відомі реальні результати
        (через метод mark_actual_outcome).
        """
        evaluated = [d for d in self.decisions if d.actual_failure is not None]
        if not evaluated:
            print("[Agent] Адаптація: немає оцінених рішень.")
            return

        for d in evaluated:
            predicted_failure = d.risk_level in ("WARNING", "CRITICAL")
            actual_failure    = bool(d.actual_failure)
            if predicted_failure and not actual_failure:
                self._false_alarms += 1
            if not predicted_failure and actual_failure:
                self._missed_failures += 1

        # Корекція порогів
        if self._false_alarms > self._missed_failures * 2:
            self.th_warn  = min(self.th_warn  + 0.02, 0.60)
            self.th_crit  = min(self.th_crit  + 0.02, 0.85)
            print(f"[Agent] Адаптація: забагато хибних тривог → "
                  f"пороги підвищено (warn={self.th_warn:.2f}, crit={self.th_crit:.2f})")
        elif self._missed_failures > self._false_alarms:
            self.th_warn  = max(self.th_warn  - 0.03, 0.25)
            self.th_crit  = max(self.th_crit  - 0.03, 0.50)
            print(f"[Agent] Адаптація: пропускаємо відмови → "
                  f"пороги знижено (warn={self.th_warn:.2f}, crit={self.th_crit:.2f})")
        else:
            print("[Agent] Адаптація: пороги залишаються без змін.")

    def mark_actual_outcome(self, step: int, actual: int):
        """Позначає реальний результат для рішення на кроці step."""
        if 0 <= step < len(self.decisions):
            self.decisions[step].actual_failure = actual

    # УТИЛІТИ

    def summary(self) -> dict:
        """Підсумкова статистика роботи агента."""
        counts = {"NORMAL": 0, "WARNING": 0, "CRITICAL": 0}
        for d in self.decisions:
            counts[d.risk_level] = counts.get(d.risk_level, 0) + 1
        return {
            "total_steps":       self._total_steps,
            "normal":            counts["NORMAL"],
            "warnings":          counts["WARNING"],
            "critical_alerts":   counts["CRITICAL"],
            "false_alarms":      self._false_alarms,
            "missed_failures":   self._missed_failures,
            "threshold_warning": self.th_warn,
            "threshold_critical":self.th_crit,
        }

    def save_log(self, path: str = "agent_log.json"):
        """Зберігає журнал рішень у JSON."""
        log = [
            {
                "timestamp":    d.timestamp,
                "failure_prob": d.failure_prob,
                "risk_level":   d.risk_level,
                "action":       d.action,
                "reasoning":    d.reasoning,
                "lstm_forecast":d.lstm_forecast,
            }
            for d in self.decisions
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"[Agent] Журнал збережено: {path}")
