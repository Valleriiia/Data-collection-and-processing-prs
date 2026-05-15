"""
Генерація синтетичного датасету для predictive maintenance.
Базується на структурі реального датасету AI4I 2020 (UCI ML Repository).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_dataset(n_samples: int = 2000, seed: int = 1) -> pd.DataFrame:
    np.random.seed(seed)

    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i * 0.25) for i in range(n_samples)]

    # Базові показники — нормальний режим роботи
    temperature = np.random.normal(70, 8, n_samples)
    vibration   = np.random.normal(6,  1.5, n_samples)
    pressure    = np.random.normal(4.2, 0.3, n_samples)
    current     = np.random.normal(42, 4, n_samples)
    rpm         = np.random.normal(2980, 40, n_samples)
    runtime     = np.arange(n_samples) * 0.25  # накопичене напрацювання (год)

    failure = np.zeros(n_samples, dtype=int)

    # Симуляція 8 інцидентів деградації → відмова
    failure_events = [200, 450, 700, 900, 1150, 1400, 1650, 1900]
    for fe in failure_events:
        if fe >= n_samples:
            continue
        window = min(80, fe)
        for i in range(fe - window, fe):
            progress = (i - (fe - window)) / window  # 0→1
            temperature[i] += progress * 30 * np.random.uniform(0.8, 1.2)
            vibration[i]   += progress * 10 * np.random.uniform(0.8, 1.2)
            pressure[i]    -= progress * 0.8
            current[i]     += progress * 12
            rpm[i]         -= progress * 150

        # Сама відмова і 5 точок після неї
        end = min(fe + 5, n_samples)
        failure[fe:end] = 1

    # Кліпування в реалістичні межі
    temperature = np.clip(temperature, 40, 120)
    vibration   = np.clip(vibration,    0,  25)
    pressure    = np.clip(pressure,     2,   6)
    current     = np.clip(current,     20,  70)
    rpm         = np.clip(rpm,        2700, 3100)

    df = pd.DataFrame({
        "timestamp":   timestamps,
        "temperature": temperature.round(2),
        "vibration":   vibration.round(3),
        "pressure":    pressure.round(3),
        "current":     current.round(2),
        "rpm":         rpm.round(0).astype(int),
        "runtime_hours": runtime.round(2),
        "failure":     failure,
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    out = "data/sensor_data.csv"
    df.to_csv(out, index=False)
    total    = len(df)
    failures = df["failure"].sum()
    print(f"Датасет збережено: {out}")
    print(f"Записів: {total} | Відмов: {failures} ({failures/total*100:.1f}%)")
    print(df.head())
