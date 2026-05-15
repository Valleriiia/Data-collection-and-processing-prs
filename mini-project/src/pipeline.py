"""
Pipeline обробки даних:
  1. Завантаження CSV
  2. Очищення (пропуски, аномалії)
  3. Feature Engineering (ковзні вікна, rate-of-change)
  4. Нормалізація
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "temperature", "vibration", "pressure",
    "current", "rpm", "runtime_hours",
]

WINDOW = 12  # 12 × 15 хв = 3 год


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"[Pipeline] Завантажено {len(df)} записів з {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Пропущені значення → медіана по колонці
    for col in FEATURE_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Видалення дублікатів
    df = df.drop_duplicates(subset=["timestamp"])

    # Фільтрація викидів методом IQR для кожної ознаки
    for col in FEATURE_COLS:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q1, upper=q3)

    after = len(df)
    print(f"[Pipeline] Очищення: {before} → {after} записів")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Додає статистичні ознаки на ковзному вікні."""
    for col in ["temperature", "vibration", "pressure", "current"]:
        # Ковзне середнє
        df[f"{col}_mean{WINDOW}"] = (
            df[col].rolling(window=WINDOW, min_periods=1).mean().round(3)
        )
        # Стандартне відхилення (нестабільність)
        df[f"{col}_std{WINDOW}"] = (
            df[col].rolling(window=WINDOW, min_periods=1).std().fillna(0).round(3)
        )
        # Rate-of-change (швидкість змін)
        df[f"{col}_roc"] = df[col].diff().fillna(0).round(3)

    # RMS вібрації (найважливіша ознака)
    df["vibration_rms"] = (
        (df["vibration"] ** 2).rolling(window=WINDOW, min_periods=1).mean().apply(np.sqrt).round(3)
    )

    print(f"[Pipeline] Feature engineering: {len(df.columns)} колонок")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Повертає список всіх ознак (без timestamp і target)."""
    exclude = {"timestamp", "failure"}
    return [c for c in df.columns if c not in exclude]


def split_data(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    """Розділення train/val/test без перемішування (часовий ряд!)."""
    n = len(df)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def scale_features(train, val, test, feature_cols):
    """StandardScaler, навчений тільки на train."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_val   = scaler.transform(val[feature_cols])
    X_test  = scaler.transform(test[feature_cols])
    return X_train, X_val, X_test, scaler


def run_pipeline(path: str):
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)
    train, val, test = split_data(df)
    X_train, X_val, X_test, scaler = scale_features(train, val, test, feature_cols)
    y_train = train["failure"].values
    y_val   = val["failure"].values
    y_test  = test["failure"].values
    print(f"[Pipeline] Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return {
        "df": df,
        "train": train, "val": val, "test": test,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }
