"""
Моделі для прогнозування відмов:
  - RandomForestClassifier (класифікація: відмова / норма)
  - LSTMPredictor (прогноз часового ряду на N кроків)

Для LSTM використовується тільки numpy (без tensorflow),
щоб не вимагати важких залежностей для демонстрації.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)


# 1. Random Forest — класифікація відмов
class FailureClassifier:
    """
    Обгортка над RandomForestClassifier.
    Навчається на (X, y) і повертає P(failure) для нових точок.
    """

    def __init__(self, n_estimators: int = 200, class_weight="balanced"):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_leaf=5,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )
        self.feature_cols: list = []
        self.is_trained = False

    def train(self, X_train, y_train, feature_cols: list):
        self.feature_cols = feature_cols
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("[RF] Модель навчена.")

    def predict_proba(self, X) -> np.ndarray:
        """Повертає ймовірність класу 1 (відмова)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X, y, label: str = "Test") -> dict:
        proba = self.predict_proba(X)
        pred  = (proba >= 0.5).astype(int)
        metrics = {
            "accuracy":  round(accuracy_score(y, pred),  4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall":    round(recall_score(y, pred, zero_division=0),    4),
            "f1":        round(f1_score(y, pred, zero_division=0),        4),
            "roc_auc":   round(roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.0, 4),
        }
        cm = confusion_matrix(y, pred)
        print(f"\n[RF] Метрики ({label}):")
        for k, v in metrics.items():
            print(f"  {k:10s}: {v}")
        print(f"  Матриця плутанини:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")
        return metrics

    def feature_importance(self) -> list:
        """Список (назва_ознаки, важливість) відсортований за спаданням."""
        pairs = sorted(
            zip(self.feature_cols, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        return pairs


# 2. LSTM — прогноз часового ряду
class LSTMPredictor:
    """
    Спрощений LSTM-прогнозувальник на numpy.
    Реальний проєкт використовував би tensorflow.keras або pytorch.

    Прогнозує наступне значення температури/вібрації
    на основі вікна `window_size` попередніх значень.
    """

    def __init__(self, window_size: int = 24, hidden_size: int = 32):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.is_trained  = False
        rng = np.random.default_rng(42)
        # Спрощені ваги (у реальному LSTM навчаються backprop through time)
        self.Wx = rng.normal(0, 0.1, (hidden_size, 1))
        self.Wh = rng.normal(0, 0.1, (hidden_size, hidden_size))
        self.bh = np.zeros(hidden_size)
        self.Wy = rng.normal(0, 0.1, (1, hidden_size))
        self.by = np.zeros(1)

    def _normalize(self, series: np.ndarray):
        mu, sigma = series.mean(), series.std() + 1e-8
        return (series - mu) / sigma, mu, sigma

    def train(self, series: np.ndarray, epochs: int = 30, lr: float = 0.001):
        """Навчання на одновимірному часовому ряді (наприклад, температура)."""
        norm, self._mu, self._sigma = self._normalize(series)
        X, y = [], []
        for i in range(len(norm) - self.window_size):
            X.append(norm[i: i + self.window_size])
            y.append(norm[i + self.window_size])
        X, y = np.array(X), np.array(y)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xi, yi in zip(X, y):
                # Forward pass (спрощений RNN-крок)
                h = np.zeros(self.hidden_size)
                for t in range(self.window_size):
                    h = np.tanh(self.Wx @ [[xi[t]]] + self.Wh @ h.reshape(-1, 1) + self.bh.reshape(-1, 1)).flatten()
                out = (self.Wy @ h.reshape(-1, 1) + self.by).item()
                loss = (out - yi) ** 2
                epoch_loss += loss

                # Backward pass (градієнт виходу)
                d_out = 2 * (out - yi)
                self.Wy -= lr * d_out * h.reshape(1, -1)
                self.by -= lr * d_out

            losses.append(epoch_loss / len(X))

        self.is_trained = True
        print(f"[LSTM] Навчено {epochs} епох. Фінальний MSE: {losses[-1]:.4f}")

    def predict_next(self, window: np.ndarray) -> float:
        """Прогнозує одне наступне значення (денормалізоване)."""
        norm = (window - self._mu) / (self._sigma + 1e-8)
        h = np.zeros(self.hidden_size)
        for t in range(len(norm)):
            h = np.tanh(self.Wx @ [[norm[t]]] + self.Wh @ h.reshape(-1, 1) + self.bh.reshape(-1, 1)).flatten()
        out = (self.Wy @ h.reshape(-1, 1) + self.by).item()
        return round(float(out * self._sigma + self._mu), 2)

    def predict_horizon(self, last_window: np.ndarray, horizon: int = 4) -> list:
        """Прогнозує `horizon` кроків вперед (авторегресивно)."""
        window = list(last_window[-self.window_size:])
        preds  = []
        for _ in range(horizon):
            nxt = self.predict_next(np.array(window))
            preds.append(nxt)
            window.append(nxt)
            window.pop(0)
        return preds

    def evaluate_rmse(self, series: np.ndarray) -> float:
        norm, mu, sigma = self._normalize(series)
        errors = []
        for i in range(self.window_size, len(norm)):
            window = series[i - self.window_size: i]
            pred   = self.predict_next(window)
            errors.append((pred - series[i]) ** 2)
        rmse = round(float(np.sqrt(np.mean(errors))), 4)
        print(f"[LSTM] RMSE на тестових даних: {rmse}")
        return rmse
