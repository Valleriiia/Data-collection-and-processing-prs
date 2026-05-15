"""
Метрики якості агента та моделей.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)


def classification_report(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    metrics.update({"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)})
    return metrics


def print_metrics(metrics: dict, title: str = "Метрики"):
    print(f"\n  {title}")
    labels = {
        "accuracy": "Accuracy ", "precision": "Precision",
        "recall":   "Recall   ", "f1":        "F1-score ",
        "roc_auc":  "ROC-AUC  ",
    }
    for k, label in labels.items():
        if k in metrics:
            print(f"  {label}: {metrics[k]:.4f}")
    if "TP" in metrics:
        print(f"  Матриця плутанини:")
        print(f"    TP={metrics['TP']:4d}  FP={metrics['FP']:4d}")
        print(f"    FN={metrics['FN']:4d}  TN={metrics['TN']:4d}")


def agent_performance(decisions: list, actuals: list) -> dict:
    """
    Оцінює ефективність агента (рівень CRITICAL = прогноз відмови).
    decisions : список AgentDecision
    actuals   : список реальних міток (0/1)
    """
    y_pred = [1 if d.risk_level == "CRITICAL" else 0 for d in decisions]
    y_proba = [d.failure_prob for d in decisions]
    n = min(len(y_pred), len(actuals))
    return classification_report(actuals[:n], y_pred[:n], np.array(y_proba[:n]))
