import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss, precision_score,
                             recall_score, f1_score, confusion_matrix)

def evaluate_probabilities(y_true, y_proba, threshold=0.5):
    """
    Evaluate probability predictions for binary classification.

    metrics : PR-AUC, ROC-AUC, Brier score, and threshold-based metrics
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)

    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "positive_rate": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    return metrics

def print_metrics(name, metrics):
    """
    Print evaluation metrics.
    """
    print(f"\n{name} metrics:")
    print(f"  Positive rate:            {metrics['positive_rate']:.6f}")
    print(f"  PR-AUC:                   {metrics['pr_auc']:.6f}")
    print(f"  ROC-AUC:                  {metrics['roc_auc']:.6f}")
    print(f"  Brier score:              {metrics['brier_score']:.6f}")
    print(f"  Precision @ threshold:    {metrics['precision_at_threshold']:.6f}")
    print(f"  Recall @ threshold:       {metrics['recall_at_threshold']:.6f}")
    print(f"  F1 @ threshold:           {metrics['f1_at_threshold']:.6f}")
    print(f"  Confusion matrix:")
    print(metrics["confusion_matrix"])