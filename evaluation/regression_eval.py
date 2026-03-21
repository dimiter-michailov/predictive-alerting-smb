import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

def get_threshold_from_train_scores(train_scores, quantile=0.99):
    if not (0 <= quantile <= 1):
        raise ValueError(
            f"quantile must be in [0, 1], got {quantile}. "
        )
    return float(np.quantile(train_scores, quantile))

def evaluate_anomaly_scores(y_true, y_scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores)

    y_pred = (y_scores >= threshold).astype(int)

    metrics = {
        "positive_rate": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, y_scores)),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    return metrics

def print_metrics(name, metrics):
    print(f"\n{name} metrics:")
    print(f"  Positive rate:            {metrics['positive_rate']:.6f}")
    print(f"  PR-AUC:                   {metrics['pr_auc']:.6f}")
    print(f"  ROC-AUC:                  {metrics['roc_auc']:.6f}")
    print(f"  Precision @ threshold:    {metrics['precision_at_threshold']:.6f}")
    print(f"  Recall @ threshold:       {metrics['recall_at_threshold']:.6f}")
    print(f"  F1 @ threshold:           {metrics['f1_at_threshold']:.6f}")
    print("  Confusion matrix:")
    print(metrics["confusion_matrix"])