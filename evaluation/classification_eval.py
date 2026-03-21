import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_score,
                             recall_score, f1_score, confusion_matrix)

def evaluate_probabilities(y_true, y_proba, threshold=0.5):
    """
    Evaluate probability predictions for binary classification.

    metrics : PR-AUC, Brier score, false alert rate, and threshold-based metrics
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)

    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    false_alert_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "threshold": float(threshold),
        "positive_rate": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "false_alert_rate": float(false_alert_rate),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,
    }

    return metrics

def make_threshold_grid(y_proba):
    fixed_grid = np.linspace(0.05, 0.95, 19)
    quantile_grid = np.quantile(y_proba, np.linspace(0.10, 0.90, 17))
    threshold_grid = np.unique(np.clip(np.concatenate([fixed_grid, quantile_grid]), 0.0, 1.0))
    return threshold_grid

def find_best_threshold(y_true, y_proba, threshold_grid=None, f1_margin=1e-4):
    """
    Pick the best threshold on the validation set.

    The threshold is chosen by best F1 score.
    If F1 scores are very close, ties are broken by higher recall,
    then higher precision.
    """
    if threshold_grid is None:
        threshold_grid = make_threshold_grid(y_proba)

    best_threshold = None
    best_metrics = None

    for threshold in threshold_grid:
        metrics = evaluate_probabilities(y_true, y_proba, threshold=threshold)

        if best_metrics is None:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue

        current_f1 = metrics["f1_at_threshold"]
        best_f1 = best_metrics["f1_at_threshold"]

        if current_f1 > best_f1 + f1_margin:
            best_threshold = float(threshold)
            best_metrics = metrics
        elif abs(current_f1 - best_f1) <= f1_margin:
            current_key = (
                metrics["recall_at_threshold"],
                metrics["precision_at_threshold"],
            )
            best_key = (
                best_metrics["recall_at_threshold"],
                best_metrics["precision_at_threshold"],
            )

            if current_key > best_key:
                best_threshold = float(threshold)
                best_metrics = metrics

    return best_threshold, best_metrics

def print_metrics(name, metrics):
    """
    Print evaluation metrics.
    """
    print(f"\n{name} metrics:")
    if "threshold" in metrics:
        print(f"  Best threshold (F1-based):         {metrics['threshold']:.6f}")
    print(f"  Positive rate:            {metrics['positive_rate']:.6f}")
    print(f"  PR-AUC:                   {metrics['pr_auc']:.6f}")
    print(f"  Brier score:              {metrics['brier_score']:.6f}")
    print(f"  False alert rate:         {metrics['false_alert_rate']:.6f}")
    print(f"  Precision @ threshold:    {metrics['precision_at_threshold']:.6f}")
    print(f"  Recall @ threshold:       {metrics['recall_at_threshold']:.6f}")
    print(f"  F1 @ threshold:           {metrics['f1_at_threshold']:.6f}")
    print(f"  Confusion matrix:")
    print(metrics["confusion_matrix"])