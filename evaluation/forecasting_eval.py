import numpy as np
from sklearn.metrics import (average_precision_score, precision_score, recall_score, f1_score, confusion_matrix)

def get_threshold_from_train_scores(train_scores, quantile=0.99):
    if not (0 <= quantile <= 1):
        raise ValueError(
            f"quantile must be in [0, 1], got {quantile}. "
        )
    return float(np.quantile(train_scores, quantile))

def make_forecast_error_scores(y_true, y_pred):
    """
    Turn forecast errors into one scalar score per sample.

    For single-point forecasting, this becomes mean absolute error
    across features.

    For recursive H-step forecasting, this becomes mean absolute error
    across both horizon steps and features.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    errors = np.abs(y_true - y_pred)

    if errors.ndim == 1:
        return errors

    reduce_axes = tuple(range(1, errors.ndim))
    return errors.mean(axis=reduce_axes)

def evaluate_forecast_predictions(y_true, y_pred):
    """
    Evaluate raw forecast quality.

    Works for:
    - single-point forecasts:   (n_samples, n_features)
    - H-step path forecasts:    (n_samples, horizon, n_features)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    errors = y_true - y_pred

    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }

def make_future_horizon_labels(y, end_indices, horizon):
    """
    Build binary labels for future anomaly detection.

    For each sample ending at end_idx:
    label = 1 if trully an anomaly happens in the next `horizon` steps, else 0
    """
    y = np.asarray(y).astype(int)

    labels = []
    for end_idx in end_indices:
        future_slice = y[end_idx + 1 : end_idx + 1 + horizon]
        labels.append(int(np.any(future_slice == 1)))

    return np.asarray(labels, dtype=int)

def evaluate_anomaly_scores(y_true, y_scores, threshold):
    """
    Evaluate anomaly detections as if binary classification problem per prediction.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores)

    y_pred = (y_scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    false_alert_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    if np.any(y_true == 1):
        pr_auc = float(average_precision_score(y_true, y_scores))
    else:
        pr_auc = 0.0

    metrics = {
        "positive_rate": float(y_true.mean()),
        "pr_auc": pr_auc,
        "false_alert_rate": float(false_alert_rate),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,
    }
    return metrics

def print_forecast_metrics(name, metrics):
    print(f"\n{name} forecast metrics:")
    print(f"  MAE:                      {metrics['mae']:.6f}")
    print(f"  MSE:                      {metrics['mse']:.6f}")
    print(f"  RMSE:                     {metrics['rmse']:.6f}")

def print_anomaly_metrics(name, metrics):
    print(f"\n{name} anomaly metrics:")
    print(f"  Positive rate:            {metrics['positive_rate']:.6f}")
    print(f"  PR-AUC:                   {metrics['pr_auc']:.6f}")
    print(f"  False alert rate:         {metrics['false_alert_rate']:.6f}")
    print(f"  Precision @ threshold:    {metrics['precision_at_threshold']:.6f}")
    print(f"  Recall @ threshold:       {metrics['recall_at_threshold']:.6f}")
    print(f"  F1 @ threshold:           {metrics['f1_at_threshold']:.6f}")
    print("  Confusion matrix:")
    print(metrics["confusion_matrix"])