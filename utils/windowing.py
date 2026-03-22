import numpy as np

def make_forecasting_windows(X, window_size, horizon=1):
    """
    Create sliding windows for forecasting.

    X_windows[i] contains the previous `window_size` steps.
    y_future[i] contains the next `horizon` future steps.

    For horizon=1:
    - X_windows shape: (n_samples, window_size, n_features)
    - y_future shape:  (n_samples, 1, n_features)

    For general horizon=H:
    - X_windows shape: (n_samples, window_size, n_features)
    - y_future shape:  (n_samples, horizon, n_features)
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must have shape (n_timesteps, n_features).")

    if window_size < 1:
        raise ValueError("window_size must be >= 1.")

    if horizon < 1:
        raise ValueError("horizon must be >= 1.")

    n_timesteps, n_features = X.shape
    n_samples = n_timesteps - window_size - horizon + 1

    if n_samples <= 0:
        raise ValueError(
            "Not enough timesteps to create forecasting windows. "
            "Decrease window_size or horizon."
        )

    X_windows = []
    y_future = []
    end_indices = []

    for start_idx in range(n_samples):
        end_idx = start_idx + window_size - 1
        future_start = end_idx + 1
        future_end = future_start + horizon

        X_windows.append(X[start_idx : start_idx + window_size])
        y_future.append(X[future_start : future_end])
        end_indices.append(end_idx)

    return (
        np.asarray(X_windows),
        np.asarray(y_future),
        np.asarray(end_indices)
    )

def make_classification_windows(X, y, window_size, horizon):
    """
    Build training samples for classification model.

    X has shape (n_time_steps, n_features) -- point-wise features
    y has shape (n_time_steps,) -- point-wise truth labels

    From a given time point 't':
        Input window: X[t - 'window_size' + 1 : t + 1] steps of 'N' features
        Future target: 'z_t' = 1 or 0, based on whether an anomaly occurrs in y[t + 1 : t + H] steps

    Returns:
        X_windows: array of shape (n_samples, window_size, n_features)
        y_windows: array of shape (n_samples,), aka truth labels for all n_samples
        end_indices: index of the last time point of each input window
    """
    X_windows = []
    y_windows = []
    end_indices = []

    n_time_steps = X.shape[0]

    first_t = window_size - 1
    last_t = n_time_steps - horizon - 1

    for t in range(first_t, last_t + 1):
        x_window = X[t - window_size + 1:t + 1]
        future_labels = y[t + 1:t + 1 + horizon]

        target = int(np.any(future_labels == 1))

        X_windows.append(x_window)
        y_windows.append(target)
        end_indices.append(t)

    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    end_indices = np.array(end_indices)

    return X_windows, y_windows, end_indices

def flatten_windows(X_windows):
    """
    Convert windows from 3D to 2D.

    Input:
    (n_samples, window_size, n_features)
    Returns:
    (n_samples, window_size * n_features)
    """
    n_samples = X_windows.shape[0]
    return X_windows.reshape(n_samples, -1)