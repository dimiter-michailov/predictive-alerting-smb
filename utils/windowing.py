import numpy as np

def make_forecasting_windows(X, window_size):
    """
    Main SMD path:
    Build windows for forecasting model (use past window data to predict one future point's value).

    Returns:
        X_windows: (n_samples, window_size, n_features)
        y_next:    (n_samples, n_features)
        end_idx:   (n_samples,)
    """
    X_windows = []
    y_next = []
    end_indices = []

    n_time_steps = X.shape[0]

    first_t = window_size - 1
    last_t = n_time_steps - 2

    for t in range(first_t, last_t + 1):
        x_window = X[t - window_size + 1:t + 1]
        next_point = X[t + 1]

        X_windows.append(x_window)
        y_next.append(next_point)
        end_indices.append(t)

    X_windows = np.array(X_windows)
    y_next = np.array(y_next)
    end_indices = np.array(end_indices)

    return X_windows, y_next, end_indices

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