import numpy as np

def recursive_forecast_hist_gb(model, X_windows, horizon):
    """
    Predict an H-step future path recursively.

    Parameters
    model : trained one-step forecaster
    X_windows : shape (n_samples, window_size, n_features)
    horizon : int

    Returns 
    forecasts : shape (n_samples, horizon, n_features)
    """
    X_windows = np.asarray(X_windows)

    if X_windows.ndim != 3:
        raise ValueError("X_windows shape doesn't match.")

    n_samples, window_size, n_features = X_windows.shape
    forecasts = np.zeros((n_samples, horizon, n_features), dtype=float)

    rolling_windows = X_windows.copy()

    for step in range(horizon):
        X_flat = rolling_windows.reshape(n_samples, window_size * n_features)
        next_pred = model.predict(X_flat)

        forecasts[:, step, :] = next_pred

        if step < horizon - 1:
            rolling_windows = np.concatenate(
                [rolling_windows[:, 1:, :], next_pred[:, None, :]],
                axis=1
            )
    return forecasts