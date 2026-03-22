from utils.load_data import drop_constant_features, load_machine_data, get_constant_features
from utils.windowing import make_forecasting_windows, flatten_windows
from models.forecasting.hist_gb_regressor_single import (train_hist_gb_regressor, prediction_hist_gb)
from models.forecasting.hist_gb_reg_recursive import recursive_forecast_hist_gb
from evaluation.forecasting_eval import (
    get_threshold_from_train_scores,
    make_forecast_error_scores,
    evaluate_forecast_predictions,
    make_future_horizon_labels,
    evaluate_anomaly_scores,
    print_forecast_metrics,
    print_anomaly_metrics,
)

def run_forecasting_single_machine(
    machine_id,
    model_name="hist_gb_single",
    window_size=60,
    horizon=1,
    threshold_quantile=0.99,
    base_dir="data/per_machine"
):
    print(f"\nLoading data for {machine_id}...")
    X_train, X_test, y_test = load_machine_data(machine_id, base_dir=base_dir)

    constant_features = get_constant_features(X_train)
    X_train_reduced = drop_constant_features(X_train, constant_features)
    X_test_reduced = drop_constant_features(X_test, constant_features)

    print(f"\nTraining model: {model_name}...")

    if model_name == "hist_gb_single":
        X_train_windows, y_train_future, train_end_indices = make_forecasting_windows(
            X_train_reduced,
            window_size=window_size,
            horizon=1
        )
        X_test_windows, y_test_future, test_end_indices = make_forecasting_windows(
            X_test_reduced,
            window_size=window_size,
            horizon=1
        )

        X_train_flat = flatten_windows(X_train_windows)
        X_test_flat = flatten_windows(X_test_windows)

        y_train_next = y_train_future[:, 0, :]
        y_test_next = y_test_future[:, 0, :]

        model = train_hist_gb_regressor(X_train_flat, y_train_next)
        train_pred = prediction_hist_gb(model, X_train_flat)
        test_pred = prediction_hist_gb(model, X_test_flat)

        train_forecast_metrics = evaluate_forecast_predictions(y_train_next, train_pred)
        test_forecast_metrics = evaluate_forecast_predictions(y_test_next, test_pred)

        train_scores = make_forecast_error_scores(y_train_next, train_pred)
        test_scores = make_forecast_error_scores(y_test_next, test_pred)

        aligned_y_test = make_future_horizon_labels(y_test, test_end_indices, horizon=1)

    elif model_name == "hist_gb_recursive_h":
        if horizon < 1:
            raise ValueError("horizon must be >= 1 for recursive forecasting.")

        # learned model is still one-step model (horizon == 1)
        X_train_fit_windows, y_train_fit_future, _ = make_forecasting_windows(
            X_train_reduced,
            window_size=window_size,
            horizon=1
        )
        X_train_fit_flat = flatten_windows(X_train_fit_windows)
        y_train_next = y_train_fit_future[:, 0, :]

        # learned model is still one-step model
        model = train_hist_gb_regressor(X_train_fit_flat, y_train_next)

        X_train_score_windows, y_train_future, train_end_indices = make_forecasting_windows(
            X_train_reduced,
            window_size=window_size,
            horizon=horizon
        )
        X_test_score_windows, y_test_future, test_end_indices = make_forecasting_windows(
            X_test_reduced,
            window_size=window_size,
            horizon=horizon
        )

        # recursively build H points
        train_pred_path = recursive_forecast_hist_gb(
            model,
            X_train_score_windows,
            horizon=horizon
        )
        test_pred_path = recursive_forecast_hist_gb(
            model,
            X_test_score_windows,
            horizon=horizon
        )

        train_forecast_metrics = evaluate_forecast_predictions(y_train_future, train_pred_path)
        test_forecast_metrics = evaluate_forecast_predictions(y_test_future, test_pred_path)

        train_scores = make_forecast_error_scores(y_train_future, train_pred_path)
        test_scores = make_forecast_error_scores(y_test_future, test_pred_path)

        aligned_y_test = make_future_horizon_labels(y_test, test_end_indices, horizon=horizon)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    threshold = get_threshold_from_train_scores(train_scores, quantile=threshold_quantile)

    print(f"\nEvaluating anomaly scores on chosen threshold...")
    anomaly_metrics = evaluate_anomaly_scores(aligned_y_test, test_scores, threshold)

    print(f"\nMachine: {machine_id}")
    print(f"Model: {model_name}")
    print(f"Dropped constant features: {constant_features}")
    print(f"Threshold choice based on train scores: {threshold:.6f}")

    print_forecast_metrics("Train", train_forecast_metrics)
    print_forecast_metrics("Test", test_forecast_metrics)
    print_anomaly_metrics("Test", anomaly_metrics)

    return {
        "machine_id": machine_id,
        "model_name": model_name,
        "window_size": window_size,
        "horizon": horizon,
        "threshold": threshold,
        "train_forecast_metrics": train_forecast_metrics,
        "test_forecast_metrics": test_forecast_metrics,
        "anomaly_metrics": anomaly_metrics,
        "train_scores": train_scores,
        "test_scores": test_scores,
        "aligned_y_test": aligned_y_test,
        "test_end_indices": test_end_indices,
    }