from utils.load_data import drop_constant_features, load_machine_data, get_constant_features
from utils.windowing import make_forecasting_windows, flatten_windows
from models.forecasting.hist_gb_regressor import (train_hist_gb_regressor, prediction_hist_gb, err_scores_hist)
from models.forecasting.ridge_regressor import (train_ridge_regressor, prediction_ridge, err_scores_ridge)
from evaluation.regression_eval import (get_threshold_from_train_scores, evaluate_anomaly_scores, print_metrics)

def run_forecasting_single_machine(machine_id, model_name="hist_gb", window_size=60, 
                                   threshold_quantile=0.99,base_dir="data/per_machine"):
    
    print(f"\nLoading data for {machine_id}...")
    X_train, X_test, y_test = load_machine_data(machine_id, base_dir=base_dir)

    constant_features = get_constant_features(X_train)
    X_train_reduced = drop_constant_features(X_train, constant_features)
    X_test_reduced = drop_constant_features(X_test, constant_features)

    print(f"\nBuilding forecasting windows...")
    X_train_windows, y_train_next, train_end_indices = make_forecasting_windows(
        X_train_reduced,
        window_size=window_size
    )
    X_test_windows, y_test_next, test_end_indices = make_forecasting_windows(
        X_test_reduced,
        window_size=window_size
    )

    X_train_flat = flatten_windows(X_train_windows)
    X_test_flat = flatten_windows(X_test_windows)

    print(f"\nTraining model: {model_name}...")
    if model_name == "hist_gb":
        model = train_hist_gb_regressor(X_train_flat, y_train_next)
        train_pred = prediction_hist_gb(model, X_train_flat)
        test_pred = prediction_hist_gb(model, X_test_flat)

        train_scores = err_scores_hist(y_train_next, train_pred)
        test_scores = err_scores_hist(y_test_next, test_pred)

    elif model_name == "ridge":
        model = train_ridge_regressor(X_train_flat, y_train_next)
        train_pred = prediction_ridge(model, X_train_flat)
        test_pred = prediction_ridge(model, X_test_flat)

        train_scores = err_scores_ridge(y_train_next, train_pred)
        test_scores = err_scores_ridge(y_test_next, test_pred)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    threshold = get_threshold_from_train_scores(train_scores, quantile=threshold_quantile)

    aligned_y_test = y_test[test_end_indices + 1]

    print(f"\nEvaluating anomaly prediction errors based on threshold...")
    metrics = evaluate_anomaly_scores(aligned_y_test, test_scores, threshold)

    print(f"\nMachine: {machine_id}")
    print(f"Model: {model_name}")
    print(f"Dropped constant features: {constant_features}")
    print(f"Threshold from train scores: {threshold:.6f}")
    print_metrics("Test", metrics)

    return {
        "machine_id": machine_id,
        "model_name": model_name,
        "threshold": threshold,
        "metrics": metrics,
        "train_scores": train_scores,
        "test_scores": test_scores,
        "aligned_y_test": aligned_y_test,
        "test_end_indices": test_end_indices,
    }