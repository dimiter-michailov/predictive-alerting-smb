from utils.load_data import load_machine_data, drop_constant_features, get_constant_features
from utils.chrono_splitting import chronological_split
from utils.windowing import make_classification_windows, flatten_windows

from models.classification.hist_gb_classifier import train_hist_gb_classifier, predict_probabilities
from evaluation.classification_eval import evaluate_probabilities, print_metrics


def run_classifier_single_machine(machine_id, window_size=60, horizon=30, threshold=0.5, base_dir="data/per_machine"):
    X_train, X_test, y_test = load_machine_data(machine_id, base_dir=base_dir)

    constant_features = get_constant_features(X_test)
    X_test_reduced = drop_constant_features(X_test, constant_features)

    X_windows, y_windows, end_indices = make_classification_windows(
        X_test_reduced,
        y_test,
        window_size=window_size,
        horizon=horizon
    )

    (   X_train_w, y_train_w, end_train,
        X_val_w, y_val_w, end_val,
        X_holdout_w, y_holdout_w, end_holdout
    ) = chronological_split(X_windows, y_windows, end_indices)

    X_train_flat = flatten_windows(X_train_w)
    X_val_flat = flatten_windows(X_val_w)
    X_holdout_flat = flatten_windows(X_holdout_w)

    model = train_hist_gb_classifier(X_train_flat, y_train_w)

    val_proba = predict_probabilities(model, X_val_flat)
    holdout_proba = predict_probabilities(model, X_holdout_flat)

    val_metrics = evaluate_probabilities(y_val_w, val_proba, threshold=threshold)
    holdout_metrics = evaluate_probabilities(y_holdout_w, holdout_proba, threshold=threshold)

    print(f"\nMachine: {machine_id}")
    print("Model: hist_gb_classifier")
    print(f"Dropped constant features: {constant_features}")
    print_metrics("Validation", val_metrics)
    print_metrics("Holdout", holdout_metrics)

    return {
        "machine_id": machine_id,
        "val_metrics": val_metrics,
        "holdout_metrics": holdout_metrics,
        "val_proba": val_proba,
        "holdout_proba": holdout_proba,
    }