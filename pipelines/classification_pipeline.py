import numpy as np
from models.classification.hist_gb_classifier import predict_probabilities
from evaluation.classification_eval import evaluate_probabilities, print_metrics
from evaluation.scoreboard_logger import log_classifier_holdout, log_classifier_multi_holdout
from evaluation.classification_metadata import (build_classifier_metadata_text, build_classifier_multi_metadata_text)
from pipelines.multi_classification_prepare import (
    prepare_classifier_machine_data,
    apply_shared_constant_feature_drop,
    pool_prepared_machines,
)
from pipelines.classification_search import run_staged_classifier_search

def run_classifier_single_machine(machine_id, window_size=60, horizon=30, base_dir="data/per_machine"):
    prepared = prepare_classifier_machine_data(
        machine_id=machine_id,
        window_size=window_size,
        horizon=horizon,
        base_dir=base_dir,
        drop_constants=True,
    )

    if not prepared["train_has_both_classes"]:
        raise ValueError(f"{machine_id}: train split has one class")

    if not prepared["val_has_both_classes"]:
        raise ValueError(f"{machine_id}: val split has one class")

    best_result = run_staged_classifier_search(
        prepared["X_train_flat"],
        prepared["y_train_w"],
        prepared["X_val_flat"],
        prepared["y_val_w"],
    )

    best_model = best_result["model"]
    best_params = best_result["params"]
    best_threshold = best_result["threshold"]
    best_val_metrics = best_result["val_metrics"]

    holdout_proba = predict_probabilities(best_model, prepared["X_holdout_flat"])
    holdout_metrics = evaluate_probabilities(
        prepared["y_holdout_w"],
        holdout_proba,
        threshold=best_threshold
    )

    print(f"\nMachine: {machine_id}")
    print("Model: hist_gb_classifier")
    print(f"Dropped constant features: {prepared['constant_features']}")
    print(f"Chosen hyperparameters: {best_params}")
    print(f"Chosen threshold from validation: {best_threshold:.6f}")
    print_metrics("Validation", best_val_metrics)
    print_metrics("Holdout", holdout_metrics)

    metadata_text = build_classifier_metadata_text(
        machine_id=machine_id,
        window_size=window_size,
        horizon=horizon,
        best_params=best_params,
        best_threshold=best_threshold,
        val_metrics=best_val_metrics,
    )

    metadata_file = log_classifier_holdout(
        machine_id=machine_id,
        model_name="hist_gb_classifier",
        window_size=window_size,
        horizon=horizon,
        threshold=best_threshold,
        holdout_metrics=holdout_metrics,
        metadata_text=metadata_text,
        csv_path="scoreboard.csv",
    )

    return {
        "machine_id": machine_id,
        "model_name": "hist_gb_classifier",
        "best_params": best_params,
        "best_threshold": best_threshold,
        "val_metrics": best_val_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_proba": holdout_proba,
        "metadata_file": metadata_file,
    }

def run_classifier_multi_machine(machine_ids, window_size=60, horizon=30, base_dir="data/per_machine",
                                 drop_shared_constants=True, add_machine_indicators=True):
    prepared_machines = []
    skipped_machines = []

    for machine_id in machine_ids:
        prepared = prepare_classifier_machine_data(
            machine_id=machine_id,
            window_size=window_size,
            horizon=horizon,
            base_dir=base_dir,
            drop_constants=False,
        )

        if not prepared["train_has_both_classes"] or not prepared["val_has_both_classes"]:
            print(f"Skipping {machine_id}")
            skipped_machines.append(machine_id)
            continue

        prepared_machines.append(prepared)

    if not prepared_machines:
        raise ValueError("No valid machines in pool")

    shared_constant_features = []
    if drop_shared_constants:
        prepared_machines, shared_constant_features = apply_shared_constant_feature_drop(
            prepared_machines,
            window_size=window_size,
        )

    pooled = pool_prepared_machines(
        prepared_machines,
        add_machine_indicators=add_machine_indicators,
    )

    if len(np.unique(pooled["y_train_pool"])) < 2:
        raise ValueError("Pooled train has one class")

    if len(np.unique(pooled["y_val_pool"])) < 2:
        raise ValueError("Pooled val has one class")

    best_result = run_staged_classifier_search(
        pooled["X_train_pool"],
        pooled["y_train_pool"],
        pooled["X_val_pool"],
        pooled["y_val_pool"],
    )

    best_model = best_result["model"]
    best_params = best_result["params"]
    best_threshold = best_result["threshold"]
    best_val_metrics = best_result["val_metrics"]

    holdout_proba = predict_probabilities(best_model, pooled["X_holdout_pool"])
    holdout_metrics = evaluate_probabilities(
        pooled["y_holdout_pool"],
        holdout_proba,
        threshold=best_threshold
    )

    used_machine_ids = [item["machine_id"] for item in prepared_machines]

    print("\nMulti-machine run")
    print(f"Machines: {used_machine_ids}")
    if skipped_machines:
        print(f"Skipped machines: {skipped_machines}")
    print("Model: hist_gb_classifier")
    print(f"Dropped shared constant features: {shared_constant_features}")
    print(f"Added machine indicators: {add_machine_indicators}")
    print(f"Chosen hyperparameters: {best_params}")
    print(f"Chosen threshold from pooled validation: {best_threshold:.6f}")
    print_metrics("Pooled validation", best_val_metrics)
    print_metrics("Pooled holdout", holdout_metrics)

    start = 0
    for prepared in prepared_machines:
        machine_id = prepared["machine_id"]
        n_machine_holdout = len(prepared["y_holdout_w"])
        machine_proba = holdout_proba[start:start + n_machine_holdout]
        machine_metrics = evaluate_probabilities(
            prepared["y_holdout_w"],
            machine_proba,
            threshold=best_threshold
        )

        print(f"\nPer-machine holdout for {machine_id}")
        print_metrics("Holdout", machine_metrics)

        start += n_machine_holdout

    metadata_text = build_classifier_multi_metadata_text(
        machine_ids=used_machine_ids,
        window_size=window_size,
        horizon=horizon,
        best_params=best_params,
        best_threshold=best_threshold,
        val_metrics=best_val_metrics,
    )

    metadata_file = log_classifier_multi_holdout(
        machine_ids=used_machine_ids,
        model_name="hist_gb_classifier",
        window_size=window_size,
        horizon=horizon,
        threshold=best_threshold,
        holdout_metrics=holdout_metrics,
        metadata_text=metadata_text,
        csv_path="scoreboard_multi_summary.csv",
    )

    return {
        "machine_ids": used_machine_ids,
        "skipped_machines": skipped_machines,
        "model_name": "hist_gb_classifier",
        "best_params": best_params,
        "best_threshold": best_threshold,
        "val_metrics": best_val_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_proba": holdout_proba,
        "metadata_file": metadata_file,
    }