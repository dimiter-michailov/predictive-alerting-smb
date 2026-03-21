import numpy as np
from utils.load_data import load_machine_data, drop_constant_features, get_constant_features
from utils.chrono_splitting import chronological_split
from utils.windowing import make_classification_windows, flatten_windows


def prepare_classifier_machine_data(machine_id, window_size, horizon, base_dir="data/per_machine",
                                    drop_constants=True):
    print(f"\nLoading data for {machine_id}...")
    _, X_test, y_test = load_machine_data(machine_id, base_dir=base_dir)

    constant_features = []
    X_used = X_test

    if drop_constants:
        constant_features = get_constant_features(X_test)
        X_used = drop_constant_features(X_test, constant_features)

    X_windows, y_windows, end_indices = make_classification_windows(
        X_used,
        y_test,
        window_size=window_size,
        horizon=horizon
    )

    (
        X_train_w, y_train_w, end_train,
        X_val_w, y_val_w, end_val,
        X_holdout_w, y_holdout_w, end_holdout
    ) = chronological_split(
        X_windows,
        y_windows,
        end_indices,
        gap=horizon,
    )

    X_train_flat = flatten_windows(X_train_w)
    X_val_flat = flatten_windows(X_val_w)
    X_holdout_flat = flatten_windows(X_holdout_w)

    return {
        "machine_id": machine_id,
        "constant_features": constant_features,
        "X_train_flat": X_train_flat,
        "y_train_w": y_train_w,
        "X_val_flat": X_val_flat,
        "y_val_w": y_val_w,
        "X_holdout_flat": X_holdout_flat,
        "y_holdout_w": y_holdout_w,
    }


def pool_prepared_machines(prepared_machines):
    X_train_pool = np.vstack([item["X_train_flat"] for item in prepared_machines])
    y_train_pool = np.concatenate([item["y_train_w"] for item in prepared_machines])

    X_val_pool = np.vstack([item["X_val_flat"] for item in prepared_machines])
    y_val_pool = np.concatenate([item["y_val_w"] for item in prepared_machines])

    X_holdout_pool = np.vstack([item["X_holdout_flat"] for item in prepared_machines])
    y_holdout_pool = np.concatenate([item["y_holdout_w"] for item in prepared_machines])

    return {
        "X_train_pool": X_train_pool,
        "y_train_pool": y_train_pool,
        "X_val_pool": X_val_pool,
        "y_val_pool": y_val_pool,
        "X_holdout_pool": X_holdout_pool,
        "y_holdout_pool": y_holdout_pool,
    }