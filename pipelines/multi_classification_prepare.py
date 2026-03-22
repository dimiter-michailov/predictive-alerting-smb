import numpy as np
from utils.load_data import load_machine_data, drop_constant_features, get_constant_features
from utils.chrono_splitting import chronological_split
from utils.windowing import make_classification_windows, flatten_windows

def _has_both_classes(y):
    return len(np.unique(y)) == 2

def prepare_classifier_machine_data(machine_id, window_size, horizon, base_dir="data/per_machine",
                                    drop_constants=True):
    print(f"\nLoading data for {machine_id}...")
    _, X_test, y_test = load_machine_data(machine_id, base_dir=base_dir)

    constant_features = get_constant_features(X_test)
    X_used = X_test

    if drop_constants:
        X_used = drop_constant_features(X_test, constant_features)

    X_windows, y_windows, end_indices = make_classification_windows(X_used, y_test, window_size=window_size, horizon=horizon)

    (
        X_train_w, y_train_w, end_train,
        X_val_w, y_val_w, end_val,
        X_holdout_w, y_holdout_w, end_holdout
    ) = chronological_split(X_windows, y_windows, end_indices, gap=max(window_size, horizon))

    X_train_flat = flatten_windows(X_train_w)
    X_val_flat = flatten_windows(X_val_w)
    X_holdout_flat = flatten_windows(X_holdout_w)

    return {
        "machine_id": machine_id,
        "constant_features": constant_features,
        "n_features_original": X_test.shape[1],
        "X_train_flat": X_train_flat,
        "y_train_w": y_train_w,
        "X_val_flat": X_val_flat,
        "y_val_w": y_val_w,
        "X_holdout_flat": X_holdout_flat,
        "y_holdout_w": y_holdout_w,
        "train_has_both_classes": _has_both_classes(y_train_w),
        "val_has_both_classes": _has_both_classes(y_val_w),
        "holdout_has_both_classes": _has_both_classes(y_holdout_w),
    }

def _drop_flat_constant_features(X_flat, window_size, n_features_original, constant_features):
    if not constant_features:
        return X_flat

    X_windows = X_flat.reshape(len(X_flat), window_size, n_features_original)
    X_windows = np.delete(X_windows, constant_features, axis=2)
    return flatten_windows(X_windows)

def apply_shared_constant_feature_drop(prepared_machines, window_size):
    if not prepared_machines:
        return prepared_machines, []

    shared_constant_features = set(prepared_machines[0]["constant_features"])
    for item in prepared_machines[1:]:
        shared_constant_features &= set(item["constant_features"])

    shared_constant_features = sorted(shared_constant_features)

    if not shared_constant_features:
        return prepared_machines, shared_constant_features

    updated_prepared_machines = []

    for item in prepared_machines:
        updated_item = dict(item)

        updated_item["X_train_flat"] = _drop_flat_constant_features(
            item["X_train_flat"],
            window_size,
            item["n_features_original"],
            shared_constant_features,
        )
        updated_item["X_val_flat"] = _drop_flat_constant_features(
            item["X_val_flat"],
            window_size,
            item["n_features_original"],
            shared_constant_features,
        )
        updated_item["X_holdout_flat"] = _drop_flat_constant_features(
            item["X_holdout_flat"],
            window_size,
            item["n_features_original"],
            shared_constant_features,
        )

        updated_prepared_machines.append(updated_item)

    return updated_prepared_machines, shared_constant_features

def _append_machine_indicator(X, machine_index, n_machines):
    indicator = np.zeros((X.shape[0], n_machines), dtype=X.dtype)
    indicator[:, machine_index] = 1.0
    return np.hstack([X, indicator])

def pool_prepared_machines(prepared_machines, add_machine_indicators=False):
    X_train_parts = []
    y_train_parts = []
    X_val_parts = []
    y_val_parts = []
    X_holdout_parts = []
    y_holdout_parts = []

    n_machines = len(prepared_machines)

    for machine_index, item in enumerate(prepared_machines):
        X_train = item["X_train_flat"]
        X_val = item["X_val_flat"]
        X_holdout = item["X_holdout_flat"]

        if add_machine_indicators:
            X_train = _append_machine_indicator(X_train, machine_index, n_machines)
            X_val = _append_machine_indicator(X_val, machine_index, n_machines)
            X_holdout = _append_machine_indicator(X_holdout, machine_index, n_machines)

        X_train_parts.append(X_train)
        y_train_parts.append(item["y_train_w"])

        X_val_parts.append(X_val)
        y_val_parts.append(item["y_val_w"])

        X_holdout_parts.append(X_holdout)
        y_holdout_parts.append(item["y_holdout_w"])

    X_train_pool = np.vstack(X_train_parts)
    y_train_pool = np.concatenate(y_train_parts)

    X_val_pool = np.vstack(X_val_parts)
    y_val_pool = np.concatenate(y_val_parts)

    X_holdout_pool = np.vstack(X_holdout_parts)
    y_holdout_pool = np.concatenate(y_holdout_parts)

    return {
        "X_train_pool": X_train_pool,
        "y_train_pool": y_train_pool,
        "X_val_pool": X_val_pool,
        "y_val_pool": y_val_pool,
        "X_holdout_pool": X_holdout_pool,
        "y_holdout_pool": y_holdout_pool,
    }