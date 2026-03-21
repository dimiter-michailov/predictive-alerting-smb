import pickle
import pandas as pd

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_machine_data(machine_id, base_dir="data/per_machine"):
    X_train = load_pickle(f"{base_dir}/{machine_id}_train.pkl")
    X_test = load_pickle(f"{base_dir}/{machine_id}_test.pkl")
    y_test = load_pickle(f"{base_dir}/{machine_id}_test_label.pkl")
    return X_train, X_test, y_test

def get_constant_features(X):
    """
    Get constant columns from X
    """
    X_df = pd.DataFrame(X)
    std_per_feature = X_df.std()

    constant_features = std_per_feature[std_per_feature == 0].index.tolist()
    return constant_features

def drop_constant_features(X, constant_features):
    """
    Drop constant columns on X
    """
    X_reduced = pd.DataFrame(X).drop(columns=constant_features).values
    return X_reduced

