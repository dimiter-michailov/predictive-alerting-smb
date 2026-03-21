import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

def train_hist_gb_regressor(X_train, y_train, verbose=1):
    base_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        early_stopping=True,
        random_state=42,
        verbose=verbose
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    return model

def prediction_hist_gb(model, X):
    return model.predict(X)

def err_scores_hist(y_true, y_pred):
    errors = np.abs(y_true - y_pred)
    scores = errors.mean(axis=1)
    return scores