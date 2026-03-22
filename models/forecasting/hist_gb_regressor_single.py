import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

def train_hist_gb_regressor(X_train, y_train, learning_rate=0.05, max_iter=300, max_leaf_nodes=31, 
                            min_samples_leaf=20, l2_regularization=0.0,verbose=1):
    """
    Train the HistGradientBoostingRegressor for one-step multivariate forecasting.
    """
    y_train = np.asarray(y_train)

    base_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=False,
        random_state=42,
        verbose=verbose
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    return model

def prediction_hist_gb(model, X):
    """
    Predict one future multivariate point.
    """
    return model.predict(X)