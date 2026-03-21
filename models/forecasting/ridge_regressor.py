import numpy as np
from sklearn.linear_model import Ridge

def train_ridge_regressor(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def prediction_ridge(model, X):
    return model.predict(X)

def err_scores_ridge(y_true, y_pred):
    errors = np.abs(y_true - y_pred)
    scores = errors.mean(axis=1)
    return scores