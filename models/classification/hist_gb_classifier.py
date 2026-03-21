import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

def make_sample_weights(y):
    """
    Give higher weight to positive samples based on class imbalance.
    """
    # Initialize vector as [1..1]
    weights = np.ones(len(y), dtype=float)

    # count positives/negatives
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)

    if n_pos > 0:
        pos_weight = n_neg / n_pos
        weights[y == 1] = pos_weight

    return weights

def train_hist_gb_classifier(X_train, y_train, learning_rate=0.05, max_iter=300,
                             max_leaf_nodes=31, min_samples_leaf=20, verbose=1):
    """
    Train the HistGradientBoostingClassifier.
    """
    sample_weights = make_sample_weights(y_train)

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        early_stopping=False,
        random_state=42,
        verbose=verbose
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model

def predict_probabilities(model, X):
    """
    Return probability of the positive class.
    """
    return model.predict_proba(X)[:, 1]