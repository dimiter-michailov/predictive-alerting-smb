def chronological_split(X, y, end_indices, train_frac=0.7, val_frac=0.15):
    """
    Split the samples by chronological time order into train, val and test.

    Used for only for the classification model, itself only used for comparison purposes.
    """
    n_samples = len(X)

    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))

    X_train = X[:train_end]
    y_train = y[:train_end]
    end_train = end_indices[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    end_val = end_indices[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]
    end_test = end_indices[val_end:]

    return (
        X_train, y_train, end_train,
        X_val, y_val, end_val,
        X_test, y_test, end_test
    )