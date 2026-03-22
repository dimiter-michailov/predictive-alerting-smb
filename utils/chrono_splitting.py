def chronological_split(X, y, end_indices, train_frac=0.7, val_frac=0.15, gap=0):
    """
    Split the samples by chronological time order into train, val and test.

    Used for only for the classification model, itself only used for comparison purposes.
    """
    n_samples = len(X)

    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))

    val_start = train_end + gap
    test_start = val_end + gap

    if val_start > val_end:
        # gap is based on the chosen horizon H
        raise ValueError("\nGap is too large: validation split invalid.")

    if test_start > n_samples:
        # gap is based on the chosen horizon H
        raise ValueError("\nGap is too large: test split invalid.")

    X_train = X[:train_end]
    y_train = y[:train_end]
    end_train = end_indices[:train_end]

    X_val = X[val_start:val_end]
    y_val = y[val_start:val_end]
    end_val = end_indices[val_start:val_end]

    X_test = X[test_start:]
    y_test = y[test_start:]
    end_test = end_indices[test_start:]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("\nEmpty chronological split.")

    return (
        X_train, y_train, end_train,
        X_val, y_val, end_val,
        X_test, y_test, end_test
    )