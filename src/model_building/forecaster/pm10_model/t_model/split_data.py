def split_data(X, y, logger):
    logger.info("Splitting data chronologically...")

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    logger.info("Data split completed successfully.")

    return X_train, X_test, y_train, y_test