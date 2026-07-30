from sklearn.model_selection import train_test_split

def split_data(df, logger, test_size=0.2, random_state=42):
    X = df.drop(columns=['pm25'])
    y = df['pm25']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test