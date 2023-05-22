from sklearn.model_selection import train_test_split

def split(X, y):
    X_train_cv, X_test, y_train_cv, y_test = train_test_split(X.values,
                                                          y.values,
                                                          test_size=0.2,
                                                          random_state=42,
                                                          stratify=y)
    return X_train_cv, X_test, y_train_cv, y_test