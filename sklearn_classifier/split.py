import numpy as np
from sklearn.model_selection import train_test_split

def prep_data(data, features=None, target=None):
    """

    :param data: pandas data frame of feature matrix
    :param features: list of names for feature columns. None if all except last column
    :param target: name of target column. None if last column
    :return: numpy 2D array of features, transposed 1D array of labels
    """
    if features:
        _X = data[features]
    else:
        _X = data[:,:-1]
    if target:
        _y = data[target]
    else:
        _y = data[:,-1]
    X = np.array(_X)
    y = np.array(_y)
    return X, y


def simple_split(X, y, test_size, by_era=True):
    if by_era:
        pass
    else:
        X_train, X_test, y_train, y_test = simple_split(X, y, test_size)
    return X_train, X_test, y_train, y_test


def k_fold_split(X, y, n_splits, by_era=True):
    return None