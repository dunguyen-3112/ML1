from decisionTree import DecisionTreeRegressor
import pandas as pd
import numpy as np

def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def standardize(X):
    """ Standardize the dataset X """
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

data = pd.read_csv('./TempLinkoping2016.txt', sep="\t")

time = np.atleast_2d(data["time"].values).T
temp = np.atleast_2d(data["temp"].values).T

X = standardize(time)        # Time. Fraction of the year [0, 1]
y = temp[:, 0]  # Temperature. Reduce to one-dim

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
d =  DecisionTreeRegressor()

d.fit(X_train,y_train)
d.predict(X_test)

