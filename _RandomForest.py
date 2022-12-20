import random
from decisionTree import DecisionTreeRegressor
import numpy as np


class RandomForestRegressor:
    def __init__(self, n_estimators, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # for each tree in the forest:
        for i in range(self.n_estimators):
            # create a bootstrapped sample of the training data
            bootstrapped_X, bootstrapped_y = self.bootstrap(X, y)
            # train a decision tree on the bootstrapped sample
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(bootstrapped_X, bootstrapped_y)
            # add the trained tree to the list of trees
            self.trees.append(tree)

    def predict(self, X):
        # make predictions using all the trained trees in the forest
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        # return the mean prediction across all the trees
        return np.mean(predictions, axis=0)

    def bootstrap(self, X, y):
        # create a bootstrapped sample of the training data
        n_samples = len(X)
        bootstrapped_indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        bootstrapped_X = X[bootstrapped_indices]
        bootstrapped_y = y[bootstrapped_indices]
        return bootstrapped_X, bootstrapped_y
