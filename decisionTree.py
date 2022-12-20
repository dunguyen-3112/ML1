import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.left = None
        self.right = None
        self.split_index = None
        self.split_value = None
        self.prediction = None
        
    def fit(self, X, y):
        if self.max_depth is not None and self.max_depth <= 0:
            self.prediction = np.mean(y)
            return
        
        # Find the best split
        best_split_index, best_split_value = self._find_best_split(X, y)
        
        if best_split_index is None:
            # If we can't find a good split, make this a leaf node and store the mean of the target values
            self.prediction = np.mean(y)
            return
        
        # Split the data into left and right branches
        left_mask = X[:, best_split_index] < best_split_value
        right_mask = ~left_mask
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # Recursively fit the left and right branches
        self.left = DecisionTreeRegressor(self.max_depth - 1 if self.max_depth is not None else None)
        self.left.fit(X_left, y_left)
        self.right = DecisionTreeRegressor(self.max_depth - 1 if self.max_depth is not None else None)
        self.right.fit(X_right, y_right)
        
        # Store the split for prediction
        self.split_index = best_split_index
        self.split_value = best_split_value
        
    def predict(self, X):
        if self.prediction is not None:
            return self.prediction
        
        if X[self.split_index] < self.split_value:
            return self.left.predict(X)
        else:
            return self.right.predict(X)
    
    def _find_best_split(self, X, y):
        best_split_index = None
        best_split_value = None
        min_error = np.inf
        
        for i in range(X.shape[1]):
            for x in X[:, i]:
                left_mask = X[:, i] < x
                right_mask = ~left_mask
                y_left, y_right = y[left_mask], y[right_mask]
                error = np.mean((y_left - np.mean(y_left)) ** 2) + np.mean((y_right - np.mean(y_right)) ** 2)
                if error < min_error:
                    min_error = error
                    best_split_index = i