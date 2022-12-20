from __future__ import division, print_function
import numpy as np
from sklearn import datasets
from utils import train_test_split, accuracy_score, Plot
from randomforest import RandomForest



data = datasets.load_digits()
X = data.data
y = data.target

print(np.unique(y))