''' Example of semi-boost implementation '''

import os
import sys
import importlib
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from sklearn.svm import SVC

# Set working directory to file directory
os.chdir('/data/home/joliver/github/semi_boost/src')
import SemiBoost
importlib.reload(SemiBoost)


''' SIMULATE SEMI SUPERVISED DATASET '''
X, y = make_classification(n_features = 20, n_samples = 10000)

labels = np.copy(y)
labels[labels == 0] = -1

# create some unlabeled data
random_unlabeled_points = rng.rand(len(y)) < 0.3
labels[random_unlabeled_points] = 0
y = labels

''' SEMIBOOST SKLEARN STYLE '''

clf = SVC()
importlib.reload(SemiBoost)
model = SemiBoost.SemiBoostClassifier(base_model = clf)

semiboost_model = model.fit(X, y, n_neighbors = 4, n_jobs = 10)
