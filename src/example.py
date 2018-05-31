''' Example of semi-boost implementation '''

from sklearn import datasets
import numpy as np
from SemiBoost import SemiBoostModel
from sklearn.svm import SVC

# Import data to use
iris = datasets.load_breast_cancer()
rng = np.random.RandomState(42)
labels = np.copy(iris.target)
labels[labels == 0] = -1

# create some unlabeled data
random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
labels[random_unlabeled_points] = 0

X, y = iris.data, labels

y

clf = SVC()
model = SemiBoostModel(hello = 'hello world', base_model = clf)

model.fit(X, y)
