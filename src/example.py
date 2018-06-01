''' Example of semi-boost implementation '''
import os
import importlib
# os.chdir('src')

from sklearn import datasets
import numpy as np
import SemiBoost
importlib.reload(SemiBoost)
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

clf = SVC()
model = SemiBoost.SemiBoostModel(hello = 'hello world', base_model = clf)

S, H, idx_label = model.fit(X, y)
X.shape
S[:,idx_label].todense().dot(y[idx_label]).shape

Ht = np.exp(H).reshape((H.shape[0],1))
Ht.shape
S[:,idx_label].shape
(S*H).shape
np.multiply(S,np.exp(H)).shape
np.multiply(np.multiply(S,(y==1)),np.exp(H)).shape
#=============================

clf = SVC()
clf.fit(iris.data, iris.target)
clf.score(iris.data, iris.target)

(y == 1)
