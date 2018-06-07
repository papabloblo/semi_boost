from sklearn.datasets import make_classification, make_gaussian_quantiles, make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import pdist,squareform
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from mlxtend.plotting import plot_decision_regions

import SemiBoost

import importlib
importlib.reload(SemiBoost)


def mejora_semiboost(n = 20, clf = DecisionTreeClassifier(),
                     n_features = 5, n_samples = 1000, ratio_unsampled = 0.5,
                     data_simulation = 'make_classification'):

    ROC_semiboost = list()
    ROC_clf = list()

    for i in range(n):

        ''' SIMULATE SEMI SUPERVISED DATASET '''
        if data_simulation == 'make_classification':
            X, y = make_classification(n_features = n_features, n_samples = n_samples,
                                       n_redundant = 0, n_clusters_per_class = 1)

        elif data_simulation == 'make_blobs':
            X, y = make_blobs(n_features=n_features, centers=2, n_samples = n_samples)

        elif data_simulation == 'make_gaussian_quantiles':
            X, y = make_gaussian_quantiles(n_features=n_features, n_classes=2, n_samples = n_samples)

        elif data_simulation == 'make_moons':
            X,y = make_moons(n_samples = n_samples)

        elif data_simulation == 'make_circles':
            X, y = make_circles(n_samples=n_samples)

        else:
            print('Unknown data simulation method')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        labels = np.copy(y_train)
        labels[labels == 0] = -1

        # create some unlabeled data
        random_unlabeled_points = np.random.rand(len(y_train)) < ratio_unsampled
        labels[random_unlabeled_points] = 0
        y_train = labels

        ''' SEMIBOOST SKLEARN STYLE '''
        model = SemiBoost.SemiBoostClassifier(base_model = clf)
        model.fit(X_train, y_train, n_neighbors = 3, n_jobs = 10, max_models = 15, similarity_kernel='rbf', verbose = False)

        ROC_semiboost.append(roc_auc_score(model.predict(X_test),y_test))

        ''' BASE CLASSIFIER '''
        model = clf
        XX = X_train[~random_unlabeled_points, ]
        yy = y_train[~random_unlabeled_points]
        model.fit(XX, yy)
        ROC_clf.append(roc_auc_score(model.predict(X_test),y_test))

    return(np.mean(np.array(ROC_semiboost)-np.array(ROC_clf)), np.std(np.array(ROC_semiboost)-np.array(ROC_clf)))



def plot_classification(clf = DecisionTreeClassifier(),
                     n_features = 2, n_samples = 1000, ratio_unsampled = 0.99,
                     data_simulation = 'make_classification'):

                             ''' SIMULATE SEMI SUPERVISED DATASET '''
                             if data_simulation == 'make_classification':
                                 X, y = make_classification(n_features = n_features, n_samples = n_samples,
                                                            n_redundant = 0, n_clusters_per_class = 1)

                             elif data_simulation == 'make_blobs':
                                 X, y = make_blobs(n_features=n_features, centers=2, n_samples = n_samples)

                             elif data_simulation == 'make_gaussian_quantiles':
                                 X, y = make_gaussian_quantiles(n_features=n_features, n_classes=2, n_samples = n_samples)

                             elif data_simulation == 'make_moons':
                                 X,y = make_moons(n_samples = n_samples)

                             elif data_simulation == 'make_circles':
                                 X, y = make_circles(n_samples=n_samples)

                             else:
                                 print('Unknown data simulation method')

                             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                             labels = np.copy(y_train)
                             labels[labels == 0] = -1

                             # create some unlabeled data
                             random_unlabeled_points = np.random.rand(len(y_train)) < ratio_unsampled
                             labels[random_unlabeled_points] = 0
                             y_train = labels

                             ''' SEMIBOOST SKLEARN STYLE '''
                             model = SemiBoost.SemiBoostClassifier(base_model = clf)
                             model.fit(X_train, y_train, n_jobs = 10, max_models = 10, similarity_kernel='rbf', verbose = False)


                             ''' BASE CLASSIFIER '''
                             basemodel = clf
                             XX = X_train[~random_unlabeled_points, ]
                             yy = y_train[~random_unlabeled_points]
                             basemodel.fit(XX, yy)

                             ''' Plot '''
                             gs = gridspec.GridSpec(1, 2)
                             fig = plt.figure(figsize=(10,8))

                             labels = ['SemiBoost', 'BaseModel']
                             for clf, lab, grd in zip([model, basemodel],
                                                     labels,
                                                     [0,1]):

                                                     ax = plt.subplot(gs[0, grd])
                                                     fig = plot_decision_regions(X=X_test, y=y_test, clf=clf, legend=2)
                                                     plt.title(lab)

                             plt.show()
