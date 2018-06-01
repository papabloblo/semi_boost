
import numpy as np
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sympy import KroneckerDelta

class SemiBoostModel():

    def __init__(self, hello = 'hello world', base_model =SVC()):

        self.hello = hello
        self.BaseModel = base_model

    def fit(self, X, y, n_neighbors=4, n_jobs = 1, n_models = 10, labels = [1,-1]):

        ''' Fit model'''
        # Localize labeled data
        idx_label = np.argwhere((y == labels[0]) | (y == labels[1])).flatten()
        # First we need to create the similarity matrix
        S = neighbors.kneighbors_graph(X,
                                            n_neighbors=n_neighbors,
                                            mode='distance',
                                            include_self=True,
                                            n_jobs=n_jobs)

        #=============================================================
        # Initialise variables
        #=============================================================

        # First model
        clf = self.BaseModel
        clf.fit(X[idx_label,], y[idx_label])

        H = clf.predict(X)
        # Loop for adding sequential models
        for t in range(n_models):

            #=============================================================
            # Calculate p_i and q_i for every sample
            #=============================================================
            p_1 = np.sum(np.multiply(S[:,idx_label],np.exp(H))*(y==1), axis=1)
            print('hola')

            #=============================================================
            # Compute predicted label z_i
            #=============================================================

            #=============================================================
            # Sample observations
            #=============================================================

            #=============================================================
            # Fit BaseModel to samples using predicted labels
            #=============================================================

            #=============================================================
            # Compute weight (a) for the BaseModel
            #=============================================================

            #=============================================================
            # Update final model
            #=============================================================



        return S, H, idx_label
