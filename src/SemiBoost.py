from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

class SemiBoostModel():

    def __init__(self, hello = 'hello world', base_model =SVC()):

        self.hello = hello
        self.BaseModel = base_model

    def fit(self, X, y, n_neighbors=4, n_jobs = 1, n_models = 10):

        ''' Fit model'''

        # First we need to create the similarity matrix
        S = sklearn.neighbors.kneighbors_graph(X,
                                            n_neighbors=n_neighbors,
                                            mode='distance',
                                            include_self=True,
                                            n_jobs=n_jobs)

        # Accumulate fitted models in list
        models = list()

        # Loop for adding sequential models
        for t in range(n_models):

            #=============================================================
            # Calculate p_i and q_i for every sample
            #=============================================================
            # If it's the first iteration no EnsembledModel exists
            if t==0:


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



        return S
