import numpy as np
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from scipy import sparse

class SemiBoostClassifier():

    def __init__(self, base_model =SVC()):

        self.BaseModel = base_model

    def fit(self, X, y, n_neighbors=4, n_jobs = 1, n_models = 10, labels = [1,-1], C = 1, sample_percent = 10):

        ''' Fit model'''
        # Localize labeled data
        idx_label = np.argwhere((y == labels[0]) | (y == labels[1])).flatten()
        idx_not_label = np.array([i for i in np.arange(len(y)) if i not in idx_label])
        # First we need to create the similarity matrix
        S = neighbors.kneighbors_graph(X,
                                            n_neighbors=n_neighbors,
                                            mode='distance',
                                            include_self=True,
                                            n_jobs=n_jobs)

        S += np.identity(S.shape[0])
        S = sparse.csr_matrix(S)

        #=============================================================
        # Initialise variables
        #=============================================================
        self.models = []
        self.weights = []

        # Loop for adding sequential models
        for t in range(n_models):
            #=============================================================
            # Predict unlabeled samples with the total model
            #=============================================================
            if t==0:
                # Fit first model to unlabeled observations
                H = np.zeros(idx_not_label.shape[0])

            else:
                H = 0
                for i in range(len(self.models)):
                    H += self.weights[i]*self.models[i].predict(X[idx_not_label])

            #=============================================================
            # Calculate p_i and q_i for every sample
            #=============================================================
            p_1 = np.multiply(np.multiply(S[:,idx_not_label].todense(),np.exp(-2*H))[idx_label,:].T,(y[idx_label]==1)).sum(axis = 1)
            p_2 = C/2*np.multiply(np.multiply(np.exp(-H),S[:,idx_not_label].todense()),np.exp(H)).T[:,idx_not_label].sum(axis=1)
            p = np.add(p_1, p_2)
            p = np.squeeze(np.asarray(p))

            q_1 = np.multiply(np.multiply(S[:,idx_not_label].todense(),np.exp(2*H))[idx_label,:].T,(y[idx_label]==-1)).sum(axis = 1)
            q_2 = C/2*np.multiply(np.multiply(np.exp(H),S[:,idx_not_label].todense()),np.exp(-H)).T[:,idx_not_label].sum(axis=1)
            q = np.add(q_1, q_2)
            q = np.squeeze(np.asarray(q))

            #=============================================================
            # Compute predicted label z_i
            #=============================================================
            z = np.sign(p-q)
            z_conf = np.abs(p-q)
            #=============================================================
            # Sample sample_percent most confident predictions
            #=============================================================
            # Identify samples
            perc = np.percentile(z_conf, 100 - sample_percent)
            idx_sample = np.squeeze(idx_not_label[np.argwhere(z_conf >= perc)])

            # Create new X_t, y_t
            idx_total_sample = np.concatenate([idx_label,idx_sample])
            X_t = X[idx_total_sample,]
            np.put(y, idx_sample, z[np.argwhere(z_conf >= perc)])# Include predicted to train new model
            y_t = y[idx_total_sample]

            ''' HAY QUE MUESTREAR DE ACORDE CON EL PESO Z_CONF'''

            #=============================================================
            # Fit BaseModel to samples using predicted labels
            #=============================================================
            # Fit model to unlabeled observations
            clf = self.BaseModel
            clf.fit(X_t, y_t)
            # Make predictions
            h = clf.predict(X[idx_not_label])

            # Refresh indexes
            idx_label = idx_total_sample
            idx_not_label = np.array([i for i in np.arange(len(y)) if i not in idx_label])
            print('There are still ', idx_not_label.shape[0], ' unlabeled observations')

            #=============================================================
            # Compute weight (a) for the BaseModel
            #=============================================================
            a = np.log(np.dot(p,h==1) + np.dot(q,h==-1)) - np.log(np.dot(p,h==-1) + np.dot(q,h==1))

            #=============================================================
            # Update final model
            #=============================================================
            # Save model
            self.models.append(clf)
            #save weights
            self.weights.append(a)

            #=============================================================
            # If no samples are left without label, break
            if len(idx_not_label) == 0:
                print('All observations have been labeled')
                print('Number of iterations: ',t + 1)
                break

        # When the training loop is over, ensemble the models using the weights
        self.model = VotingClassifier(self.models, weights = self.weights, voting = 'soft', n_jobs = n_jobs)

        return self.model
