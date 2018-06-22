import numpy as np
from sklearn import neighbors
from sklearn.svm import SVC
from scipy import sparse
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics.pairwise import rbf_kernel

class SemiBoostClassifier():

    def __init__(self, base_model =SVC()):

        self.BaseModel = base_model

    def fit(self, X, y,
            n_neighbors=4, n_jobs = 1,
            max_models = 15,
            sample_percent = 0.5,
            sigma_percentile = 90,
            labels = [1,-1],
            similarity_kernel = 'rbf',
            verbose = True):

        ''' Fit model'''
        # Localize labeled data
        idx_label = np.argwhere((y == labels[0]) | (y == labels[1])).flatten()
        idx_not_label = np.array([i for i in np.arange(len(y)) if i not in idx_label])

        # The parameter C is defined in the paper as
        C = idx_label.shape[0]/idx_not_label.shape[0]

        # First we need to create the similarity matrix
        if similarity_kernel == 'knn':

            self.S = neighbors.kneighbors_graph(X,
                                                n_neighbors=n_neighbors,
                                                mode='distance',
                                                include_self=True,
                                                n_jobs=n_jobs)

            self.S = sparse.csr_matrix(self.S)

        elif similarity_kernel == 'rbf':
            # First aprox
            self.S = np.sqrt(rbf_kernel(X, gamma = 1))
            # set gamma parameter as the 15th percentile
            sigma = np.percentile(np.log(self.S), sigma_percentile)
            sigma_2 = (1/sigma**2)*np.ones((self.S.shape[0],self.S.shape[0]))
            self.S = np.power(self.S, sigma_2)
            # Matrix to sparse
            self.S = sparse.csr_matrix(self.S)

        else:
            print('No kernel type ', similarity_kernel)


        #=============================================================
        # Initialise variables
        #=============================================================
        self.models = []
        self.weights = []
        H = np.zeros(idx_not_label.shape[0])

        # Loop for adding sequential models
        for t in range(max_models):
            #=============================================================
            # Calculate p_i and q_i for every sample
            #=============================================================
            p_1 = np.einsum('ij,j', self.S[:,idx_label].todense(), (y[idx_label]==1))[idx_not_label]*np.exp(-2*H)
            p_2 = np.einsum('ij,j', self.S[:,idx_not_label].todense(), np.exp(H))[idx_not_label]*np.exp(-H)
            p = np.add(p_1, p_2)
            p = np.squeeze(np.asarray(p))

            q_1 = np.einsum('ij,j', self.S[:,idx_label].todense(), (y[idx_label]==-1))[idx_not_label]*np.exp(2*H)
            q_2 = np.einsum('ij,j', self.S[:,idx_not_label].todense(), np.exp(-H))[idx_not_label]*np.exp(H)
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
            # Sampling weights
            sample_weights = z_conf/np.sum(z_conf)
            # If there are non-zero weights
            if np.any(sample_weights != 0):
                idx_aux = np.random.choice(np.arange(len(z)),
                                              size = int(sample_percent*len(idx_not_label)),
                                              p = sample_weights,
                                              replace = False)
                idx_sample = idx_not_label[idx_aux]

            else:
                print('No similar unlabeled observations left.')
                break

            # Create new X_t, y_t
            idx_total_sample = np.concatenate([idx_label,idx_sample])
            X_t = X[idx_total_sample,]
            np.put(y, idx_sample, z[idx_aux])# Include predicted to train new model
            y_t = y[idx_total_sample]

            #=============================================================
            # Fit BaseModel to samples using predicted labels
            #=============================================================
            # Fit model to unlabeled observations
            clf = self.BaseModel
            clf.fit(X_t, y_t)
            # Make predictions for unlabeled observations
            h = clf.predict(X[idx_not_label])

            # Refresh indexes
            idx_label = idx_total_sample
            idx_not_label = np.array([i for i in np.arange(len(y)) if i not in idx_label])

            if verbose:
                print('There are still ', idx_not_label.shape[0], ' unlabeled observations')

            #=============================================================
            # Compute weight (a) for the BaseModel as in (12)
            #=============================================================
            e = (np.dot(p,h==-1) + np.dot(q,h==1))/(np.sum(np.add(p,q)))
            a = 0.25*np.log((1-e)/e)
            #=============================================================
            # Update final model
            #=============================================================
            # If a<0 the model is not converging
            if a<0:
                if verbose:
                    print('Problematic convergence of the model. a<0')
                break

            # Save model
            self.models.append(clf)
            #save weights
            self.weights.append(a)
            # Update
            H = np.zeros(len(idx_not_label))
            w = np.sum(self.weights)
            for i in range(len(self.models)):
                H = np.add(H, self.weights[i]*self.models[i].predict(X[idx_not_label]))
                # H = np.add(H, self.weights[i]*self.models[i].predict_proba(X[idx_not_label])[:,1]/w)

            # H = np.array(list(map(lambda x: 1 if x>0 else -1, H)))
            #=============================================================
            # Breaking conditions
            #=============================================================

            # Maximum number of models reached
            if (t==max_models) & verbose:
                print('Maximum number of models reached')

            # If no samples are left without label, break
            if len(idx_not_label) == 0:
                if verbose:
                    print('All observations have been labeled')
                    print('Number of iterations: ',t + 1)
                break

        if verbose:
            print('\n The model weights are \n')
            print(self.weights)



    def predict(self, X):
        estimate = np.zeros(X.shape[0])
        # Predict weighting each model
        w = np.sum(self.weights)
        for i in range(len(self.models)):
            # estimate = np.add(estimate,  self.weights[i]*self.models[i].predict_proba(X)[:,1]/w)
            estimate = np.add(estimate, self.weights[i]*self.models[i].predict(X))
        estimate = np.array(list(map(lambda x: 1 if x>0 else -1, estimate)))
        estimate = estimate.astype(int)
        return estimate
