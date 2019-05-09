from hmmlearn import hmm
from sklearn.base import BaseEstimator
import numpy as np

# lambda which flattens list of lists
flatten = lambda l: [item for sublist in l for item in sublist]

class HMM(BaseEstimator):
    """Single spaced hidden markov model classifier."""
    def __init__(self, components):
        self.components = components
        self.flatten = lambda l: [item for sublist in l for item in sublist]

    def fit(self, X, y):
        dict_y = dict()

        # Partition data in labels
        for i in range(len(X)):
            if y[i] not in dict_y:
                dict_y[y[i]] = []
            
            dict_y[y[i]].append(X[i])
        
        self.models = dict()

        # Make and fit model for each label
        for y_k, X_list in dict_y.items():
            X_len = [len(x) for x in X_list]
            X_tmp = np.array(flatten(X_list)).reshape(-1, 1)
            
            self.models[y_k] = hmm.GaussianHMM(n_components=self.components).fit(X_tmp, lengths=X_len)
        
        return self
        
    def predict(self, X):
        predicts = []
        for branch in X:
            # get len of branch and reshape it
            b_len = len(branch)
            branch = np.array(branch).reshape(-1, 1)
            best_y = -1
            best_score = None
            for y, model in self.models.items():
                score = model.score(branch, lengths=[b_len])
                if best_score is None or score > best_score:
                    best_y = y
                    best_score = score
            
            predicts.append(best_y)
        
        return predicts