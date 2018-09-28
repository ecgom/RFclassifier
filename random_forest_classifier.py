from scipy.stats import mode
import numpy as np

from tree_builder import *

class RandomForestClassifier(object):

    def __init__(self,num_trees=lambda x: x,max_features=np.sqrt,max_depth=10, \
                min_samples_split=2,bootstrap=1):
        self.num_trees = num_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.forest = []

    def fit(self, X, y):
        self.forest = []
        N_dataset = len(X)
        n_to_bootstrap = np.round(N_dataset*self.bootstrap)

        for i in range(0, self.num_trees):
            # bootstrap training dataset
            idx = np.random.choice(X.shape[0],n_to_bootstrap)

            X_subset, y_subset = X[idx, :], y[idx]
            tree = tree_builder(self.max_features,self.max_depth,self.min_samples_split)
            tree.fit(X_subset, y_subset)
            self.forest.append(tree)

    def predict(self, X):
        
        n_samples = X.shape[0]
        n_trees = len(self.forest)
        predictions = np.empty([n_trees, n_samples])
        for i in range(n_trees):
            predictions[i] = self.forest[i].predict_all_X(X)

        m = mode(predictions)
        return(m[0])





    
