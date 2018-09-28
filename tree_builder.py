import numpy as np
from scipy.stats import mode

def entropy(Y):
    unique, counts = np.unique(Y, return_counts=True)
    s = 0.0
    total = np.sum(counts)
    for i, num_y in enumerate(counts):
        probability_y = (num_y/total)
        s += (probability_y)*np.log(probability_y)
    return -s

def information_gain(y,y_true,y_false):
    return entropy(y) - (entropy(y_true)*len(y_true) + entropy(y_false)*len(y_false))/len(y)

class tree_builder(object):

    def __init__(self, max_features = lambda x: x, max_depth = lambda y:y, min_samples_split = lambda z:z):
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split


    def predict_all_X(self, X):
        predictions = []
        for i,x in enumerate(X):
            pred = self.predict_one_sample(x,None)
            predictions.append(pred)
        return np.asarray(predictions).flatten()

    def predict_one_sample(self, x, node):
        if node == None:
            node = self.root

        # base case: we've reached a leaf
        if isinstance(node, Leaf):
            return node.prediction
        if isinstance(node, Decision_Node):
            if node.threshold.match(x):
                return self.predict_one_sample(x,node.true_branch)
            else:
                return self.predict_one_sample(x,node.false_branch)

    def find_best_split(self,X,y,feature_indices):

        max_IG, best_threshold = 0., None
        

        for col in feature_indices:
            vals = np.unique(X[:,col])
            if len(vals) == 1: #skip the feature columns where all samples have the same value
                continue
            for v in vals:
                question = Threshold(col, v)

                #try splitting the dataset
                X_true,y_true, X_false, y_false = self.make_split(X, y, question)

                if len(X_true) == 0 or len(X_false) == 0:
                    continue

                gain = information_gain(y, y_true, y_false)

                if gain > max_IG:
                    max_IG, best_threshold = gain, question

        return max_IG, best_threshold

    def make_split(self, X, y , threshold):
        X_true, y_true, X_false, y_false = [], [], [], []

        for i in range(len(y)):
            if threshold.match(X[i]):
                X_true.append(X[i])
                y_true.append(y[i])
            else:
                X_false.append(X[i])
                y_false.append(y[i])

        X_true = np.array(X_true)
        y_true = np.array(y_true)
        X_false = np.array(X_false)
        y_false = np.array(y_false)

        return X_true, y_true, X_false, y_false


    def build_tree(self, X, y, p_indices,depth):

        gain, threshold = self.find_best_split(X,y,p_indices)

        #base case : if result has no more gain
        if depth is self.max_depth or len(y) < self.min_samples_split or gain == 0:
            return Leaf(y)

        #if not base case, make the split with the best gain and threshold
        left_partition, y_left, right_partition, y_right = self.make_split(X,y, threshold)

        #recursively build downwards for each child
        true_branch = self.build_tree(left_partition, y_left,p_indices,depth+1)
        false_branch = self.build_tree(right_partition, y_right,p_indices,depth+1)

        return Decision_Node(threshold,true_branch,false_branch)

    def fit(self, X, y):
        num_features = X.shape[1]
        num_sub_features = int(self.max_features(num_features))

        #randomly get p number of feature indices
        feature_indices = np.random.choice(num_features, num_sub_features)

        self.root = self.build_tree(X,y,feature_indices,0)

    def init_root():
        node = self.root
        return node

class Leaf:
        def __init__(self, y):
            mod, count = mode(y)
            self.prediction = mod

class Decision_Node(object):
        def __init__(self, threshold, true_branch, false_branch):
            self.threshold = threshold
            self.true_branch = true_branch
            self.false_branch = false_branch

class Threshold:

        def __init__(self, column, value):
            self.column = column
            self.value = value

        def match(self, example):
            # Compare the feature value in an example to the
            # feature value for this threshold
            val = example[self.column]
            return val >= self.value
