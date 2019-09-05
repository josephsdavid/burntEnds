import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report # other metrics?
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import itertools
from multiprocessing import Pool, cpu_count

# adapt this code below to run your analysis

# Recommend to be done before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie
# use logistic regression, each with 3 different sets of hyper parrameters for each

class Classifier:
    # make a classifier class, instantiate it with default hyperparameters
    def __init__(self, model, hyperPars = {}):
        self.model = model(**hyperPars)
        self.hyperPars = hyperPars

    # call method
    def __call__(self):
        print("model: ",self.model)
        print("Hyper parameters: ", self.hyperPars)


    # training method
    def train(self, features, labels):
        self.model.fit(features, labels)

    # prediction method for convenience
    def predict(features):
        self.model.predict(features)




# Next we need to make a few cross validation functions. We will do
# cross validation, stratified cross validation, and repeated cross validation


class Resampling:
    def __init__(self, method, repeats = 10, folds = 5, stratify = False):
        if(stratify == False):
            if(method == "cv"):
                self.method = KFold(n_splits = folds)
            elif(method == "repeatedcv"):
                self.method = RepeatedKFold(n_splits = folds, n_repeats = repeats)
        else:
            self.method = StratifiedKFold

    def __call__(self):
        print(self.method)


# next lets make constructors for different hyperparameter sets, then we can
# expand them into a grid or whatever

def makeDiscreteParam(values):
    return(values)

# accepts a lower bound, upper bound, and transformation function
# yes i know list comprehensions exist but
def makeIntegerParam(lower, upper, trafo = None):
    if(trafo == None):
        return([i for i in np.arange(lower, upper+1, dtype = int) ])
    else:
        return([trafo(i) for i in np.arange(lower, upper+1, dtype = int)])

def makeDoubleParam(lower, upper, by = 0.1, trafo = None):
    if(trafo == None):
        return([i for i in np.arange(lower, upper, by, dtype = float)])
    else:
        return([trafo(i) for i in np.arange(lower, upper, by, dtype = float)])



def expandgrid(items):
   product = itertools.product(*((items[i] )for i in sorted(items)))
   return(list(product))


cv = Resampling("cv")
rf = Classifier(RandomForestClassifier)

pars = {
    "n_estimators":makeIntegerParam(1,5, lambda x: 100*x),
    "criterion":makeDiscreteParam(["gini","entropy"]),
    "max_depth":makeIntegerParam(1,5, lambda x: 10*x)#,
    #"min_weight_fraction_leaf":makeDoubleParam(0,2,trafo = lambda x: x*2)
}
print(expandgrid(pars))


def tune(model, sampling, hyperPars = {}, parallel = False):
    if(parallel == False):
        grid = expandgrid(hyperPars)
        print(grid[0])
        print(np.shape(grid))
        print(type(grid[0]))
        for rows in np.shape(grid[0]):



tune(rf, cv, pars)




















# Recommend to be done before live class 3
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings

# Recommend to be done before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
L = np.ones(M.shape[0])
print(L)
#n_folds = 5
#
#data = (M, L, n_folds)
#
#def run(a_clf, data, clf_hyper={}):
#  M, L, n_folds = data # unpack data containter
#  kf = KFold(n_splits=n_folds) # Establish the cross validation
#  ret = {} # classic explicaiton of results
#
#  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
#    clf = a_clf(**clf_hyper) # unpack paramters into clf is they exist
#
#    clf.fit(M[train_index], L[train_index])
#
#    pred = clf.predict(M[test_index])
#
#    ret[ids]= {'clf': clf,
#               'train_index': train_index,
#               'test_index': test_index,
#               'accuracy': accuracy_score(L[test_index], pred)}
#  return ret
#
#results = run(RandomForestClassifier, data, clf_hyper={})
