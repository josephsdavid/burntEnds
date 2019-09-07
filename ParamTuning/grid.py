import numpy as np
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report # other metrics?
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import itertools
from multiprocessing import Process, Queue
import sklearn.datasets

X,Y = sklearn.datasets.load_breast_cancer(return_X_y=True)




# adapt this code below to run your analysis

# Recommend to be done before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie
# use logistic regression, each with 3 different sets of hyper parrameters for each

# First I think the default values for sklearn are ridiculous
# so we will redefine the classifiers (these are hidden) to have reasonable
# values. Mostly the nTrees, as well as for example the regularization defaults
# in logistic regression
class classifRandomForest:
    def __init__(self,
                 nTrees = 100,
                 crit = "gini",
                 depth = None,
                 nodeSamples = 2,
                 leafSamples = 1,
                 weightFrac = 0.,
                 features = "auto",
                 nodes = None,
                 bootstrap = True,
                 oob = False,
                 n_jobs = -1,
                 seed = None):
        self.model = RandomForestClassifier(n_estimators = nTrees,
                     criterion = crit,
                     max_depth = depth,
                     min_samples_split = nodeSamples,
                     min_samples_leaf = leafSamples,
                     min_weight_fraction_leaf = weightFrac,
                     max_features = features,
                     max_leaf_nodes = nodes,
                     bootstrap = bootstrap,
                     oob_score = oob,
                     n_jobs = -1,
                     random_state = seed
                     )
    def __call__(self):
        print(self.model)




#rf = classifRandomForest(200)

#rf.train(X,y)

#print(rf())

class Classifier:
    # make a classifier class, instantiate it with default hyperparameters
    def __init__(self, method, hyperPars = {}):
        self.identity = method
        if (method == "RandomForest"):
            if (hyperPars == {}):
                self.method = classifRandomForest()
            else:
                self.method = classifRandomForest(**hyperPars)

    # call method
    def __call__(self):
        print("model: ",self.method.model)

    def train(self,features, labels):
        self.method.model.fit(features, labels)

    #prediction method for convenience
    def predict(self,features):
        return(self.method.model.predict(features))

rf = Classifier("RandomForest", hyperPars = {"nTrees":200, "seed":47})





# Next we need to make a few cross validation functions. We will do
# cross validation, stratified cross validation, and repeated cross validation


class Resampling:
    def __init__(self, method, repeats = 10, folds = 5, stratify = False, seed = None):
        if(stratify == False):
            if(method == "cv"):
                self.method = KFold(n_splits = folds, random_state = seed)
            elif(method == "repeatedcv"):
                self.method = RepeatedKFold(n_splits = folds, n_repeats = repeats, random_state = seed)
        else:
            self.method = StratifiedKFold(n_splits = folds, n_repeats = repeats, random_state = seed)

    def __call__(self):
        print(self.method)

cv = Resampling("repeatedcv", folds = 2, repeats = 5, seed = 47)
# next lets make constructors for different hyperparameter sets, then we can
# expand them into a grid or whatever

class discreteParam:
    def __init__(self,name, values):
        self.name = name
        self.values = values
    def __call__(self):
        print("param: ", self.name, "\nvalues: ", self.values)
    def __iter__(self):
        return(self)


class integerParam:
    def __init__(self, name, lower, upper, transFun = None):
        self.idx = 0
        self.name = name
        if(transFun == None):
            self.values = ([i for i in np.arange(lower, upper+1, dtype = int)])
        else:
            self.values = ([transFun(i) for i in np.arange(lower, upper+1, dtype = int)])
    def __call__(self):
        print("param: ", self.name)
        print("values: ", self.values)
    def __iter__(self):
        return(self)



class floatParam:
    def __init__(self, name, lower, upper, by = 0.1, transFun = None):
        self.name = name
        if(transFun == None):
            self.values = ([i for i in np.arange(lower, upper+1, by, dtype = float)])
        else:
            self.values = ([transFun(i) for i in np.arange(lower, upper+1, by, dtype = float)])
    def __call__(self):
        print("param: ", self.name, "\nvalues: ", self.values)

    def __iter__(self):
        return(self)


def getNames(param):
    return(param.name)


class tuneGrid:
    def __init__(self,model, features, labels, sampling, metric, *params):


    # our default paramter test
        self.method = model.identity
        self.metric = metric

        # get the names of the hyperparameters we are tuning
        self.names = list(map(getNames, params))
        self.params = params
        self.features = features
        self.labels = labels
        self.sampler = sampling
        d = {}
        for p in self.params:
            d[p.name] = p.values
        self.x = d
        grid = list(itertools.product(*((d[i] )for i in (d))))
        self.grid = grid
        self.ran = False
        self.bestTry = int(0)
        self.bestScore = float(0)
        self.bestPars = {}
        self.results = []
    def run(self):
        self.ran = True
        #
        # instantiate an empty dict
        f = self.features
        L = self.labels

        # set up the dict using the params
        pars = {}
        print("testing other models")
        # turn it into a lovely list of dicts for our unpacking
        for row in range(len(self.grid)):
            for column in range(len(self.names)):
                pars[self.names[column]] = (self.grid[row][column])
                clf = Classifier(self.method, pars)
            print("calculating:",(row + 1),"out of:", len(self.grid))
            scores = []
            for ids,(train_index, test_index) in enumerate(self.sampler.method.split(f)):
                X_train, X_test = f[train_index], f[test_index]
                y_train, y_test = L[train_index], L[test_index]
                clf.train(features = X_train, labels = y_train)
                # I would like to get the clf.predict method working
                preds = clf.predict(X_test)
                scores.append(self.metric(preds, y_test))
            res = sum(scores)/len(scores)
            print(res)
            self.results.append(res)
        self.bestScore = max(self.results)
        self.bestTry = self.results.index(self.bestScore)
        for col in range(len(self.names)):
            self.bestPars[self.names[col]] = self.grid[self.bestTry][col]

    def save(self, path = "grid.obj"):
        filehandler = open(path,'wb')
        pickle.dump(self, filehandler)

    def __call__(self):
        if(self.ran):
            print("----------------------")
            print("model: ", self.method)
            print("----------------------")
            print("parameters: \n", self.names)
            print("----------------------")
            print("space:\n", self.grid)
            print("----------------------")
            print("results:\n", self.results)
            print("----------------------")
            print(
                "Best Paramter Set:",
                self.bestPars
            )
            print("----------------------")
            print(" Best Score:", self.bestScore)
        else:
            print("----------------------")
            print("model: ", self.method)
            print("----------------------")
            print("parameters: \n", self.names)
            print("----------------------")
            print("space:\n", self.grid)
            print("----------------------")






#t =  tuneGrid(rf,
#         X,
#         Y,
#         cv, roc_auc_score,
#         discreteParam("crit",["gini","entropy"]),
#         integerParam("nTrees", 1, 2, lambda x: 200*x),
#         integerParam("depth", 2,3,transFun = lambda x: 2**x)
#            )
#

#t.run()
#(t())

#t.save()

def load(filename):
    with open(filename, 'rb') as fileObj:
        raw_data = fileObj.read()
        return(pickle.loads(raw_data))

t  = load("grid.obj")

t()













# accepts a lower bound, upper bound, and transformation function
# yes i know list comprehensions exist but

cv = Resampling("cv")
rf = Classifier(RandomForestClassifier)

#pars = {
#    "n_estimators":makeIntegerParam(1,5, lambda x: 100*x),
#    "criterion":makeDiscreteParam(["gini","entropy"]),
#    "max_depth":makeIntegerParam(1,5, lambda x: 10*x)#,
#    #"min_weight_fraction_leaf":makeDoubleParam(0,2,trafo = lambda x: x*2)
#}
#print(expandgrid(pars))


#tune(rf, cv, pars)




















# Recommend to be done before live class 3
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings

# Recommend to be done before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

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
