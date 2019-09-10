import numpy as np
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score # other metrics?
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from random import choices, sample
import itertools
import math
from multiprocessing import Process, Queue
#import sklearn.datasets
from operator import itemgetter as  itemget
import scipy.stats as ss

#X,Y = sklearn.datasets.load_breast_cancer(return_X_y=True)




# adapt this code below to run your analysis

# Recommend to be done before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie
# use logistic regression, each with 3 different sets of hyper parrameters for each

# First I think the default values for sklearn are ridiculous
# so we will redefine the classifiers (these are hidden) to have reasonable
# values. Mostly the n_estimators, as well as for example the regularization defaults
# in logistic regression
class classifRandomForest:
    def __init__(self,
                 n_estimators = 100,
                 criterion = "gini",
                 max_depth = None,
                 min_samples_split = 2,
                 min_samples_leaf = 1,
                 min_weight_fraction_leaf = 0.,
                 max_features = "auto",
                 max_leaf_nodes = None,
                 bootstrap = True,
                 oob_score = False,
                 min_impurity_decrease = 0.,
                 n_jobs = -1,
                 random_state = None):
        self.model = RandomForestClassifier(n_estimators = n_estimators,
                     criterion = criterion,
                     max_depth = max_depth,
                     min_samples_split = min_samples_split,
                     min_samples_leaf = min_samples_leaf,
                     min_weight_fraction_leaf = min_weight_fraction_leaf,
                     max_features = max_features,
                     max_leaf_nodes = max_leaf_nodes,
                     bootstrap = bootstrap,
                     oob_score = oob_score,
                     min_impurity_decrease = min_impurity_decrease,
                     n_jobs = -1,
                     random_state = random_state
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

rf = Classifier("RandomForest", hyperPars = {"n_estimators":200,"random_state":69})






# Next we need to make a few cross validation functions. We will do
# cross validation, stratified cross validation, and repeated cross validation


class Resampling:
    def __init__(self, method, repeats = 10, folds = 5, stratify = False, random_state = None):
        if(stratify == False):
            if(method == "cv"):
                self.method = KFold(n_splits = folds, random_state = random_state)
            elif(method == "repeatedcv"):
                self.method = RepeatedKFold(n_splits = folds, n_repeats = repeats, random_state = random_state)
        else:
            # does not work yet
            self.method = StratifiedKFold(n_splits = folds, random_state = random_state)

    def __call__(self):
        print(self.method)

cv = Resampling("cv", folds = 3, repeats = 5)
# next lets make constructors for different hyperparameter sets, then we can
# expand them into a grid or whatever

class discreteParam:
    def __init__(self,name, values):
        self.name = name
        self.values = values
    def __call__(self):
        print("param: ", self.name, "\nvalues: ", self.values)
    def isInt(self):
        return(False)
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

    def isInt(self):
        return(True)



class floatParam:
    def __init__(self, name, lower, upper, transFun):
        self.name = name
        self.values = ([transFun(i) for i in np.arange(lower, upper+1, dtype = float)])
    def __call__(self):
        print("param: ", self.name, "\nvalues: ", self.values)

    def __iter__(self):
        return(self)

    def isInt(self):
        return(False)




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
            return(self.method,
            self.names,
            self.grid,
             self.results,
                self.bestPars,
            self.bestScore)
        else:
            return(self.method,
            self.names,
            self.grid)

    def print(self):
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








#t.run()
#(t())

#t.save()

def load(filename):
    with open(filename, 'rb') as fileObj:
        raw_data = fileObj.read()
        return(pickle.loads(raw_data))


#t()

# model multiplexer

class Tuner:
    def __init__(self, *searches):
        tuners = []
        for s in searches:
            tuners.append(s)
        self.tuners = tuners
        self.results = []
    def run(self):
        for tuner in self.tuners:
            tuner.run()
            print(tuner.bestScore)
            print(tuner.bestPars)
            self.results.append(tuner)

    def save(self, path = "mutiTuner.obj"):
        filehandler = open(path,'wb')
        pickle.dump(self, filehandler)


#t =  tuneGrid(rf,
#         X,
#         Y,
#         cv, roc_auc_score,
#         discreteParam("criterion",["gini","entropy"]),
#         integerParam("n_estimators", 1, 2, lambda x: 200*x),
#         integerParam("max_depth", 2,3,transFun = lambda x: 2**x)
#            )
#
#r =  tuneGrid(rf,
#         X,
#         Y,
#         cv, roc_auc_score,
#         integerParam("n_estimators", 2, 4, lambda x: 200*x),
#         integerParam("max_depth", 2,6,transFun = lambda x: 2**x)
#            )
#
#attempt = Tuner(t,r)
#attempt.run()


# now lets create an iterated racing optimization method, and then we will fix up the
# rest of our world. We do this now because we are excited

# first lets create a random sampling method

class tuneRandom:
    def __init__(self,model, features, labels, sampling, metric, iters,*params):
        self.method = model.identity
        self.metric = metric

        # get the names of the hyperparameters we are tuning
        self.names = list(map(getNames, params))
        self.params = params
        self.features = features
        self.labels = labels
        self.sampler = sampling

        l = []
        for p in self.params:
            l.append(choices(p.values, k = iters))
        self.grid = list(map(list, zip(*l)))
        self.ran = False
        self.bestTry = int(0)
        self.bestScore = float(0)
        self.bestPars = {}
        self.results = []
        self.iters = iters
    def run(self):
        self.ran = True

        pars = {}

        f = self.features
        L = self.labels
        indices = np.random.randint(1, len(self.grid)-1, size = self.iters)
        for row in indices:
            for column in range(len(self.names)):
                pars[self.names[column]] = (self.grid[row][column])
            clf = Classifier(self.method, pars)
            clf()
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

    def save(self, path = "randomSearch.obj"):
        filehandler = open(path,'wb')
        pickle.dump(self, filehandler)

    def __call__(self):
        if(self.ran):
            return(self.method,
            self.names,
            self.grid,
             self.results,
                self.bestPars,
            self.bestScore)
        else:
            return(self.method,
            self.names,
            self.grid)

    def print(self):
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





#testCase =  tuneRandom(rf,
#         X,
#         Y,
#         cv, roc_auc_score,10,
#         integerParam("n_estimators", 1, 100, lambda x: 10*x),
#         integerParam("max_depth", 1,20,transFun = lambda x: x**2)
#            )
#testCase.run()

#testCase.save()
#
#testCase.print()
#
#
#
#



'''
 Description of irace:
 instances: I
 params: X
 loss function: L
 budget: B
 NRaces = 2 + log2(len(params))
 for j in nraces:
 Bj = (B - Bused)/(nRaces - j  + 1)
 mu = user defined
 SampleSize = Bj/(mu + min(5,j))
 First run:
 j = 1
 update Bj
 calculte SampleSize
 populate random population of hyperparameters with size sample size, sampled
 from the user defined parameter space (I)
 calculate mean L over k fold (repeated) cross validation, for each item in our
 population
 j = 2
 while loop start
 updateBJ()
 calculate SampleSize
 rank population by loss (mean rank at ties)
 perform statistical test on ranks, to determne which ones are weak
 typically a friedman test with post hoc tests is used
 discard the statistically weak, leaving behind the elites
 sample elites once with custom probability distribution that scales with rank
 (aka weaker elites are more likely to be sampled)
 the sampled rank gets you the parent set
 use parent set hyperparameter values as mean
 calculate std of each hyperparameter in elites
 add new samples N(mean, std) to the elites until we reach the newly calculated sample size
 race the new samples (CV etc.)
 j+=1
 see http://iridia.ulb.ac.be/IridiaTrSeries/link/IridiaTr2011-004.pdf
'''

# we are going to make a ranked quantile race, returning by default the top n
# percentile
class tuneIrace:
    mu = 0.5
    def __init__(self,model, features, labels, sampling, metric, budget,*params):
        self.types = list(map(lambda x: x.isInt(), params))
        self.method = model.identity
        self.metric = metric
        self.nRaces = math.ceil(2 + math.log(len(params),2))
        self.Bused = 0
        self.Bj = 0
        self.j = 1
        # get the names of the hyperparameters we are tuning
        self.names = list(map(getNames, params))
        self.params = params
        self.features = features
        self.labels = labels
        self.sampler = sampling
        self.grid = []
        #self.grid = grid
        #self.grid = list(map(tuple, zip(*l)))
        self.ran = False
        self.bestTry = int(0)
        self.bestScore = float(0)
        self.scores = []
        self.ranks = []
        self.eliteIndices = []
        self.bestPars = {}
        self.budget = budget
    def save(self, path = "mutiTuner.obj"):
        filehandler = open(path,'wb')
        pickle.dump(self, filehandler)
    def getSampleSize(self):
        return(
            math.ceil(self.Bj / (self.mu + min(5, self.j)))
        )
    def updateBJ(self):
        self.Bj = (self.budget - self.Bused) / (self.nRaces - self.j + 1)

    # figure out this
    def friedman(self, ranks, sampleSize, space):
        Rj = sum(ranks)
        k = sampleSize
        m = len(space)
        numLHS = m-1
        RHS = []
        def numRHS(i):
            return((Rj - (k * (i+1)/2))**2)
        for i in range(m):
           RHS.append(numRHS(i))
        numerator = numLHS * sum(RHS)

        denominator = []
        predenominator = []
        for l in range(k):
            for j in range(m):
                res = Rj**2 - (l*j*((j+1)**2))/4
                predenominator.append(res)
            denominator.append(sum(predenominator))
        stat = (numerator/sum(denominator))
        return(1-ss.chi2.cdf(stat, m-1))
# also figure this thing out, for now just have  probability scale with rank
    def getRankProbs(self, initialSize,rank):
            numerator = (initialSize) - rank +1
            denominator = initialSize * (initialSize+1) /2
            return((numerator/denominator))


    def initialRun(self):
        print("running",self.nRaces, "iterated races")
        self.updateBJ()
        sampleSize = self.getSampleSize()
        l = []
        for p in self.params:
            l.append(choices(p.values, k = sampleSize))
        self.grid = list(map(tuple, zip(*l)))
        initialSet = self.grid
        f = self.features
        L = self.labels
        pars = {}
        results = []
        for row in range(len(initialSet)):
            for column in range(len(self.names)):
                pars[self.names[column]] = (initialSet[row][column])
            clf = Classifier(self.method, pars)
            self.Bused += 1
            print("calculating:",(row + 1),"out of:", len(initialSet))
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
            results.append((row, res))
        self.scores = np.asarray([x[1] for x in results])
        self.ranks = ss.rankdata(scores)
        self.eliteIndices = np.where(self.ranks >= np.mean(self.ranks) - 0.5*np.std(self.ranks))
        elites = np.asarray(initialSet)[self.eliteIndices].tolist()
        self.grid = elites
        self.j = 2

    def run(self):
        self.initialRun()
        while(self.Bused <= self.budget and self.j <= self.nRaces):
            self.updateBJ()
            eliteRanks = np.asarray(self.ranks)[self.eliteIndices]
            worstIndex = self.scores[self.eliteIndices].argmin()
            print(len(self.ranks))
            print(len(self.grid[0]))
            parent = choices(self.grid, [x/(len(self.grid) + 1) for x in eliteRanks])
            sds = np.asarray(np.std(self.grid, axis = 0))

            for i in range(self.getSampleSize() - len(self.grid)):
                val = (np.absolute(np.random.normal(parent,sds)).tolist())
                valflat = list(itertools.chain(*val))
                self.grid.append(valflat)
            self.grid[:] = [tuple(l) for l in self.grid]
            print(self.grid)

            for row in range(len(self.grid)):
                t = tuple()
                for column in range(len(self.params)):
                    print(self.grid[row][column])
                    if (self.types[column] == True):
                        t = (*t,int(math.ceil(self.grid[row][column])))
                    else:
                        t = (*t,self.grid[row][column])
                self.grid[row] = t

            f = self.features
            L = self.labels
            pars = {}
            results = []
            for row in range(len(self.grid)):
                for column in range(len(self.names)):
                    pars[self.names[column]] = (self.grid[row][column])
                clf = Classifier(self.method, pars)
                self.Bused += 1
                print("calculating:",(row + 1),"out of:", len(self.grid))
                scores = []
                for ids,(train_index, test_index) in enumerate(self.sampler.method.split(f)):
                    X_train, X_test = f[train_index], f[test_index]
                    y_train, y_test = L[train_index], L[test_index]
                    clf.train(features = X_train, labels = y_train)
                    preds = clf.predict(X_test)
                    scores.append(self.metric(preds, y_test))
                res = sum(scores)/len(scores)
                print(res)
                results.append((row, res))
            self.scores = np.asarray([x[1] for x in results])
            self.ranks = ss.rankdata(scores)
            self.eliteIndices = np.where(self.ranks >= np.mean(self.ranks) - 0.5 * np.std(self.ranks))
            self.bestScore = self.scores[self.scores.argmax()]
            self.bestTry =self.scores.argmax()
            for col in range(len(self.names)):
                self.bestPars[self.names[col]] = self.grid[self.bestTry][col]
            self.grid = np.asarray(self.grid)[self.eliteIndices].tolist()
            self.j += 1

    def save(self, path = "race.obj"):
        filehandler = open(path,'wb')
        pickle.dump(self, filehandler)


            #then we add a j

            # update the grid

            # then we do a new sample




        # figure this out

        #print(self.friedman(ranks, sampleSize, self.grid))
        # plan: make a quantile




#    def run(self):
#        self.ran = True
#
#        pars = {}
#
#        f = self.features
#        L = self.labels
#        indices = np.random.randint(1, len(self.grid)-1, size = self.iters)
#        for row in indices:
#            for column in range(len(self.names)):
#                pars[self.names[column]] = (self.grid[row][column])
#            clf = Classifier(self.method, pars)
#            clf()
#            scores = []
#            for ids,(train_index, test_index) in enumerate(self.sampler.method.split(f)):
#                X_train, X_test = f[train_index], f[test_index]
#                y_train, y_test = L[train_index], L[test_index]
#                clf.train(features = X_train, labels = y_train)
#                # I would like to get the clf.predict method working
#                preds = clf.predict(X_test)
#                scores.append(self.metric(preds, y_test))
#            res = sum(scores)/len(scores)
#            print(res)
#            self.results.append(res)
#        self.bestScore = max(self.results)
#        self.bestTry = self.results.index(self.bestScore)
#        for col in range(len(self.names)):
#            self.bestPars[self.names[col]] = self.grid[self.bestTry][col]
#
#    def save(self, path = "randomSearch.obj"):
#        filehandler = open(path,'wb')
#        pickle.dump(self, filehandler)
#
#    def __call__(self):
#        if(self.ran):
#            return(self.method,
#            self.names,
#            self.grid,
#             self.results,
#                self.bestPars,
#            self.bestScore)
#        else:
#            return(self.method,
#            self.names,
#            self.grid)
#
#    def print(self):
#        if(self.ran):
#            print("----------------------")
#            print("model: ", self.method)
#            print("----------------------")
#            print("parameters: \n", self.names)
#            print("----------------------")
#            print("space:\n", self.grid)
#            print("----------------------")
#            print("results:\n", self.results)
#            print("----------------------")
#            print(
#                "Best Paramter Set:",
#                self.bestPars
#            )
#            print("----------------------")
#            print(" Best Score:", self.bestScore)
#        else:
#            print("----------------------")
#            print("model: ", self.method)
#            print("----------------------")
#            print("parameters: \n", self.names)
#            print("----------------------")
#            print("space:\n", self.grid)
#            print("----------------------")
#



#randomCase =  tuneRandom(rf,
#         X,
#         Y,
#         cv, f1_score,500,
#         integerParam("n_estimators", 1, 100, lambda x: 10*x),
#         integerParam("max_depth", 1,20,transFun = lambda x: x**2),
#                      integerParam("max_leaf_nodes", 1, 100, lambda x: 10*x),
#                         floatParam("min_impurity_decrease", 0, 49, lambda x: x/100)
#            )
#raceCase =  tuneIrace(rf,
#         X,
#         Y,
#         cv, f1_score,500,
#         integerParam("n_estimators", 1, 100, lambda x: 10*x),
#         integerParam("max_depth", 1,20,transFun = lambda x: x**2),
#                      integerParam("max_leaf_nodes", 1, 100, lambda x: 10*x),
#                         floatParam("min_impurity_decrease", 0, 49, lambda x: x/100)
#            )
#raceCase.mu = 0.3
#tune = Tuner(raceCase, randomCase)
#
#tune.run()
#tune.save()
##raceCase.run()
#
#print(
#    "random Search: ",tune.results[1].bestScore
#)
#print(
#    "Race: ",tune.results[0].bestScore
#)




# accepts a lower bound, upper bound, and transformation function
# yes i know list comprehensions exist but

#cv = Resampling("cv")
#rf = Classifier(RandomForestClassifier)

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
