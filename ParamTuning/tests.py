import ParamTuning as pt
from sklearn.ensemble import RandomForestRegressor
import collections as cl
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score # other metrics?
X,Y = sklearn.datasets.load_breast_cancer(return_X_y=True)

# define sampling
cv = pt.Resampling("cv", folds = 3)

# define classifiers
rf = pt.Classifier("randomforest", hyperPars = {"random_state":34})
lr = pt.Classifier("logreg", hyperPars = {"random_state":34})

# run a grid search
#gridSearch =  pt.tuneGrid(rf,
#         X,
#         Y,
#         cv, roc_auc_score,
#         pt.integerParam("n_estimators", 2, 5, lambda x: 200*x),
#         pt.integerParam("max_depth", 2,9,transFun = lambda x: 2**x)
#            )
#gridSearch.run()
#gridSearch.save()





# run an iterated race
#raceCase =  pt.tuneIrace(rf,
#         X,
#         Y,
#         cv, f1_score,500,
#         pt.integerParam("n_estimators", 1, 100, lambda x: 10*x),
#         pt.integerParam("max_depth", 1,20,transFun = lambda x: x**2),
#         pt.integerParam("max_leaf_nodes", 1, 100, lambda x: 10*x),
#         pt.floatParam("min_impurity_decrease", 0, 49, lambda x: x/100)
#            )
#raceCase.mu = 0.05
#raceCase.run()
#
#raceCase.save()


# run a random search
randomSearch =  pt.tuneRandom(rf,
         X,
         Y,
         cv, f1_score,500,
         pt.integerParam("n_estimators", 1, 100, lambda x: 10*x),
         pt.integerParam("max_depth", 1,20,transFun = lambda x: x**2),
         pt.integerParam("max_leaf_nodes", 1, 100, lambda x: 10*x),
         pt.floatParam("min_impurity_decrease", 0, 49, lambda x: x/100)
            )

#randomSearch.run()
#
#randomSearch.save()


# define objects for multiple tunings
#rfRace =  pt.tuneIrace(rf,
#         X,
#         Y,
#         cv, f1_score,500,
#         pt.integerParam("n_estimators", 1, 100, lambda x: 10*x),
#         pt.integerParam("max_depth", 1,20,transFun = lambda x: x**2),
#                      pt.integerParam("max_leaf_nodes", 1, 100, lambda x: 10*x),
#                         pt.floatParam("min_impurity_decrease", 0, 49, lambda x: x/100)
#            )
#
#lrGrid =  pt.tuneGrid(lr,
#         X,
#         Y,
#                   cv, f1_score ,pt.integerParam("max_iter", 1, 10, lambda x: 10*x),
#                      pt.discreteParam("penalty",["l1","l2"] )
#                   )
#multiTune = pt.Tuner( rfRace, lrGrid)
#
#multiTune.run()
#multiTune.save()




rfRace = pt.load("race.obj")

modelm = pt.load("mutiTuner.obj")

# we can get these with indices, or by using the saved stuff, tuner updates
# global state so this is nice

g = pt.load("grid.obj")



bestClass = pt.Classifier("randomforest", hyperPars = rfRace.bestPars)

print(bestClass())

print(rfRace.bestPars)


print(g.grid)
print(g.names)

r = pt.load("randomSearch.obj")

# plotting not implemented for irace because it doesnt make sense
def makeDict(tuner):
    arr = np.array(tuner.grid)
    print(arr)
    d = {}
    for col in range(len(tuner.names)):
        d[tuner.names[col]] = list(arr[:,col])
    return(d)

def plotPars(obj):

    forest = pt.Classifier("randomforest", hyperPars = {"random_state":34})
    d = makeDict(obj)
    metric = obj.metric
    scores = obj.results
    names = list(d.keys())
    rd = {}
    # regression to figure out relationship between single hyperparameter
    for n in names:
        r = RandomForestRegressor(random_state = 6)
        r.fit(X = np.array(d[n]).reshape(-1,1), y = scores)
        p = r.predict((np.array(np.unique(d[n])).reshape(-1,1)))
        rd[n] = list(p)

    for n in names:
        plt.figure()
        plt.plot(np.unique(d[n]), rd[n])
        plt.plot(d[n], scores, 'o')
        plt.title(n)
    plt.show()
# kind of ugly but does the trick
plotPars(g)
