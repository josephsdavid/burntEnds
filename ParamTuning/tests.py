import ParamTuning as pt
import sklearn.datasets
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score # other metrics?
X,Y = sklearn.datasets.load_breast_cancer(return_X_y=True)

rf = pt.Classifier("RandomForest", hyperPars = {"random_state":34})

cv = pt.Resampling("cv", folds = 3)


raceCase =  pt.tuneIrace(rf,
         X,
         Y,
         cv, f1_score,500,
         pt.integerParam("n_estimators", 1, 100, lambda x: 10*x),
         pt.integerParam("max_depth", 1,20,transFun = lambda x: x**2),
         pt.integerParam("max_leaf_nodes", 1, 100, lambda x: 10*x),
         pt.floatParam("min_impurity_decrease", 0, 49, lambda x: x/100)
            )
raceCase.mu = 0.05
#raceCase.run()

#raceCase.save()

rfRace = pt.load("race.obj")



bestClass = pt.Classifier("RandomForest", hyperPars = raceCase.bestPars)

print(bestClass())

print(raceCase.bestPars)

