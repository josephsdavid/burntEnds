# ParamTuning
A framework for tuning machine learning hyperparameters

## Features:

### Definition of parameter space

First, simple methods to define hyperparameter spaces are given. Note these are classes

```python
import ParamTuning as pt

# We include three classes of hyperparameters: discrete, integer, and float
# Usage:
# discreteParam("name",[values])

dsct = pt.discreteParam("criterion",["gini","entropy"])
dsct()

# integerParam("name", lower, upper, transFun(optional))

intgr = pt.integerParam("n_estimators", 10, 1000) # 10-1000 counting by 1s
intgr()

intgrtrans = pt.integerParam("n_estimators", 5, 50, lambda x: x*20) # 100-1000 counting by 20s
intgrtrans()

# floatParam("name", lower, upper, transFun(Required, otherwise by default it will just increase by 1s))

flt = pt.floatParam("min_impurity_decrease", 0, 49, lambda x: x/100)
flt()
```

### Unified API for model definition (WIP)

We can define classifiers all through a single API:

```python
rf = pt.Classifier("RandomForest", hyperPars = {"random_state":34})

lr = pt.Classifier("logreg", hyperPars = {"random_state":34})
```

Note that the hyperparameters are input as a dict, and have the same values as the normal sklearn stuff, so we do not have to learn anything new. Similarly:

```python
rf.train(X,y)

lr.train(X,y)

rf.predict(X_test)

lr.predict(X_test)
```

there is a simple API for training and predicting

#### TODO
[ ] Implement more models than rf and lr, its easy but its boring

### Cross-Validation Schemes

Similarly, there is a unified API for cross validation. Currently, cross validation and repeated cross validation are featured, while stratified cross validation is buggy at best.

```python
cv = pt.Resampling("cv", folds = 5)

rcv = pt.Resampling("repeatedcv", folds = 5, repeats = 10)
```


### Grid searches

First, a classic implementation of grid searches is included, with the `tuneGrid` class


```python
# we can use any metric
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score

import sklearn.datasets
X,Y = sklearn.datasets.load_breast_cancer(return_X_y=True)


grid = pt.tuneGrid()


grid =  pt.tuneGrid(rf,
         X,
         Y,
         cv, f1_score,
	 dsct,
	 intgrtrans,
	 flt,
         pt.integerParam("max_depth", 1,20,transFun = lambda x: x**2)
            )

grid()
grid.run()
grid()

print(grid.results)
print(grid.bestPars)
print(grid.bestScore)


rfTuned = Classifier("randomforest", grid.bestPars)
```

### Random Searches

Sometimes, a grid search is not the most efficient way to tune a hyperparameter, or sometimes our parameter space is too big to fit in memory, or we dont want to waste an entire day. in that case, we can do a random search:

```python
random =  pt.tuneRandom(rf,
         X,
         Y,
         cv, f1_score,1000,
         pt.integerParam("n_estimators", 1, 100, lambda x: 10*x),
         pt.integerParam("max_depth", 1,20,transFun = lambda x: x**2),
         pt.integerParam("max_leaf_nodes", 1, 100, lambda x: 10*x),
         pt.floatParam("min_impurity_decrease", 0, 49, lambda x: x/100)
            )
random.run()
```

In this case, we will have 1000 random samples of our grid search.


### Iterated Racing (work in progress, near completion)

One very cool and not often used method of optimization is iterated racing. Basically, what it does, is it takes a relatively small random sample of the parameter space (the sample size is dependent upon the ONLY user defined parameter(optionally, it has a sane default value of 0.1)). It will then run all of those, make a prediction and evaluate using our metric, and record that result. It will then rank all of the parameters, and then eliminate ones more than 0.5 standard deviations less than the mean rank (this will be replaced with a friedman chi square test with post hoc tests as soon as i figure out how to implement it). The standard deviation of each of the hyperparameters of the survivors is recorded. The survivors are then sampled again, this time using a distribution which scales with rank (so weaker survivors are more likely to be selected), selecting a single "parent" set. Then, a new set of hyperparameters are derived from the parent, using a normal distribution with mean of the parent values and sd of the survivors. These are then combined with the survivors, to form a significantly smaller sample set. This process is repeated until we run out of our computational budget, the maximum number of races are completed, or we reach a final model. The idea of this is that we A) avoid being stuck in a local optima by checking the worst of the best, not the best of the best, B) as the number of races increase, our standard of deviation, or search scope, will narrow, fine tuning the parameters, and C) With each race, the cross validation sampling will be a bit different, meaning our winning model will have seen more instances of the dataset, and hopefully more robust when applied to new data with the derived hyperparameter set, and C) it is much faster and computationally cheaper than a grid search, a massive random search, simulated annealing, or bayesian optimization, and can produce comparable results.


```python
irace =  pt.tuneIrace(rf,
         X,
         Y,
         cv, f1_score,1000,
         pt.integerParam("n_estimators", 1, 100, lambda x: 10*x),
         pt.integerParam("max_depth", 1,20,transFun = lambda x: x**2),
         pt.integerParam("max_leaf_nodes", 1, 100, lambda x: 10*x),
         pt.floatParam("min_impurity_decrease", 0, 49, lambda x: x/100)
            )
irace.mu = 0.05 # defaults to 0.1, mu defines the breadth of the races, higher values of mu will sample a larger portion of the parameter set. This and the budget (in this case 500) are the only things you really need to worry about. The default values are completely sane

irace.nRaces = 5 # defaults to celing(2 + log2(nPars)), defines the depth of the search. higher values will run more races. The default values ensure that the races finish beforeyou reach your computational limit, I wouldnt mess with this too much unless you just want to there is no need

irace.run()

print(irace.bestScore)
print(random.bestScore)
print(irace.bestPars)
print(random.bestPars)
rfRaced = pt.Classifier("RandomForest", irace.bestPars)
```


### Tuning multiple models:

It is super easy to tune and compare multiple models:


```python
randomCase =  tuneRandom(rf,
         X,
         Y,
         cv, f1_score,500,
         integerParam("n_estimators", 1, 100, lambda x: 10*x),
         integerParam("max_depth", 1,20,transFun = lambda x: x**2),
                      integerParam("max_leaf_nodes", 1, 100, lambda x: 10*x),
                         floatParam("min_impurity_decrease", 0, 49, lambda x: x/100)
            )
raceCase =  tuneIrace(rf,
         X,
         Y,
         cv, f1_score,500,
         integerParam("n_estimators", 1, 100, lambda x: 10*x),
         integerParam("max_depth", 1,20,transFun = lambda x: x**2),
                      integerParam("max_leaf_nodes", 1, 100, lambda x: 10*x),
                         floatParam("min_impurity_decrease", 0, 49, lambda x: x/100)
            )

tune = Tuner(raceCase, randomCase)

tune.run()
tune.save()

print(
    "random Search: ",tune.results[1].bestScore
)
print(
    "Race: ",tune.results[0].bestScore
)
```

### Saving searches:

All searches have a `.save()` method, where you can optionally specify the directory to save. Otherwise they are saved to your working directory. There is also a pt.load function:

```python

# raceCase was ran in the Tuner.run() function
raceCase.save("race.obj")

rfRace = pt.load("race.obj")
```

### Visualising





