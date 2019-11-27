import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import statistics as stats
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
boston = load_boston()

#np.where(boston.feature_names == 'LSTAT')
X = boston.data
X_names = boston.feature_names
y = boston.target
y_name = np.array('MEDV')

def _add_column(arr, c):
    res = np.vstack([[arr[:,i] for i in range(arr.shape[1])], c]).T
    return(res)
def _add_rn(arr):
    rn = np.clip(np.random.normal(size = arr.shape[0]), -1., 1.)
    return(_add_column(arr, rn))


class explainer_array:
    def __init__(self, model, features, targets, feature_labels = None, target_labels = None):
        ## Come up with a nicer way to initialize with init args, maybe look at sklearn
        self.model = model
        self.X = features
        self.y = targets
        self.X_names = feature_labels
        self.target_labels = target_labels


class permutation_importance:
    def __init__(self,
                 explainer,
                 loss,
                 kind = "prop",
                 n_rounds = 5,
                 base = False,
                 X_train = None,
                 y_train = None):
        x_labels = explainer.feature_labels
        X_test = explainer.features
        y_test = explainer.targets
        self.importance = np.zeros((n_rounds, x_labels.shape[0]))
        self.loss = loss
        self.L_0 = self.loss(y_test, explainer.model.predict(X_test))
        for n in range(0, n_rounds):
            for i in range(x_labels.shape[0]):
                X_temp = X_test.copy()
                X_temp[:,i] = np.random.permutation(X_temp[:,i])
                var_loss = self.loss(y_test, explainer.model.predict(X_temp))
                if (kind is not "prop"):
                    self.importance[n,i] = var_loss - self.L_0
                else:
                    self.importance[n,i] = var_loss / self.L_0
        # figure out baseline, because this is ridiculously complicated

        # this is at the end
        # add random noise if we are going to have a baseline!

        if (base is True and X_train is not None and y_train is not None):
            x_labels2=np.hstack([x_labels, "baseline"])
            X_train, X_test = (_add_rn(arr) for arr in [X_train, X_test])
