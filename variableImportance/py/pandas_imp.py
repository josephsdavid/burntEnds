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
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns = ['MEDV'])

def split(df, p_train = 0.75, random_state = 0):
    train = df.sample(frac = p_train, random_state = random_state)
    test = df.drop(train.index)
    return(train, test)

(X_train, X_test), (y_train, y_test) = (split(x) for x in [X, y])

lm =  LinearRegression()
knn = KNeighborsRegressor(7)
rf = RandomForestRegressor()
mods = [lm, knn, rf]
for m in mods:
    m.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error as loss_mse

loss_mse(y_test, lm.predict(X_test))
loss_mse(y_test, rf.predict(X_test))




def permutation_importance(model, x, y, loss, base = False, x_train = None, y_train = None, kind = "prop", n_rounds = 5):
    explan = x.columns
    baseline = loss(y, model.predict(x))
    res = {k:[] for k in explan}
    if (base is True):
        res["baseline"] = []
    for n in range(0, n_rounds):
        for i in range(0, len(explan)):
            col = explan[i]
            x_temp = x.copy()
            x_temp[col] =  np.random.permutation(x_temp[col])
            if (kind is not "prop"):
                res[col].append(loss(y, model.predict(x_temp)) -  baseline)
            else:
                res[col].append(loss(y, model.predict(x_temp)) /  baseline)
        if (base is True):
            x_temp = x.copy()
            x_train2 = x_train.copy()
            # this is not right
            x_temp["baseline"] = np.clip(np.random.normal(size = len(x_temp)), -1., 1.)
            x_train2["baseline"] = np.clip(np.random.normal(size = len(x_train2)), -1., 1.)
            mod2 = type(model)()
            mod2.fit(x_train2, y_train)
            if (kind is not "prop"):
                res["baseline"].append(loss(y, mod2.predict(x_temp)) -  baseline)
            else:
                res["baseline"].append(loss(y, mod2.predict(x_temp)) /  baseline)
    return(pd.DataFrame.from_dict(res))


permutation_importance(lm, X_test, y_test, loss_mse, True,  X_train, y_train)

def get_name(obj):
    name =[x for x in globals() if globals()[x] is obj][0]
    return(name)

imps = {}
for m in mods:
    imps[get_name(m)] = permutation_importance(m, X_test, y_test, loss_mse, True,  X_train, y_train, n_rounds = 100)



plt.style.use("ggplot")
def plot(df, ax = None, color = 'blue'):
    df1 = (df.apply(stats.mean, 0, result_type = "broadcast")).drop(df.index[1:])
    df_temp = df1.loc[:, df.columns != 'baseline']
    df2 = df_temp.melt(var_name = 'variable', value_name = 'importance')
    df2 = df2.sort_values(by = "importance")
    df.sort_index(axis = 1, ascending = False)
    #ax = sns.barplot(x = 'cols', y = 'vals', data = df2, label = "variable_importance")
    df2.plot(kind = 'barh', x = 'variable', y = 'importance', width = 0.8, ax = ax, color = color)
    for n in df.columns:
        if n is "baseline":
            plt.axvline(x = df[n][0])
            plt.annotate('baseline',
                         xy = (df[n][0], 1),
                         xytext = (df[n][0] + 0.4, 3),
                         arrowprops = dict(facecolor = 'black',
                                           shrink = 0.05),
                         bbox = dict(boxstyle = "square", fc = (1,1,1)))

fig = plt.figure()
for i in range(len(imps.keys())):
    ax = fig.add_subplot(len(list(imps.keys())),1, i+1)
    #c = cm.Paired(i/len(imps.keys()), 1)
    c = sns.color_palette("hls", i+1)[i]
    plot(imps[list(imps.keys())[i]], ax = ax, color = c)
    ax.set_title(list(imps.keys())[i])
plt.show()
