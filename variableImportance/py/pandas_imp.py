import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston, fetch_california_housing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import statistics as stats
import matplotlib.cm as cm

cruise = pd.read_csv("https://github.com/bot13956/ML_Model_for_Predicting_Ships_Crew_Size/raw/master/cruise_ship_info.csv")




X = cruise.loc[:, cruise.columns != "crew"]
X = X.loc[:, X.columns != "Ship_name"]
X = X.loc[:, X.columns != "Cruise_line"]
y = cruise.loc[:, cruise.columns == "crew"]


def split(df, p_train = 0.75, random_state = 0):
    train = df.sample(frac = p_train, random_state = random_state)
    test = df.drop(train.index)
    return(train, test)

(X_train, X_test), (y_train, y_test) = (split(x) for x in [X, y])

lm =  LinearRegression()
knn = KNeighborsRegressor(7)
rf = RandomForestRegressor(n_estimators = 100)
mods = [lm, knn, rf]
for m in mods:
    m.fit(X_train, y_train)

rf.feature_importances_
X_train.columns[rf.feature_importances_ > 0.3 ]

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
    imps[get_name(m)] = permutation_importance(m, X_test, y_test, loss_mse, True,  X_train, y_train, n_rounds = 5)



plt.style.use("seaborn-whitegrid")

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


def _normalize(x):
    num = x-x.min()
    den = x.max() - x.min()
    return(num/den)

def pdp_var(model, x, var):
    explan = sorted(x[var])
    preds = []
    for i in range(0, len(explan)):
        tmp = []
        X_tmp = x.copy()
        # pandas is dumb
        val = np.asarray(explan)[i]
        X_tmp[var] = val
        preds.append(model.predict(X_tmp))
    preds = np.asarray(preds).reshape(len(x), len(explan))
    pv = preds.mean(axis = 0)
    return(explan, pv)


def pdp_df(model, x):
    res = {}
    for c in X.columns:
        print("calculating pdp for:" + get_name(model) + " column:" + c)
        d = pd.DataFrame()
        d["Value"], d["Average Prediction"] = pdp_var(model, X_test, c)
        res[c] = d
    return(res)

def pdp_importance(model, x):
    pdpdf = pdp_df(model, x)
    v = {k:np.std(pdpdf[k]["Average Prediction"]) for k in pdpdf.keys()}
    return(v)




pdp_imps = {get_name(m):pdp_importance(m, X_test) for m in mods}

attempt = pdp_df(rf, X_test)

plt.style.use("seaborn-whitegrid")
fig = plt.figure()
for k in range(0,len(attempt.keys())):
    ax = fig.add_subplot(2,3, k+1)
    #c = cm.Paired(i/len(imps.keys()), 1)
    c = sns.color_palette("hls", k+1)[k]
    df = attempt[list(attempt.keys())[k]]
    sns.lineplot(x = "Value", y = "Average Prediction", ax = ax, color = c, data = df)
    ax.set_title(list(attempt.keys())[k])
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
plt.show()

def plot_pdp_imp(d, ax = None, color = "blue"):
    df = pd.DataFrame()
    df["Variable"] = d.keys()
    df["Importance"] = d.values()
    df.sort_values("Importance").plot(kind = "barh", x = "Variable", y = "Importance", width = 0.8, ax = ax, color = c)

fig = plt.figure()
for i in range(len(pdp_imps.keys())):
    ax = fig.add_subplot(len(list(pdp_imps.keys())),1, i+1)
    #c = cm.Paired(i/len(imps.keys()), 1)
    c = sns.color_palette("hls", i+1)[i]
    plot_pdp_imp(pdp_imps[list(pdp_imps.keys())[i]], ax = ax, color = c)
    ax.set_title(list(pdp_imps.keys())[i])
plt.show()

# note the failure of pdp Because of correlation

## ICE land


def ICE_var(model, x, var):
    explan = sorted(x[var])
    med = explan
    preds = []
    for i in range(0, len(explan)):
        tmp = []
        X_tmp = x.copy()
        # pandas is dumb
        val = np.asarray(explan)[i]
        X_tmp[var] = val
        preds.append(model.predict(X_tmp))
    preds = np.asarray(preds).reshape(len(x), len(explan))
    return(explan, preds)

iceEx, icetempt = ICE_var(rf, X, "Tonnage")
icetempt.shape

iceEx[0]

def plotICE(ex, values, ax = None):
    mid = math.floor((len(ex))/2)
    for i in range(0, len(iceEx)):
        if (ax is None):
            plt.plot(ex,values[mid,:] - values[i,:], 'black', alpha = 0.1)
            plt.plot(ex, values[mid,:] - values.mean(0), "orange", alpha = 0.6)
        else:
            ax.plot(ex,values[mid,:] - values[i,:], 'black', alpha = 0.1)
            ax.plot(ex, values[mid,:] - values.mean(0), "#d7f500", alpha = 0.6)

plotICE(iceEx, icetempt)

plt.show()

fig = plt.figure()
for i in range(len(X.columns)):
    iceX, iceY = ICE_var(rf, X, X.columns[i])
    ax = fig.add_subplot(2,3, i+1)
    plotICE(iceX, iceY, ax = ax)
    ax.set_title(X.columns[i])

plt.show()

icetempt.shape

np.std(icetempt, 0).mean()

def ICE_vips(model, x):
    imps = {}
    for i in range(len(x.columns)):
        icex, icey = ICE_var(model, x, x.columns[i])
        imps[x.columns[i]] = (np.std(icey, 0)).mean()
    return(imps)

ice_imps = {get_name(m): ICE_vips(m, X) for m in mods}

def plot_ICE_imp(d, ax = None, color = "blue"):
    df = pd.DataFrame()
    df["Variable"] = d.keys()
    df["Importance"] = d.values()
    df.sort_values("Importance").plot(kind = "barh", x = "Variable", y = "Importance", width = 0.8, ax = ax, color = c)

fig = plt.figure()
for i in range(len(ice_imps.keys())):
    ax = fig.add_subplot(len(list(ice_imps.keys())),1, i+1)
    #c = cm.Paired(i/len(imps.keys()), 1)
    c = sns.color_palette("hls", i+1)[i]
    plot_ICE_imp(ice_imps[list(ice_imps.keys())[i]], ax = ax, color = c)
    ax.set_title(list(ice_imps.keys())[i])
plt.show()
