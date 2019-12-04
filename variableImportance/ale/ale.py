import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


from matplotlib import gridspec

def _ax_title(ax, title, subtitle):
	"""
	Prints title on figure.
	Parameters
	----------
	fig : matplotlib.axes.Axes
		Axes objet where to print titles.
	title : string
		Main title of figure.
	subtitle : string
		Sub-title for figure.
	"""
	ax.set_title(title + "\n" + subtitle)
	#fig.suptitle(subtitle, fontsize=10, color="#919191")

def _ax_labels(ax, xlabel, ylabel):
	"""
	Prints labels on axis' plot.
	Parameters
	----------
	ax : matplotlib.axes.Axes
		Axes object where to print labels.
	xlabel : string
		Label of X axis.
	ylabel : string
		Label of Y axis.
	"""
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)


def _ax_quantiles(ax, quantiles, twin='x'):
	"""
	Plot quantiles of a feature over opposite axis.
	Parameters
	----------
	ax : matplotlib.Axis
		Axis to work with.
	quantiles : array-like
		Quantiles to plot.
	twin : string
		Possible values are 'x' or 'y', depending on which axis to plot quantiles.
	"""
	print("Quantiles :", quantiles)
	if twin == 'x':
		ax_top = ax.twiny()
		ax_top.set_xticks(quantiles)
		ax_top.set_xticklabels(["{1:0.{0}f}%".format(int(i / (len(quantiles) - 1) * 100 % 1 > 0), i / (len(quantiles) - 1) * 100) for i in range(len(quantiles))], color="#545454", fontsize=7)
		ax_top.set_xlim(ax.get_xlim())
	elif twin =='y':
		ax_right = ax.twinx()
		ax_right.set_yticks(quantiles)
		ax_right.set_yticklabels(["{1:0.{0}f}%".format(int(i / (len(quantiles) - 1) * 100 % 1 > 0), i / (len(quantiles) - 1) * 100) for i in range(len(quantiles))], color="#545454", fontsize=7)
		ax_right.set_ylim(ax.get_ylim())

def _ax_scatter(ax, points):
	print(points)
	ax.scatter(points.values[:,0], points.values[:,1], alpha=0.5, edgecolor=None)

def _ax_grid(ax, status):
	ax.grid(status, linestyle='-', alpha=0.4)

#def _ax_labels(ax, label):


def _ax_boxplot(ax, ALE, cat, **kwargs):
	ax.boxplot(cat, ALE, **kwargs)

def _ax_hist(ax, x, **kwargs):
	sns.rugplot(x, ax=ax, alpha=0.2)




def _first_order_quant_plot(ax, quantiles, ALE, **kwargs):
	#ax.plot(quantiles, ALE, **kwargs)
	ax.plot((quantiles[1:] + quantiles[:-1]) / 2, ALE, **kwargs)
	#ax.scatter(quantiles, ALE, color="#1f77b4", s=12)


# https://github.com/blent-ai/ALEPython/blob/dev/alepython/ale.py#L107
def _first_order_ale_quant(predictor, train_set, feature, quantiles):
	"""Computes first-order ALE function on single continuous feature data.
	Parameters
	----------
	predictor : function
		Prediction function.
	train_set : pandas DataFrame
		Training set on which model was trained.
	feature : string
		Feature's name.
	quantiles : array-like
		Quantiles of feature.
	"""
    # preallocate ALE space
	ALE = np.zeros(len(quantiles) - 1)  # Final ALE function

	for i in range(1, len(quantiles)):
        # cut the train set in a little box between the quantiles
        # very nice
		subset = train_set[(quantiles[i - 1] <= train_set[feature]) & (train_set[feature] < quantiles[i])]

		# Without any observation, local effect on splitted area is null
		if len(subset) != 0:
            # lower bounf
			z_low = subset.copy()
            # upper bound
			z_up = subset.copy()
			# The main ALE idea that compute prediction difference between same data except feature's one
			z_low[feature] = quantiles[i - 1]
			z_up[feature] = quantiles[i]
			ALE[i - 1] += (predictor(z_up) - predictor(z_low)).sum() / subset.shape[0]


	ALE = ALE.cumsum()  # The accumulated effect
	ALE -= ALE.mean()  # Now we have to center ALE function in order to obtain null expectation for ALE function
	return(ALE)



def ale_plot(model, train_set, features, bins=10, monte_carlo=False, predictor=None, features_classes=None, **kwargs):
	"""Plots ALE function of specified features based on training set.
	Parameters
	----------
	model : object or function
		A Python object that contains 'predict' method. It is also possible to define a custom prediction function with 'predictor' parameters that will override 'predict' method of model.
	train_set : pandas DataFrame
		Training set on which model was trained.
	features : string or tuple of string
		A single or tuple of features' names.
	bins : int
		Number of bins used to split feature's space.
	monte_carlo : boolean
		Compute and plot Monte-Carlo samples.
	predictor : function
		Custom function that overrides 'predict' method of model.
	features_classes : list of string
		If features is first-order and is a categorical variable, plot ALE according to discrete aspect of data.
	monte_carlo_rep : int
		Number of Monte-Carlo replicas.
	monte_carlo_ratio : float
		Proportion of randomly selected samples from dataset at each Monte-Carlo replica.
	"""
	fig = plt.figure()
	if not isinstance(features, (list, tuple, np.ndarray)):
		features = np.asarray([features])

	if len(features) == 1:
		quantiles = np.percentile(train_set[features[0]], [1. / bins * i * 100 for i in range(0, bins + 1)])  # Splitted areas of feature

		if monte_carlo:
			mc_rep = kwargs.get('monte_carlo_rep', 50)
			mc_ratio = kwargs.get('monte_carlo_ratio', 0.1)
			mc_replicates = np.asarray([[np.random.choice(range(train_set.shape[0])) for _ in range(int(mc_ratio * train_set.shape[0]))] for _ in range(mc_rep)])
			for k, rep in enumerate(mc_replicates):
				train_set_rep = train_set.iloc[rep, :]
				if features_classes is None:
					mc_ALE = _first_order_ale_quant(model.predict if predictor is None else predictor, train_set_rep, features[0], quantiles)
					_first_order_quant_plot(fig.gca(), quantiles, mc_ALE, color="#1f77b4", alpha=0.06)

		if features_classes is None:
			ALE = _first_order_ale_quant(model.predict if predictor is None else predictor, train_set, features[0], quantiles)
			_ax_labels(fig.gca(), "Feature '{}'".format(features[0]), "")
			_ax_title(fig.gca(), "First-order ALE of feature '{0}'".format(features[0]),
				"Bins : {0} - Monte-Carlo : {1}".format(len(quantiles) - 1, mc_replicates.shape[0] if monte_carlo else "False"))
			_ax_grid(fig.gca(), True)
			_ax_hist(fig.gca(), train_set[features[0]])
			_first_order_quant_plot(fig.gca(), quantiles, ALE, color="black")
			_ax_quantiles(fig.gca(), quantiles)
		else:
			_ax_boxplot(fig.gca(), quantiles, ALE, color="black")
	elif len(features) == 2:
		quantiles = [np.percentile(train_set[f], [1. / bins * i * 100 for i in range(0, bins + 1)]) for f in features]

		if features_classes is None:
			ALE = _second_order_ale_quant(model.predict if predictor is None else predictor, train_set, features, quantiles)
			#_ax_scatter(fig.gca(), train_set.loc[:, features])
			_second_order_quant_plot(fig.gca(), quantiles, ALE)
			_ax_labels(fig.gca(), "Feature '{}'".format(features[0]), "Feature '{}'".format(features[1]))
			_ax_quantiles(fig.gca(), quantiles[0], twin='x')
			_ax_quantiles(fig.gca(), quantiles[1], twin='y')
			_ax_title(fig.gca(), "Second-order ALE of features '{0}' and '{1}'".format(features[0], features[1]),
				"Bins : {0}x{1}".format(len(quantiles[0]) - 1, len(quantiles[1]) - 1))




from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston, fetch_california_housing

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


ale_plot(rf, X_train, "Tonnage")

for c in X.columns:
    ale_plot(rf, X_test, c, bins = 20)
    plt.savefig(c+"ale.png")
