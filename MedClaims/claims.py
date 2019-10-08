import csv
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
import pickle
from collections import  defaultdict
import ParamTuning as pt
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# read in the csv
def readCSV(path):
    with open(path, mode='r') as infile:
        reader = csv.DictReader(infile)
        result = {}

        for row in reader:
            for column, value in row.items():
                result.setdefault(column, []).append(value)
    return(result)


# save the dict so we dont have to read the csv repeatedly

def save(obj,path):
    filehandler = open(path, 'wb')
    pickle.dump(obj, filehandler)


# load the objects
def load(filename):
    with open(filename, 'rb') as fileObj:
        raw_data = fileObj.read()
        return(pickle.loads(raw_data))



# claims = readCSV('claim.sample.csv')

# save(claims, 'claims.obj')

claims = load('claims.obj')

# print(claims.keys())


# regex we will use repeatedly, so we save it
reg = re.compile('^[Jj]')

# 1A

def countJCodes(regex, obj):
    l = obj["Procedure.Code"]
    return(list(filter(regex.match, l)))

print(
    len(countJCodes(reg, claims))+1
)

# 51030 claim codes start with J. note that just searching for J returns same
# result, meaning no typos

# 1B

# get indices matching regex
def getIndex (regex, obj, col):
    l = obj[col]
    idx = [i for i, item in enumerate(l) if regex.match(item)]
    return(idx)

# build a dict matching indices
def build_dict(seq, key):
    return dict((d[key], dict(d, index=index)) for (index, d) in enumerate(seq))

def makeDict(regex, obj, col):
    idx = getIndex(regex, obj, col)
    res = {key:[value[i] for i in idx] for key, value in obj.items()}
    return(res)

# make j claims dict
jclaims = makeDict(reg, claims, 'Procedure.Code')

# now filter that down to in network

inn = re.compile("I")

inNetJ = (makeDict(inn, jclaims, 'In.Out.Of.Network'))

# convert a list of strings to floats

def toFloat(obj, col):
    return[float(i) for i in obj[col]]

def sumCol(obj, col):
    l = toFloat(obj, col)
    return(sum(l))

print(sumCol(inNetJ, 'Provider.Payment.Amount'))

# 2417220.96 was payed to in network providors by j claim patients


# 1c

# make a list of lists "rowise" on the dict, using jclaims and column
# specification

def getTwoCols(obj, colA, colB):
    l1 = obj[colA]
    l2 = obj[colB]
    #return[ [l1[i], l2[i]] for i in range(len(l1)) ]
    return(list(map(list,zip(l1,l2))))


def collapseSum(obj, grouping, other):
    l = getTwoCols(obj, grouping, other)
    res = defaultdict(float)
    for code, value in l:
        res[code] += float(value)
    return(res.items())


def getTop5(obj, grouping, value):
    l = collapseSum(obj, grouping, value)
    return(sorted(l, key =  lambda x: x[1], reverse = True)[:5])


print(getTop5(jclaims, 'Procedure.Code','Provider.Payment.Amount'))


# Number 2

# first make a provider dict

def makeColDict(obj, keys, values):
    l = getTwoCols(obj, keys, values)
    d = defaultdict(list)
    for i in l:
        d[i[0]] += i[1:]
    return(d)

#print(makeColDict(jclaims, 'Provider.ID','Provider.Payment.Amount'))

# next we need to write a function that counts zero vs not zero
def countPayments(l):
    zCount = 0
    nzCount = 0
    for i in l:
        if float(i) == 0:
            zCount += 1
        else:
            nzCount += 1
    return(nzCount, zCount)

d = makeColDict(jclaims, 'Provider.ID','Provider.Payment.Amount')

test = d[list(d.keys())[0]]

print(countPayments(test))


def makePaymentDict(obj, keys, values):
    d = makeColDict(obj, keys, values)
    res = {k: countPayments(d[k]) for k in d.keys()}
    return(res)

print(list(makePaymentDict(jclaims, 'Provider.ID','Provider.Payment.Amount').values()))
def plotPayments(obj = jclaims, keys = 'Provider.ID', values = 'Provider.Payment.Amount'):
    d = makePaymentDict(obj, keys, values)
    plt.scatter(*zip(*list(d.values())))
    # its too much work to make the labels, ill do that later
    plt.xlabel('Paid')
    plt.ylabel('unpaid')
    plt.title('Paid vs unpaid Jclaims')
    for k in d.keys():
        plt.annotate(k, d[k], size = 6)
    plt.show()

#plotPayments()

# Notes: we can see that A) there are way more unpaid than paid in general, and
# it seems to scale positively. It is very concerning, there are basically zero
# companies with more unpaid than paid, while most of them have a ridiculous
# amount more unpaid than paid



# 3A

# Make a new jdict where we update the types of the data

def convertItem(obj, item, to):
    res = list(map(to, obj[item]))
    return(res)


jclaims['unpaid'] = convertItem(jclaims, 'Provider.Payment.Amount', lambda x: int(float(x) == 0))

# get unpaid percentage

print(
    sum(jclaims['unpaid']) / (len(jclaims['unpaid']) + 1)
)

# 88% unpaid

# first thing to realize is we have a major class imbalance, and that these
# columns have the wrong type

Y = np.array(jclaims['unpaid'])

# Visualize all numeric data:

import seaborn as sns

def boxplot(x,y, d, log = False):
    if log == False:
        sns.boxplot(d[x],list(map(float,d[y])))
    else:
        sns.boxplot(d[x],list(map(lambda x: np.log((float(x))),d[y])))


#import pdb; pdb.set_trace()  # XXX BREAKPOINT

#boxplot('unpaid', 'Claim.Charge.Amount',jclaims, log = True)

#numerics = ["Subscriber.Payment.Amount", "Claim.Charge.Amount"]

#for n in numerics:
#    plt.figure()
#    boxplot('unpaid', n, jclaims)
#    plt.title(n)
#
#plt.show()

# these are weird features, so we will encode them as binary variables

jclaims['Subscriber.Payed'] = convertItem(jclaims, 'Subscriber.Payment.Amount', lambda x: int(float(x) == 0))


#for j in jclaims.keys():
#    if (j == 'unpaid' or
#        j == "V1" or j == "Claim.Number" or
#        j == "Claim.Line.Number" or
#        j == "Member.Id" or j == "Claim.Charge.Amount" or
#        j == 'Subscriber.Payment.Amount' or
#        j == 'Provider.Payment.Amount'):
#            pass
#    else:
#        plt.figure()
#        sns.countplot(x = jclaims[j], hue = jclaims['unpaid'])
#        plt.title(j)
#        path = j + '.png'
#        plt.savefig(path, bbox_inches = "tight")







# turn the dict into arrays:


# Use dict vectorizer because we have never used it

def encoder(obj):
    res = {}
    labeler = LabelEncoder()
    onehot = OneHotEncoder(categories = 'auto', sparse = False)
    for j in obj:
        if (j == 'unpaid' or
            j == 'Subscriber.Payed' or
            j == "V1" or j == "Claim.Number" or
            j == "Claim.Line.Number" or
            j == "Member.ID" or j == "Claim.Charge.Amount" or
            j == 'Subscriber.Payment.Amount' or
            j == 'Provider.Payment.Amount' or
            j == 'Subscriber.Payed' or
            j == 'Claim.Subscriber.Type' or
            j == 'Subgroup.Index' or
            j == 'Denial.Reason.Code' or
            j == "Place.Of.Service.Code"):
            pass
        else:
            inted = labeler.fit_transform(obj[j])
            oneEnc = onehot.fit_transform(inted.reshape(len(inted), 1))
            res[j] = oneEnc
    res['Claim.Charge.Amount'] = np.array(list(map(float,obj['Claim.Charge.Amount']))).reshape(len(obj['Claim.Charge.Amount']),1)
    res['Subscriber.Index'] = np.array(list(map(float,obj['Subscriber.Index']))).reshape(len(obj['Subscriber.Index']),1)
    #res['Provider.Payment.Amount'] = np.array(list(map(float,obj['Provider.Payment.Amount']))).reshape(len(obj['Claim.Charge.Amount']),1)
    return(res)


x = encoder(jclaims)
#print(x)
#
#for j in jclaims:
#    if x == 'unpaid' or x == 'Provider.Payment.Amount':
#        pass
#    else:
#        l =  np.array(jclaims[j])
#
#v = DictVectorizer()
#
#l = LabelEncoder()
#
#print(
#    l.fit_transform(x.items())
#)


# make a matrix of all our things:

#for xi in x.keys():

#print(np.concatenate(x.values(), axis = 1))

X = np.hstack(list(x.values()))



# Initial model: My features arent horrific, but there is much to be done
# Just a well tuned random forest. Obviously we will have to deal with the
# horrific class imbalance but lets see how we do right off the bat:

rf = pt.Classifier("randomforest")
cv = pt.Resampling("cv", folds = 3)

from sklearn.metrics import roc_auc_score, f1_score

from yellowbrick.classifier import ClassificationReport

rfRace = pt.tuneIrace(rf,
                      X,
                      Y,
                      cv, f1_score, 500,
                       pt.integerParam("n_estimators", 1, 100, lambda x: 10*x),
         pt.integerParam("max_depth", 1,20,transFun = lambda x: x**2),
                      pt.integerParam("max_leaf_nodes", 1, 100, lambda x: 10*x),
                         pt.floatParam("min_impurity_decrease", 0, 49, lambda x: x/100))

rfRace.mu = 0.05

rfRace.run()
rfRace.save()





