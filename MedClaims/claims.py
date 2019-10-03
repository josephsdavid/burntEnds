import csv
import re
import numpy as np
import pickle

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

