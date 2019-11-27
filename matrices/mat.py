import numpy as np

# Decision making with Matrices
# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations.
# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.
# Then you should decided if you should split into two groups so eveyone is happier.
# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.
# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of
# decsion making problems that are currently not leveraging machine learning.
# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.

people = {'Jane': {'willingness to travel': 0.1596993,
                  'desire for new experience':0.67131344,
                  'cost':0.15006726,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.01892,
                  },
          'Bob': {'willingness to travel': 0.63124581,
                  'desire for new experience':0.20269888,
                  'cost':0.01354308,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.15251223,
                  },
          'Mary': {'willingness to travel': 0.49337138 ,
                  'desire for new experience': 0.41879654,
                  'cost': 0.05525843,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.03257365,
                  },
          'Mike': {'willingness to travel': 0.08936756,
                  'desire for new experience': 0.14813813,
                  'cost': 0.43602425,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.32647006,
                  },
          'Alice': {'willingness to travel': 0.05846052,
                  'desire for new experience': 0.6550466,
                  'cost': 0.1020457,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.18444717,
                  },
          'Skip': {'willingness to travel': 0.08534087,
                  'desire for new experience': 0.20286902,
                  'cost': 0.49978215,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.21200796,
                  },
          'Kira': {'willingness to travel': 0.14621567,
                  'desire for new experience': 0.08325185,
                  'cost': 0.59864525,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.17188723,
                  },
          'Moe': {'willingness to travel': 0.05101531,
                  'desire for new experience': 0.03976796,
                  'cost': 0.06372092,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.84549581,
                  },
          'Sara': {'willingness to travel': 0.18780828,
                  'desire for new experience': 0.59094026,
                  'cost': 0.08490399,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.13634747,
                  },
          'Tom': {'willingness to travel': 0.77606127,
                  'desire for new experience': 0.06586204,
                  'cost': 0.14484121,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.01323548,
                  }
          }



restaurants  = {'flacos':{'distance' : 2,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                        },
              'Joes':{'distance' : 5,
                        'novelty' : 1,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      },
              'Poke':{'distance' : 4,
                        'novelty' : 2,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },
              'Sush-shi':{'distance' : 4,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },
              'Chick Fillet':{'distance' : 3,
                        'novelty' : 2,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                      },
              'Mackie Des':{'distance' : 2,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      },
              'Michaels':{'distance' : 2,
                        'novelty' : 1,
                        'cost': 1,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                      },
              'Amaze':{'distance' : 3,
                        'novelty' : 5,
                        'cost': 2,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },
              'Kappa':{'distance' : 5,
                        'novelty' : 1,
                        'cost': 2,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      },
              'Mu':{'distance' : 3,
                        'novelty' : 1,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      }
}


# here i define some functions that we can construct a numpy array with names
# and columns etc

# this gets the highest level of keys, which will be either people or
# restaurants
def get_rownames(nested_dict):
    return(list(nested_dict.keys()))

# this gets the next level of keys, which will be the attributes of the previous
# thing (the row names). These will be columns in out matrix
def get_colnames(nested_dict):
    return(get_rownames(nested_dict[get_rownames(nested_dict)[0]]))

# This will get the nested values with a pretty sweet little comprehension
def get_values(nested_dict):
    res = [list(nested_dict[r].values()) for r in get_rownames(nested_dict)]
    return(res)

# we put all our constructor helpers in a list for some sweet generator
# comprehensions in a bit
constructors = [get_rownames, get_colnames, get_values]

class namedMatrix(object):
    """
    NamedMatrix: class

    Attributes: rownames, colnames, matrix, and swapped
    rownames display the row names, colnames display the column names
    matrix is a numpy array of the values from the dict
    swapped is a boolean saying whether the matrix is a flipped matrix or not
    If we pass in a nested dict (nested_dict), it will automatically construct
    the object given that dict. If we do not pass it in, we can define it using
    the row names, column names, and matrix manually
    """
    def __init__(self, rownames = None, colnames = None, values = None, swapped = False, nested_dict = None):
        if nested_dict is None:
            self.rownames = rownames
            self.colnames = colnames
            self.matrix = values
            self.swapped = swapped
        else:
            # this is so nice
            rownames, colnames, values = (c(nested_dict) for c in constructors)
            self.rownames = rownames
            self.colnames = colnames
            self.matrix = np.array(values)
            self.swapped = False
    def __call__(self):
        # print nicely because we are a human. I could use __repr__ or __str__
        # here but this is my habit
        print("\t", self.colnames)
        for r in self.rownames:
            print(r,"\t",np.round(self.matrix[self.rownames.index(r),:], 2))
    def T(self):
        # returns a namedMatrix with rows columns switched
        rownames, colnames = self.colnames, self.rownames
        # could also use np.swapaxes, not sure which is faster or more sensible
        matrix = self.matrix.T
        # just save this for sanity's sake
        swapped = not self.swapped
        # the reason for our weird constructur
        return(namedMatrix(rownames, colnames, matrix, swapped))
    # access row of namedMatrix nicely
    def row(self, idx):
        if isinstance(idx, str):
            ind = self.rownames.index(idx)
        else:
            ind = idx
        return(self.matrix[ind,:])
    # same with columns
    def col(self, idx):
        if isinstance(idx, str):
            ind = self.colnames.index(idx)
        else:
            ind = idx
        return(self.matrix[:,ind])




ppl = namedMatrix(nested_dict = people)
rs =  namedMatrix(nested_dict = restaurants)
# just to print things and affirm it works
list(o() for o in [ppl, rs])

# matrix multiplication function, extended to named Matrices
def matmul(M1, M2):
    if isinstance(M1, namedMatrix):
        x = M1.matrix
    else:
        x = M1
    if isinstance(M2, namedMatrix):
        y = M2.matrix
    else:
        y = M2
    res = np.matmul(x, y)
    return(res)

# choose a person and find the top restaurant for them, using linear
# combinations. I pick Tom. Each item in this vector is how appealing that
# restaurant is to tom based on his preferences

Tom = ppl.row("Tom")
toms_rs = matmul(rs, Tom)

def show_rest_scores(nm, lc):
    """
    display restaurant and the score for that restaurant.
    Inputs: namedMatrix, linear combination
    """
    for i in range(lc.shape[0]):
        print(nm.rownames[i]+":", lc[i])

show_rest_scores(rs, toms_rs)
# !!
# each entry represents the desirability of that restaurant for that person
# tom likes Joes a lot


matmul(ppl, rs.T())
# rows are people, columns are restaurants! so a_ij = how desirable restaurant j
# is to person i

def restaurant_rater(persons, ristorante):
    """
    restaurant rater: take in people and restaurants, return preference matrix
    """
    mat = matmul(persons, ristorante.T())
    rownames = persons.rownames
    colnames = ristorante.rownames
    return(namedMatrix(rownames, colnames, mat))

ratings = restaurant_rater(ppl, rs)
ratings()

scores = ratings.matrix.sum(0)
print(ratings.colnames)
print(np.round(scores, 2))

def get_winners(rates, fun,n):
    """
    get the top n of a rating matrix given a fun which returns indices
    """
    scores = rates.matrix.sum(0)
    ids = fun(scores)[:n]
    winScores = scores[ids]
    winners = np.array(rates.colnames)[ids]
    return winScores, winners

# since we are optimizing we are going to argsort and then reverses
# https://stackoverflow.com/a/16486305
top3_scores, top3 = get_winners(ratings, lambda x: np.argsort(x)[::-1], 3)
print(top3)
print(top3_scores)

bottom3_scores, bottom3 = get_winners(ratings, np.argsort, 3)
print(bottom3)
print(bottom3_scores)






