# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 20:06:31 2019

@author: Chris
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics
import matplotlib.cm as cm  #https://matplotlib.org/api/cm_api.html

# Decision making with Matrices

# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations.

# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.
# Then you should decided if you should split into two groups so eveyone is happier.

# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.

# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of
# decsion making problems that are currently not leveraging machine learning.

# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.


#to determine random values for weights
print(np.array([np.random.dirichlet(np.ones(5),size=1)]))



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

# Transform the user data into a matrix(M_people). Keep track of column and row ids.

                                       # convert each person's values to a list

peopleKeys, peopleValues = [], []
lastKey = 0
for k1, v1 in people.items():
    row = []

    for k2, v2 in v1.items():
        peopleKeys.append(k1+'_'+k2)
        if k1 == lastKey:
            row.append(v2)
            lastKey = k1

        else:
            peopleValues.append(row)
            row.append(v2)
            lastKey = k1


#here are some lists that show column keys and values
print(peopleKeys)
print(peopleValues)



peopleMatrix = np.array(peopleValues)

peopleMatrix.shape


# Next you collected data from an internet website. You got the following information.

#1 is bad, 5 is great

np.random.randint(5, size=4)+1

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


# Transform the restaurant data into a matrix(M_resturants) use the same column index.


restaurantsKeys, restaurantsValues = [], []

for k1, v1 in restaurants.items():
    for k2, v2 in v1.items():
        restaurantsKeys.append(k1+'_'+k2)
        restaurantsValues.append(v2)

#here are some lists that show column keys and values
print(restaurantsKeys)
print(restaurantsValues)

len(restaurantsValues)
#reshape to 2 rows and 4 columns

#converting lists to np.arrays is easy
restaurantsMatrix = np.reshape(restaurantsValues, (10,4))

restaurantsMatrix

restaurantsMatrix.shape

# Matrix multiplication
# Dot products are the matrix multiplication of a row vectors and column vectors (n,p) * (p,n)
#  shape example: ( 2 X 4 ) * (4 X 2) = 2 * 2
a = np.array([[1, 0], [0, 1]])
b = np.array([[1],[2]])

a.shape, b.shape

# when 2D arrays are involved, np.dot give the matrix product.
np.dot(a,b)

# documentation: https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
# intuition: https://www.mathsisfun.com/algebra/matrix-multiplying.html
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[7,8],[9,10],[11,12]])

c.shape, d.shape

np.dot(c,d)
# What is a matrix product?
# https://en.wikipedia.org/wiki/Matrix_multiplication
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html#numpy.matmul
# matmul give the matrix product, too.
np.matmul(a,b)


restaurantsMatrix.shape, peopleMatrix.shape
#However, this won't work...
np.matmul(restaurantsMatrix, peopleMatrix)

# The most imporant idea in this project is the idea of a linear combination.

# Informally describe what a linear combination is and how it will relate to our resturant matrix.

    #This is for you to answer! However....https://en.wikipedia.org/wiki/Linear_combination
    # essentially you are multiplying each term by a constant and summing the results.

# Choose a person and compute(using a linear combination) the top restaurant for them.
# What does each entry in the resulting vector represent?

print(peopleKeys)
print(peopleValues)

print(restaurantsKeys)
print(restaurantsValues)


restaurantsMatrix.shape, peopleMatrix.shape

#We need to swap axis on peopleMatrix
#newPeopleMatrix = np.swapaxes(peopleMatrix, 1, 0)

#https://docs.scipy.org/doc/numpy/reference/generated/numpy.swapaxes.html
newPeopleMatrix = np.swapaxes(peopleMatrix, 0, 1)

newPeopleMatrix.shape, restaurantsMatrix.shape
restaurantsMatrix.shape, newPeopleMatrix.shape

# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?
#Let's check our answers


np.set_printoptions(threshold=np.nan)

results = np.matmul(restaurantsMatrix, newPeopleMatrix)

results

#rows are restaurants, columns are people

# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?
# I believe that this is what John and  is asking for, sum by columns
np.sum(results, axis=1)


restaurantsKeys

# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.
# Do the same as above to generate the optimal resturant choice.
results



# Say that rank 1 is best

#reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
# Argsort returns the indices that would sort an array - https://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
# By default, argsort is in ascending order, but below, we make it in descending order and then add 1 since ranks start at 1
sortedResults = results.argsort()[::-1]
sortedResults


np.sum(sortedResults, axis=1)


temp = results.argsort()
ranks = np.arange(len(results))[temp.argsort()]+1

#compare ranks to results
results

np.sum(ranks, axis=1)
#Out[10]: array([55, 55, 55, 55, 55, 55, 55, 55, 55, 55], dtype=int64)


# From sillouhette analysis with K-means clustering

# using 3 clusters
#And their assigned clusters were : [2 1 1 0 2 0 0 0 2 1]
#'Jane', 'Bob', 'Mary', 'Mike', 'Alice', 'Skip', 'Kira', 'Moe', 'Sara', and 'Tom'

#groups = [2, 1, 1, 0, 2, 0, 0, 0, 2, 1]

#group 0 is Mike, Skip, Kira, and Moe
group0 = ranks[0:,[3,5,6,7]]

#group 1 is Bob, Mary, and Tom
group1 = ranks[0:,[1,2,9]]

#group 2 is Jane, Alice, and Sara
group2 = ranks[0:,[0,4,8]]

#then look at the sums for each group
#y=0 (flacos), y=1 (Joes), y=2 (Poke), y=3 (Sush-shi), y=4 (Chick Fillet),
#y=5 (Mackie Des), y=6 (Michaels), y=7 (Amaze), y=8 (Kappa), y=9 (Mu)

np.sum(group0, axis=1)
# group 0 wants to go to flacos or Chick Fillet (it is a tie)

np.sum(group1, axis=1)
# Group 1 wants to go to Joes

np.sum(group2, axis=1)
# Group 2 wants to go to Amaze



#first plot heatmap
#https://seaborn.pydata.org/generated/seaborn.heatmap.html
plot_dims = (12,10)
fig, ax = plt.subplots(figsize=plot_dims)
sns.heatmap(ax=ax, data=results, annot=True)
plt.show()

#remember a_ij is the score for a restaurant for a person
#x is the person, y is the restaurant

print(peopleKeys)
#x=0 (Jane), x=1 (Bob), x=2 (Mary), x=3 (Mike), x=4 (Alice),
#x=5 (Skip), x=6 (Kira), x=7 (Moe), x=8 (Sara), x=9 (Tom)

print(restaurantsKeys)
#y=0 (flacos), y=1 (Joes), y=2 (Poke), y=3 (Sush-shi), y=4 (Chick Fillet),
#y=5 (Mackie Des), y=6 (Michaels), y=7 (Amaze), y=8 (Kappa), y=9 (Mu)


#What is the problem if we want to do clustering with this matrix?


results.shape

#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA


peopleMatrix.shape

#we don't need to apply standard scaler since the data is already scaled
#sc = StandardScaler()
#peopleMatrixScaled = sc.fit_transform(peopleMatrix)

#The example PCA was taken from.
#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
pca = PCA(n_components=2)
peopleMatrixPcaTransform = pca.fit_transform(peopleMatrix)

print(pca.components_)
print(pca.explained_variance_)





#This function was shamefully taken from the below and modified for our purposes
#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# plot principal components
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

ax.scatter(peopleMatrixPcaTransform[:, 0], peopleMatrixPcaTransform[:, 1], alpha=0.2)
draw_vector([0, 0], [0, 1], ax=ax)
draw_vector([0, 0], [1, 0], ax=ax)
ax.axis('equal')
ax.set(xlabel='component 1', ylabel='component 2',
          title='principal components',
          xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
fig.show



# Now use peoplePCA for clustering and plotting
# https://scikit-learn.org/stable/modules/clustering.html
kmeans = KMeans(n_clusters=3)
kmeans.fit(peopleMatrixPcaTransform)

centroid = kmeans.cluster_centers_
labels = kmeans.labels_

print (centroid)
print(labels)


fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

#https://matplotlib.org/users/colors.html
colors = ["g.","r.","c."]
labelList = ['Jane', 'Bob', 'Mary', 'Mike', 'Alice', 'Skip', 'Kira', 'Moe', 'Sara', 'Tom']

for i in range(len(peopleMatrixPcaTransform)):
   print ("coordinate:" , peopleMatrixPcaTransform[i], "label:", labels[i])
   ax.plot(peopleMatrixPcaTransform[i][0],peopleMatrixPcaTransform[i][1],colors[labels[i]],markersize=10)
   #https://matplotlib.org/users/annotations_intro.html
   #https://matplotlib.org/users/text_intro.html
   ax.annotate(labelList[i], (peopleMatrixPcaTransform[i][0],peopleMatrixPcaTransform[i][1]), size=25)
ax.scatter(centroid[:,0],centroid[:,1], marker = "x", s=150, linewidths = 5, zorder =10)

plt.show()
#remember, that the order here is:

#x=0 (Jane), x=1 (Bob), x=2 (Mary), x=3 (Mike), x=4 (Alice),
#x=5 (Skip), x=6 (Kira), x=7 (Moe), x=8 (Sara), x=9 (Tom)


#cluster 0 is green, cluster 1 is red, cluster 2 is cyan (blue)




#Now do the same for restaurants

#The example PCA was taken from.
#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
restaurantsMatrix.shape

pca = PCA(n_components=2)
restaurantsMatrixPcaTransform = pca.fit_transform(restaurantsMatrix)

print(pca.components_)
print(pca.explained_variance_)


fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

ax.scatter(restaurantsMatrixPcaTransform[:, 0], restaurantsMatrixPcaTransform[:, 1], alpha=0.2)
draw_vector([0, 0], [0, 3], ax=ax)
draw_vector([0, 0], [3, 0], ax=ax)
ax.axis('equal')
ax.set(xlabel='component 1', ylabel='component 2',
          title='principal components',
          xlim=(-4, 4), ylim=(-4, 4))
fig.show



# Now use restaurantsMatrixPcaTransform for plotting
# https://scikit-learn.org/stable/modules/clustering.html
kmeans = KMeans(n_clusters=3)
kmeans.fit(restaurantsMatrixPcaTransform)

centroid = kmeans.cluster_centers_
labels = kmeans.labels_

print (centroid)
print(labels)


fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

#https://matplotlib.org/users/colors.html
colors = ["g.","r.","c."]
labelList = ['Flacos', 'Joes', 'Poke', 'Sush-shi', 'Chick Fillet', 'Mackie Des', 'Michaels', 'Amaze', 'Kappa', 'Mu']

for i in range(len(restaurantsMatrixPcaTransform)):
   print ("coordinate:" , restaurantsMatrixPcaTransform[i], "label:", labels[i])
   ax.plot(restaurantsMatrixPcaTransform[i][0],restaurantsMatrixPcaTransform[i][1],colors[labels[i]],markersize=10)
   #https://matplotlib.org/users/annotations_intro.html
   #https://matplotlib.org/users/text_intro.html
   ax.annotate(labelList[i], (restaurantsMatrixPcaTransform[i][0],restaurantsMatrixPcaTransform[i][1]), size=25)
ax.scatter(centroid[:,0],centroid[:,1], marker = "x", s=150, linewidths = 5, zorder =10)

plt.show()

#cluster 0 is green, cluster 1 is red, cluster 2 is cyan (blue)

#remember, that the order here is:
#y=0 (flacos), y=1 (Joes), y=2 (Poke), y=3 (Sush-shi), y=4 (Chick Fillet),
#y=5 (Mackie Des), y=6 (Michaels), y=7 (Amaze), y=8 (Kappa), y=9 (Mu)



#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
# I used "single" linkage,
# but you could try "complete", "average", "weighted", "centroid", "median", or "ward"

pca = PCA(n_components=2)
peopleMatrixPcaTransform = pca.fit_transform(peopleMatrix)

#Now lets try heirarchical clustering
linked = linkage(peopleMatrixPcaTransform, 'single')

#x=0 (Jane), x=1 (Bob), x=2 (Mary), x=3 (Mike), x=4 (Alice),
#x=5 (Skip), x=6 (Kira), x=7 (Moe), x=8 (Sara), x=9 (Tom)

labelList = ['Jane', 'Bob', 'Mary', 'Mike', 'Alice', 'Skip', 'Kira', 'Moe', 'Sara', 'Tom']

# explicit interface
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1)
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True, ax=ax)
ax.tick_params(axis='x', which='major', labelsize=25)
ax.tick_params(axis='y', which='major', labelsize=25)
plt.show()




#Now do the same for restaurants
pca = PCA(n_components=2)
restaurantsMatrixPcaTransform = pca.fit_transform(restaurantsMatrix)


#Now lets try heirarchical clustering
linked = linkage(restaurantsMatrixPcaTransform, 'single')


#y=0 (flacos), y=1 (Joes), y=2 (Poke), y=3 (Sush-shi), y=4 (Chick Fillet),
#y=5 (Mackie Des), y=6 (Michaels), y=7 (Amaze), y=8 (Kappa), y=9 (Mu)

labelList = ['Flacos', 'Joes', 'Poke', 'Sush-shi', 'Chick Fillet', 'Mackie Des', 'Michaels', 'Amaze', 'Kappa', 'Mu']

fig = plt.figure(figsize=(30, 15))
ax = fig.add_subplot(1, 1, 1)
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True, ax=ax)
ax.tick_params(axis='x', which='major', labelsize=25)
ax.tick_params(axis='y', which='major', labelsize=25)
plt.show()


#People Clustering metrics
#https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

print("The Calinski-Harabaz Index is used to measure better defined clusters.")
print("\nThe Calinski-Harabaz score is higher when clusters are dense and well separated.\n")

range_n_clusters = [2, 3, 4, 5, 6]
for n_clusters in range_n_clusters:
     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
     cluster_labels = clusterer.fit_predict(peopleMatrixPcaTransform)
     score = metrics.calinski_harabaz_score(peopleMatrixPcaTransform, cluster_labels)
     print("The Calinski-Harabaz score for :", n_clusters, " clusters is: ", score)



print("The Davies-Bouldin Index is used to measure better defined clusters.")
print("\nThe Davies-Bouldin score is lower when clusters more separated (e.g. better partitioned).\n")
print("Zero is the lowest possible Davies-Bouldin score.\n")

import warnings
warnings.filterwarnings("ignore")

range_n_clusters = [2, 3, 4, 5, 6]
for n_clusters in range_n_clusters:
     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
     cluster_labels = clusterer.fit_predict(peopleMatrixPcaTransform)
     score = metrics.davies_bouldin_score(peopleMatrixPcaTransform, cluster_labels)
     print("The Davies-Bouldin score for :", n_clusters, " clusters is: ", score)



#Silhouette Analysis with Kmeans Clustering on the PCA transformed People Matrix
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(peopleMatrixPcaTransform) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(peopleMatrixPcaTransform)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = metrics.silhouette_score(peopleMatrixPcaTransform, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = metrics.silhouette_samples(peopleMatrixPcaTransform, cluster_labels)

    # The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.
    # Scores around zero indicate overlapping clusters.
    # The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.

    print("\n\n\nFor n_clusters =", n_clusters,
          "\n\nThe average silhouette_score is :", silhouette_avg,
          "\n\n* The silhouette score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.",
          "\n* Scores around zero indicate overlapping clusters.",
          "\n* The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster",
          "\n\nThe individual silhouette scores were :", sample_silhouette_values,
          "\n\nAnd their assigned clusters were :", cluster_labels,
          "\n\nWhich correspond to : 'Jane', 'Bob', 'Mary', 'Mike', 'Alice', 'Skip', 'Kira', 'Moe', 'Sara', and 'Tom'")

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.rainbow(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.9)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.", fontsize=20)
    ax1.set_xlabel("The silhouette coefficient values", fontsize=20)
    ax1.set_ylabel("Cluster label", fontsize=20)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)


    # 2nd Plot showing the actual clusters formed
    colors = cm.rainbow(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(peopleMatrixPcaTransform[:, 0], peopleMatrixPcaTransform[:, 1], marker='.', s=300, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=400, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=400, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.", fontsize=20)
    ax2.set_xlabel("Feature space for the 1st feature", fontsize=20)
    ax2.set_ylabel("Feature space for the 2nd feature", fontsize=20)

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=25, fontweight='bold')

    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)

plt.show()


#Restaurant Clustering metrics

print("The Calinski-Harabaz Index is used to measure better defined clusters.")
print("\nThe Calinski-Harabaz score is higher when clusters are dense and well separated.\n")

range_n_clusters = [2, 3, 4, 5, 6]
for n_clusters in range_n_clusters:
     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
     cluster_labels = clusterer.fit_predict(restaurantsMatrixPcaTransform)
     score = metrics.calinski_harabaz_score(restaurantsMatrixPcaTransform, cluster_labels)
     print("The Calinski-Harabaz score for :", n_clusters, " clusters is: ", score)



print("The Davies-Bouldin Index is used to measure better defined clusters.")
print("\nThe Davies-Bouldin score is lower when clusters more separated (e.g. better partitioned.\n")
print("Zero is the lowest possible Davies-Bouldin score.\n")

import warnings
warnings.filterwarnings("ignore")

range_n_clusters = [2, 3, 4, 5, 6]
for n_clusters in range_n_clusters:
     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
     cluster_labels = clusterer.fit_predict(restaurantsMatrixPcaTransform)
     score = metrics.davies_bouldin_score(restaurantsMatrixPcaTransform, cluster_labels)
     print("The Davies-Bouldin score for :", n_clusters, " clusters is: ", score)




#Silhouette Analysis with Kmeans Clustering on the PCA transformed Restaurant Matrix
range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(restaurantsMatrixPcaTransform) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(restaurantsMatrixPcaTransform)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = metrics.silhouette_score(restaurantsMatrixPcaTransform, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = metrics.silhouette_samples(restaurantsMatrixPcaTransform, cluster_labels)

    # The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.
    # Scores around zero indicate overlapping clusters.
    # The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.

    print("\n\n\nFor n_clusters =", n_clusters,
          "\n\nThe average silhouette_score is :", silhouette_avg,
          "\n\n* The silhouette score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.",
          "\n* Scores around zero indicate overlapping clusters.",
          "\n* The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster",
          "\n\nThe individual silhouette scores were :", sample_silhouette_values,
          "\n\nAnd their assigned clusters were :", cluster_labels,
          "\n\nWhich correspond to : 'Flacos', 'Joes', 'Poke', 'Sush-shi', 'Chick Fillet', 'Mackie Des', 'Michaels', 'Amaze', 'Kappa', and 'Mu'")

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.jet(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.9)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.", fontsize=20)
    ax1.set_xlabel("The silhouette coefficient values", fontsize=20)
    ax1.set_ylabel("Cluster label", fontsize=20)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)


    # 2nd Plot showing the actual clusters formed
    colors = cm.jet(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(restaurantsMatrixPcaTransform[:, 0], restaurantsMatrixPcaTransform[:, 1], marker='.', s=300, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=400, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=400, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.", fontsize=20)
    ax2.set_xlabel("Feature space for the 1st feature", fontsize=20)
    ax2.set_ylabel("Feature space for the 2nd feature", fontsize=20)

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=25, fontweight='bold')

    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)

plt.show()



### Trying to answer if we can find weights from rankings

# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.
# Can you find their weight matrix?


results = np.matmul(restaurantsMatrix, newPeopleMatrix)

results



newPeopleMatrix.shape


# Say that rank 1 is best

#reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
# Argsort returns the indices that would sort an array - https://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
# By default, argsort is in ascending order, but below, we make it in descending order and then add 1 since ranks start at 1

np.set_printoptions(threshold=np.nan)


sortedResults = results.argsort()[::-1]
sortedResults.shape


np.sort(results)

#https://en.wikipedia.org/wiki/Invertible_matrix
#if we know restaurantsMatrix and Results Matrix can we calculate peopleMatrix?
#If the matrix A is invertible, then the equation Ax=b has a unique solution, namely x=A−1b.
# If A is not invertible, there may be either zero or many solutions to your problem.

results


#Ax = b
#b is results.
#x is people weights
#A is restaurantsMatrix
#if A was and n*n matrix
#then, x = A^-1*b

#A real inverse matrix example
#What does it look like when A is an n*n matrix?
a = np.array([[1,1,1],[0,2,5],[2,5,-1]])

print('Array a:')
print(a)
ainv = np.linalg.inv(a)

print('Inverse of a:')
print(ainv)

print('Matrix B is:')
b = np.array([[6],[-4],[27]])
print(b)

print('Compute A-1B:')
x = np.linalg.solve(a,b)
print(x)

print('Which is the same result as:\n     x = np.matmul(ainv,b)')
x = np.matmul(ainv,b)
print(x)
# this is the solution to linear equations x = 5, y = 3, z = -2

# however, you can't use np.linalg.inv(a) if a isn't an n*n matrix.
# recall that restuarantsMatrix is an 10 by 4 matrix
restaurantsMatrix.shape

b = results

#What happens?
ainv = np.linalg.inv(restaurantsMatrix)
x = np.matmul(ainv, b)
x.shape
x = np.swapaxes(x,0,1)
x.shape

#also... if A is n by n but is singular, then we can't get the inverse matrix
#example:
a = np.array([[2,3],[4,6]])
np.linalg.inv(a)

#singular matrix or degenerate matrix intuition:
#from: https://stackoverflow.com/questions/21638895/inverse-of-a-matrix-using-numpy

print("Imagine an old overhead film projector that shines a bright light through film onto a white wall. The pixels in the film are projected to the pixels on the wall.\n\nIf I stop the film projection on a single frame, you will see the pixels of the film on the wall and I ask you to regenerate the film based on what you see. That's easy, you say, just take the inverse of the matrix that performed the projection. An Inverse of a matrix is the reversal of the projection.\nNow imagine if the projector was corrupted, and I put a distorted lens in front of the film. Now multiple pixels are projected to the same spot on the wall. I asked you again to 'undo this operation with the matrix inverse'. You say: 'I can't because you destroyed information with the lens distortion, I can't get back to where we were, because the matrix is either Singular or Degenerate.'\n\nA matrix that can be used to transform some data into other data is invertable only if the process can be reversed with no loss of information. If your matrix can't be inverted, perhaps you are defining your projection using a guess-and-check methodology rather than using a process that guarantees a non-corrupting transform.")


# pinv returns the inverse of your matrix (A) when it is available and the pseduo inverse when A isn't
# an n by n matrix.
# http://mathworld.wolfram.com/Moore-PenroseMatrixInverse.html
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html
# https://www.quora.com/What-is-the-intuition-behind-pseudo-inverse-of-a-matrix

# The pseudo inverse of a matrix A, A^+ is the matrix that solves Ax=b
# if x is the solution, then A^+ is the matrix such that xbar = (A^+)(b)

b = results

ainv = np.linalg.pinv(restaurantsMatrix)
#x is an approximation of the peopleMatrix
x = np.matmul(ainv, b)
x.shape
x = np.swapaxes(x,0,1)
x.shape

#show how similar they are
peopleMatrix
x

#https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
#They are the same with a relative tolerance of 0.000000831529
# and an absolute tolerance of 0.000000831529
np.allclose(peopleMatrix, x, rtol=1e-14, atol=1e-14, equal_nan=False)


#The Moore-Penrose pseudoinverse is a matrix that can act as a partial replacement for the matrix inverse
#in cases where it does not exist. This matrix is frequently used to solve a system of linear equations when
#the system does not have a unique solution or has many solutions.

#For any matrix A, the pseudoinverse B exists, it is unique, and has the same dimensions as A' (the transpose matrix of A).

# If A is a 3 x 2 matrix:
A = np.array([[111, 222],[333, 444],[555, 666]])

A.shape

# A' is a 2 x 3 matrix (A transpose):
Atranspose = A.T
Atranspose.shape
# This means that the pseudoinverse B would have the same dimensions (2 x 3) as A transpose.


#If A is square and not singular (e.g. matrices are singular if and only if their determinant is 0),
#(also: Note that determinants are defined only for square matrices.)
#http://mathworld.wolfram.com/SingularMatrix.html
#http://mathworld.wolfram.com/Determinant.html
#Determinant Intuition: https://math.stackexchange.com/questions/668/whats-an-intuitive-way-to-think-about-the-determinant

#then pinv(A), A^+, is simply an expensive way to compute inv(A) (when A is square and not singular).
#However, if A is not square, or is square and singular, then inv(A) does not exist.

#The pinv(A) is computed through singular value decomposition
#https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Singular_value_decomposition_(SVD)
#https://en.wikipedia.org/wiki/Singular_value_decomposition

#the problem is going from the rank matrix to the results matrix.
#If you had the results matrix, finding the people weights matrix with the
#pinv of the Restaurant matrix would be trivial.

#However, we don't have the results matrix, we have a ranking of the results matrix.
#The ranking matrix causes a loss of information from the results matrix, which may not be recoverable.

#With the rankings matrix however, if you could find the clusters each person belongs to, you could still
#come to a conclusion about the place(s) to take the other team to lunch.



