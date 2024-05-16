import numpy as np
import operator
from itertools import combinations
import gurobipy as grb
import timeit
from itertools import chain
from sklearn.datasets import make_blobs
from scipy.spatial import distance
from math import comb
import pandas as pd
from sklearn.preprocessing import normalize
import heapq as hq
import pickle
from sklearn.preprocessing import MinMaxScaler




sorted_rel = {}

'''
instance = ["i1","i2","i3","i4","i5"]
sorted_rel["i1"] = 8.6
sorted_rel["i2"] = 8.5
sorted_rel["i3"] = 8.3
sorted_rel["i4"] = 8.1
sorted_rel["i5"] = 7.9


'''

#Q = (100)
#Q = (1798700)
#Q = (-33, -42)
#Q = (5,)
#Q = (2022)
Q = (5000)
print("Query: ", Q)

numberofSample = 10000
print("dataset size:", numberofSample)

# IMDB
'''
dataset = pd.read_csv(r'ImdbTitleRatings.csv', nrows=numberofSample)
#dataset = pd.read_csv(r'movies.csv', nrows=numberofSample)
#pd.to_numeric(dataset['Year'])
D = dataset.iloc[:, 2].values
#instance = list([tuple(e) for e in D])
instance = list([e for e in D])
print("IMDB dataset")
#print(instance)
'''



# makeblobs
'''
dataset = pd.read_csv(r'MakeBlobs100k.csv', nrows=numberofSample)
#dataset = pd.read_csv(r'2M2FMakeBlobs.csv', nrows=numberofSample)
#dataset.sample(n = numberofSample)
D = dataset.iloc[:, :].values
instance = list([tuple(e) for e in D])
print("makeblobs dataset")
#print(instance)
'''



#print(instance)
# movielens
'''
dataset=pd.read_csv(r'ratings.csv', nrows=numberofSample)
#dataset.sample(n = numberofSample)
#D = dataset.iloc[:, [2, 3]].values
D = dataset.iloc[:, 1].values
#normalized_D = normalize(D, axis=0, norm='l2')
#X = normalized_D
#instance = list([tuple(e) for e in X])
#instance = list([tuple(e) for e in D])
instance = list(e for e in D)
print("movielens dataset")
#print(instance)
'''


# yelp

dataset=pd.read_csv(r'business.csv',  nrows=numberofSample)
#dataset.sample(n = numberofSample)
#D = dataset.iloc[:, [6,7,8]].values
#D = dataset.iloc[:, [6,7]].values
D = dataset.iloc[:, [6,7,8]].values
#instance = list([tuple(e) for e in D])
#normalized_D = normalize(D, axis=0, norm='l2')
#X = normalized_D*100
instance = list([tuple(e) for e in D])
#instance = list(e for e in D)
print("yelp dataset")
#print(instance)


# airbnb
'''
dataset=pd.read_csv(r'listings1.csv',  nrows=numberofSample)
#D = dataset.iloc[:, 54].values
D = dataset.iloc[:,40].values
#instance = list([tuple(e) for e in D])
instance = list(e for e in D)
print("airbnb dataset")
print(instance)
'''


'''
s = 100000

X, Y = make_blobs(n_samples=s+1, centers=100, cluster_std=50, random_state=0)
instance = list([tuple(e) for e in X])
#print("Instance Dataset: ", instance)
dfdata = pd.DataFrame(instance)
dfdata.to_csv("MakeBlobs100k.csv")


'''




indexMap ={}
ind = 0

print(len(instance))

for e in instance:
    index = "i" + str(ind)
    #indexMap[tuple(e)] = index
    indexMap[(e,ind)] = index
    ind = ind + 1



rel = {}

for i in range(0, len(instance)):
    #sim = 1 / (1 + distance.euclidean(Q, instance[i]))
    #sim = round(sim, 2)
    #rel[indexMap[instance[i]]] = sim
    sim = instance[i][2]
    rel[indexMap[(instance[i],i)]] = sim

# print("Relevance to the query: ", rel)

sorted_rel = dict(sorted(rel.items(), key=operator.itemgetter(1), reverse=True))


maxnorm = max(sorted_rel.values())
minnorm = min(sorted_rel.values())

lower = 1
upper = 100


for k, v in sorted_rel.items():
    v = lower + (v - minnorm) * (upper - lower) / ( maxnorm- minnorm)
    sorted_rel[k] = v



print('Normalized sorted relevance list')
print(sorted_rel.values())

max_rel = list(sorted_rel.values())[0]
min_rel = list(sorted_rel.values())[len(sorted_rel) - 1]

print("Maximum Relevance score: ", max_rel)
print("Minimum Relevance score: ", min_rel)


pairwise_diversity = {}


for i in range(0, len(instance)):
    for j in range(i + 1, len(instance)):
        if i != j:
            #dist = distance.euclidean(instance[i], instance[j])
            #dist = round(dist, 2)
            dist = distance.euclidean(instance[i][:1], instance[j][:1])
            pairs = [indexMap[(instance[i],i)], indexMap[(instance[j],j)]]
            pairs.sort()
            tup = tuple(pairs)
            pairwise_diversity[tup] = dist


'''

pairwise_diversity[("i2","i3")] = 5
pairwise_diversity[("i3","i5")] = 5
pairwise_diversity[("i1","i3")] = 4
pairwise_diversity[("i3","i4")] = 4
pairwise_diversity[("i1","i4")] = 2
pairwise_diversity[("i4","i5")] = 2
pairwise_diversity[("i1","i2")] = 2
pairwise_diversity[("i2","i4")] = 2
pairwise_diversity[("i2","i5")] = 1
pairwise_diversity[("i1","i5")] = 1

'''

#sorted_pairdiv = pairwise_diversity

sorted_pairdiv = dict(sorted(pairwise_diversity.items(), key=operator.itemgetter(1), reverse=True))

#print("Sorted diversity:", sorted_pairdiv.values())

maxdivnorm = max(sorted_pairdiv.values())
mindivnorm = min(sorted_pairdiv.values())


lower2 = 1
upper2 = 5


for k, v in sorted_pairdiv.items():
    v = lower2 + (v - mindivnorm) * (upper2- lower2) / ( maxdivnorm- mindivnorm)
    sorted_pairdiv[k] = v



print("diversity list sorted")
print("Sorted normalized diversity:", sorted_pairdiv.values())


max_div = list(sorted_pairdiv.values())[0]
min_div = list(sorted_pairdiv.values())[len(sorted_pairdiv) - 1]


print("Maximum Diversity score: ", max_div)
print("Minimum Diversity score: ", min_div)



file_to_write = open("sortedRel1k-yelp.pickle", "wb")

pickle.dump(sorted_rel, file_to_write)

file_to_write2 = open("sortedDiv1k-yelp.pickle", "wb")

pickle.dump(sorted_pairdiv, file_to_write2)

