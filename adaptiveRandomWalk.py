import timeit
import pickle
import numpy as np


# Function to calculate exact MMR value given a set of set

def exactMMR(s):
    s = sorted(s)
    s = tuple(s)
    MMR_score = 0
    for item in s:

        maxdiv = 0
        for elems in s:
            if item != elems:
                pairs = [item, elems]
                pairs.sort()
                tup = tuple(pairs)
                div = sorted_pairdiv[tup]
                if maxdiv < div:
                    maxdiv = div
        MMR_i = coef * sorted_rel[item] + (1 - coef) * (maxdiv)
        MMR_score = MMR_i + MMR_score

    return MMR_score



# Function to check if adaptiveRandomWalk and NRA result match

def checkExist(resultSet, randTopK):
    for r in resultSet:
        if sorted(r) == sorted(randTopK):
            return True
    return False




def checkItemCount(itemCount,d):
    count = 0
    for val in itemCount.values():
        if val >=d:
            count = count + 1
    return count/len(itemCount)



# Adaptive random walk algorithm implementation

def adaptiveRandomWalk(theta, sorted_rel, sorted_pairdiv, k, coeff):
    resultSet = []
    itemCount = {}
    topkItemCount = {}

    alpha = 3
    for i in sorted_rel.keys():
        itemCount[i] = 0
        topkItemCount[i] = 0

    prob = []

    maxDivVal = {}
    for i in sorted_rel.keys():
        maxDivVal[i] = 0

    divItems = list(sorted_pairdiv.items())
    for key,val in divItems:
        item1 = key[0]
        item2 = key[1]
        if maxDivVal[item1] < val:
            maxDivVal[item1] = val
        if maxDivVal[item2] < val:
            maxDivVal[item2] = val

    for key, val in sorted_rel.items():
        obj = coeff * val + (1 - coeff) * maxDivVal[key]
        prob.append(pow(obj, alpha))

    probsum = sum(prob)

    prob = [i / probsum for i in prob]

    maxmmr = -1
    count = 0
    maxIt = 50000
    while (min(itemCount.values()) < 1):
        randTopK = np.random.choice(list(sorted_rel.keys()), size=k, replace=False, p=prob)
        randMMR = exactMMR(randTopK)
        if randMMR > maxmmr:
            maxmmr = randMMR

        for item in randTopK:
            itemCount[item] = itemCount[item] + 1
            if randMMR > theta:
                if checkExist(resultSet, randTopK) == False:
                    topkItemCount[item] = topkItemCount[item] + 1

        if randMMR > theta:
            if checkExist(resultSet, randTopK) == False:
                resultSet.append(randTopK)
                newProb = []
                relKeys = list(sorted_rel.keys())
                for i in range(len(relKeys)):
                    item = relKeys[i]
                    newProb.append(prob[i] / (topkItemCount[item] + 1))
                sumNewProb = sum(newProb)
                prob = [i/sumNewProb for i in newProb]

        count = count + 1
        if count > maxIt:
            break
    print("max mmr ", maxmmr)
    return resultSet


# Function to calculate recall  

def calculateRecall(exactTopK, randomTopK):
    exactTopKSorted = []
    for etopk in exactTopK.values():
        exactTopKSorted.append(list(sorted(etopk)))

    randomTopKSorted = []
    for rtopk in randomTopK:
        randomTopKSorted.append(sorted(rtopk))

    # randomTopKSorted.extend(exactTopKSorted)
    match = 0
    for pair in exactTopKSorted:
        if list(pair) in randomTopKSorted:
            match = match + 1
    print("number of match = ", match)
    print("number of exact sets = ", len(exactTopK))
    recall = match / len(exactTopK)
    return recall


# Test run
# change dataset name to test different dataset
# change n, k, coef, delta to test different settings


datasetNameList = ['airbnb']
deltaList = [0.01]
for datasetName in datasetNameList:
    for delta in deltaList:
        n = 10000
        k = 5
        coef = 0.99

        highestMMRdict = {'yelp2': 31363.200569260214,'airbnb':368.6739215084142, 'imdb':439.7612455326149,'synthetic':4902.341114004959}
        highestMMR = highestMMRdict[datasetName]


        relFileName = r"data/sortedRel10k-" + datasetName + ".pickle"
        divFileName = r"data/sortedDiv10k-" + datasetName + ".pickle"
        exactTopkFileName =  r"data/" + datasetName + "-10k-topksets-delta0-"+str(delta)[2:]+".pickle"

        # exactTopk = [('i246', 'i987', 'i994', 'i996', 'i999'),  ('i246', 'i978', 'i987', 'i994', 'i999')]

        with open(relFileName, 'rb') as f:
            sorted_rel = pickle.load(f)
        f.close()

        with open(divFileName, 'rb') as f:
            sorted_pairdiv = pickle.load(f)
        f.close()

        print("-------------------------------------------------------------------------------------------------------------")
        print("dataset = ", datasetName)
        print("delta=", delta)
        print("highest MMR = ", highestMMR)
        print("lambda coefficient =", coef)


        theta = highestMMR - delta * highestMMR

        instance = list(sorted_rel.keys())
        start = timeit.default_timer()
        resultSetRandomWalk = adaptiveRandomWalk(theta, sorted_rel, sorted_pairdiv, k, coef)
        end = timeit.default_timer()
        fileName = "Adaptive_Random_top-k_" + datasetName + "_n=" + str(n) + "_k=" + str(k) + "_delta=" + str(delta) + ".pickle"
        print("number of topk set random walk = ", len(resultSetRandomWalk))
        print("topk sets = ", resultSetRandomWalk)
        with open(fileName, 'wb') as handle:
            pickle.dump(resultSetRandomWalk, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("file saved in : ", fileName)

        print("run time = ", end - start)

        fileName = "Random_top-k_"+datasetName + "_n=" + str(n)+"_k="+str(k)+"_delta="+str(delta)+".pickle"
        with open(fileName, 'rb') as f:
            resultSetRandomWalk = pickle.load(f)
