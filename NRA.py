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
import math


K = 5
print("k = ", K)

coef = 0.99
print("lambda coefficient =", coef)

sorted_rel = {}

'''
instance = ["i1","i2","i3","i4","i5"]
sorted_rel["i1"] = 8.6
sorted_rel["i2"] = 8.5
sorted_rel["i3"] = 8.3
sorted_rel["i4"] = 8.1
sorted_rel["i5"] = 7.9


pairwise_diversity = {}

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

sorted_pairdiv = pairwise_diversity

'''
numberofSample = 10000
print("dataset size:", numberofSample)


total_sets = comb(numberofSample, K)





with open('sortedRel1k-yelp.pickle', 'rb') as f:
    sorted_rel = pickle.load(f)
f.close()

with open("sortedDiv1k-yelp.pickle", 'rb') as f:
    sorted_pairdiv  = pickle.load(f)
f.close()


#print(sorted_pairdiv)
print(len(sorted_rel))


#print("sorted rel list", sorted_rel.values())
#print("sorted diversity list", sorted_pairdiv.values())

max_div = list(sorted_pairdiv.values())[0]
min_div = list(sorted_pairdiv.values())[len(sorted_pairdiv) - 1]

max_rel = list(sorted_rel.values())[0]
min_rel = list(sorted_rel.values())[len(sorted_rel) - 1]

#####leximin

def leximin(panel_items):
    print("number of panel items:", len(panel_items))
    if len(panel_items) == 12:
        print("deb")
    m = grb.Model()
    # Variables for the output probabilities of the different panels
    lambda_p = [m.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.) for _ in panel_items]
    # To avoid numerical problems, we formally minimize the largest downward deviation from the fixed probabilities.
    x = m.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.)
    m.addConstr(grb.quicksum(lambda_p) == 1)  # Probabilities add up to 1
    itemSet = set(list(chain.from_iterable([list(itm) for itm in panel_items])))
    for item in itemSet:
        item_probability = grb.quicksum(comm_var for committee, comm_var in zip(panel_items, lambda_p)
                                        if item in committee)
        m.addConstr(item_probability >= x)

    m.setObjective(x, grb.GRB.MAXIMIZE)
    m.optimize()

    probabilities = np.array([comm_var.x for comm_var in lambda_p]).clip(0, 1)
    probabilities = list(probabilities / sum(probabilities))

    finalsetprobs = {}

    for i in range(len(panel_items)):
        for p in range(len(probabilities)):
            if i == p:
                finalsetprobs[panel_items[i]] = probabilities[p]

    print("final panels probabilities: ", finalsetprobs)

    nonzero_prob = {}
    for k, v in finalsetprobs.items():
        if v != 0:
            nonzero_prob[k] = v

    print("non zero panels probabilities:", nonzero_prob)

    print("Size of non zero probability list:", len(nonzero_prob))

    prob = 0
    for i, j in nonzero_prob.items():
        prob = prob + j

    print("total prob:", prob)

    item_probs = {}
    for i in itemSet:
        p = 0
        for item, k in nonzero_prob.items():
            if i in item:
                p = p + k

        item_probs[i] = p

    print("item probs:", item_probs)
    print("Minimum probability of items: ", min(list(item_probs.values())))

    for v in m.getVars():
        print(v.varName, v.x)

    return nonzero_prob, item_probs


#######################
# heuristic leximin
def heuristic_leximin(panel_items):
    instance = set(list(chain.from_iterable([list(itm) for itm in panel_items])))
    itemCountDic = {}
    P = []
    m = len(panel_items)

    for item in instance:
        itemCountDic[item] = 0

    i = 0
    for i in range(m):
        for item in panel_items[i]:
            itemCountDic[item] = itemCountDic[item] + 1
        P.append(1 / m)

    reduced_m = m
    for i in range(m):
        prbZero = True
        for item in panel_items[i]:
            if itemCountDic[item] < 2:
                prbZero = False
                break

        if prbZero:
            reduced_m = reduced_m - 1
            for item in panel_items[i]:
                itemCountDic[item] = itemCountDic[item] - 1
            for j in range(m):
                if i != j and P[j] != 0:
                    P[j] = P[j] + P[i] / reduced_m
            P[i] = 0
            # print("now sum = ", sum(P))

    item_probs = {}

    nonzero_prob = {}

    for i in range(m):
        if P[i] != 0:
            nonzero_prob[panel_items[i]] = P[i]

    for i in instance:
        p = 0
        for item, k in nonzero_prob.items():
            if i in item:
                p = p + k

        item_probs[i] = p

    print("panel prob = ", nonzero_prob)
    print("item prob = ", item_probs)
    print("sum = ", sum(P), "min of item prob = ", min(item_probs.values()))

    return nonzero_prob, item_probs



###########greedy leximin

def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    # if elements != universe:
    #    return None
    covered = set()
    cover = []
    probs = {}
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(set(s) - covered))
        cover.append(subset)
        covered |= set(subset)
    l = len(cover)
    setProb = 1 / l
    for sett in subsets:
        if sett in cover:
            probs[tuple(sett)] = setProb
        else:
            probs[tuple(sett)] = 0
    # print(cover)

    item_probs = {}
    for i in universe:
        p = 0
        for item, val in probs.items():
            if i in list(item):
                p = p + val

        item_probs[i] = p
    print("item probs:")
    print(item_probs)
    nonzeroprobs = {}
    for i, v in probs.items():
        if v != 0:
            nonzeroprobs[i] = v
    print("number of non zero sets:")
    print(len(nonzeroprobs))
    print("non zero set probs:")
    print(nonzeroprobs)
    return nonzeroprobs, item_probs


columns = {}
panel = []
topkSets = {}
seen_relist = []
seen_divpairlist = []

minrel = 0



# print("Least relevant item in the top-k set:", list(sorted_rel.keys())[list(sorted_rel.values()).index(minrel)])


def getNextRel(position):
    return (list(sorted_rel.keys())[position], list(sorted_rel.values())[position])


# print(getNextRel(0))

def getNextDiv(position):
    return (list(sorted_pairdiv.keys())[position], list(sorted_pairdiv.values())[position])


##############################
def exactMMR(s):
    s = sorted(s)
    s = tuple(s)
    MMR_score = 0
    for item in s:

        maxdiv =  0
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


############################




topkSets = {}
topkSets[0] = 1000000
init_j = 1
i = 2
count = 0

threshold = []
Lbounds = {}
Ubounds = {}

seen_items = []
seen_rellist = []
seen_divpairlist = []


flag = False


########################### NRA algorithm

def generateTopkSet(i, lastMax):
    global init_j
    global seen_items
    global seen_rellist
    global seen_divpairlist
    global count
    global K
    global totalComb
    global Ubounds
    global Lbounds
    global highestMMR
    global delta
    global theta

    global flag
    flag = False

    for j in range(init_j, len(sorted_rel) + 1):
        print("j = ", j)
        if j == 6:
            print("debug")

        nextRel_score = getNextRel(j - 1)[1]
        nextDiv_score = getNextDiv(j - 1)[1]

        nextRel_item = getNextRel(j - 1)[0]
        nextDiv_items = getNextDiv(j - 1)[0]

        seen_items.append(nextRel_item)
        seen_rellist.append(nextRel_item)

        seen_divpairlist.append(nextDiv_items)

        # items = []
        thisiteritems = []

        thisiteritems.append(nextRel_item)

        for elem in nextDiv_items:
            seen_items.append(elem)
            thisiteritems.append(elem)

        seen_items = list(set(seen_items))

        thisiteritems = set(thisiteritems)
        thisiteritems = list(thisiteritems)

        if len(seen_items) < K:
            continue
        else:

            ###################### threshold and upperbound calculation

            start = timeit.default_timer()

            candidate_sets = []
            for set1 in combinations(seen_items, K):
                set1 = sorted(set1)
                #testSet = {(-3.758262162, 1.551005173), (-6.88454387, 0.442749097), (9.339333815, 11.69874141), (0.82273948, -1.248441866), (-11.74065149, 17.83245323)}
                #if set(set1) == testSet:
                    #print("debug")
                set1 = tuple(set1)
                if set1 in topkSets.values():
                    continue

                candidate_sets.append(set1)

                # print(i)

                MMR_score = 0
                for item in set1:

                    maxdiv = 0
                    for elem in set1:
                        if item != elem:

                            if (item, elem) in seen_divpairlist:
                                div = sorted_pairdiv[(item, elem)]
                                if maxdiv < div:
                                    maxdiv = div
                            elif (elem, item) in seen_divpairlist:
                                div = sorted_pairdiv[(elem, item)]
                                if maxdiv < div:
                                    maxdiv = div
                            else:
                                div = nextDiv_score
                                if maxdiv < div:
                                    maxdiv = div
                                    # get rel and div from dictionary

                    if item in seen_rellist:
                        MMR_i = coef * sorted_rel[item] + (1 - coef) * (maxdiv)
                    else:
                        MMR_i = coef * (nextRel_score) + (1 - coef) * (maxdiv)

                    MMR_score = MMR_i + MMR_score

                #MMR_score = round(MMR_score, 2)
                MMR_score = math.floor(MMR_score)

                Ubounds[set1] = MMR_score

            all_values = Ubounds.values()
            max_MMR = max(all_values)

            MMR_threshold = max_MMR
            threshold.append(MMR_threshold)
            if i > 1 and MMR_threshold < theta:
                print("Breaking threshold", MMR_threshold)
                if len(candidate_sets) != 0:
                    flag = True

                break



            print("Threshold = ", MMR_threshold)

            stop = timeit.default_timer()
            th_time = stop - start

            # Ubounds = dict(di)

            # print('Time for threshold: ', th_time)

            # if MMR_threshold < dif:
            #    print("break")
            #    break

            ##########################

            ###################### lower bound

            start = timeit.default_timer()

            totalComb = 0

            for sets in combinations(seen_items, K):
                # print(i)
                totalComb = totalComb + 1

                sets = sorted(sets)
                sets = tuple(sets)
                MMR_score = 0
                if sets in topkSets.values():
                    continue
                for item in sets:

                    maxdiv = 0
                    for elem in sets:
                        if item != elem:

                            if (item, elem) in seen_divpairlist:
                                div = sorted_pairdiv[(item, elem)]
                                if maxdiv < div:
                                    maxdiv = div
                            elif (elem, item) in seen_divpairlist:
                                div = sorted_pairdiv[(elem, item)]
                                if maxdiv < div:
                                    maxdiv = div
                            else:
                                div = min_div
                                if maxdiv < div:
                                    maxdiv = div
                                    # get rel and div from dictionary

                    if item in seen_rellist:
                        MMR_i = coef * sorted_rel[item] + (1 - coef) * (maxdiv)
                    else:
                        MMR_i = coef * (min_rel) + (1 - coef) * (maxdiv)

                    MMR_score = MMR_i + MMR_score

                #MMR_score = round(MMR_score, 2)
                MMR_score = math.ceil(MMR_score)
                Lbounds[sets] = MMR_score
            # print("total combinations = ",totalComb)

            stop = timeit.default_timer()
            lb_time = stop - start
            # print('Time for lowebound: ', lb_time)

            ############### pruning and stopping condition
            start = timeit.default_timer()

            if len(Ubounds) > 1:
                maxlb = max(Lbounds.values())
                settt = [k for k, v in Lbounds.items() if v == maxlb][0]
                maxub = 0
                ubsettt = Ubounds[settt]
                Ubounds.pop(settt)

                maxub = max(Ubounds.values())
                Ubounds[settt] = ubsettt
                # Ubounds
                print("max lower bound:", maxlb)
                if maxlb >= maxub:
                    init_j = j + 1
                    print(init_j)
                    return settt  # we found the next best set

            stop = timeit.default_timer()
            prsp_time = stop - start
            # print('Time for prune and stop: ', prsp_time)

            # print("size of candidate_sets = ", len(candidate_sets))
            # print("size of Lbounds = ", len(Lbounds.keys()))
            ############################ pruning condition
            start = timeit.default_timer()

            if len(Ubounds) > 1:
                newCandSet = []
                Ubounds = list(Ubounds.items())
                hq.heapify(Ubounds)
                for le in range(len(Ubounds)):
                    #minub = hq.heappop(Ubounds)
                    minub = Ubounds[0]
                    #print(minub)
                    if minub[1] >= maxlb:
                        #hq.heappush(Ubounds, minub)
                        break

                Ubounds = dict(Ubounds)
                candidate_sets = Ubounds.keys()

                '''
                for setlb in candidate_sets:
                    #heap
                    if Ubounds[setlb] >= maxlb:
                        newCandSet.append(setlb)
                    else:
                        count = count + 1
                candidate_sets = newCandSet
                '''

                # for setlb in Lbounds.keys():
                #
                #     if Ubounds[setlb] < maxlb:
                #         count = count +1
                #         candidate_sets.remove(setlb)  # only for this iteration this set gets pruned

            stop = timeit.default_timer()
            pr_time = stop - start
            #print('Time for prune: ', pr_time)

            # stopping condition
            start = timeit.default_timer()

            if i > 1:
                if max(Lbounds.values()) >= min(threshold[i], lastMax):

                    init_j = j + 1

                    mmr = 0
                    for eachset in candidate_sets:
                        mmrset = exactMMR(eachset)
                        if mmr < mmrset:
                            mmr = mmrset
                            best = eachset
                    return best

            stop = timeit.default_timer()
            sp_time = stop - start
            #print('Time for stop: ', sp_time)

    if j == len(sorted_rel) or flag == True:
        flag = True
        mmr = 0
        for eachset in candidate_sets:
            mmrset = exactMMR(eachset)
            if mmr < mmrset:
                mmr = mmrset
                best = eachset
        return best


###########################################
#### main

#set = ['i5202', 'i9926', 'i9975', 'i9954', 'i9944']
#print("exact mmr of one set = ", exactMMR(set))


start1 = timeit.default_timer()

topkSets[1] = generateTopkSet(1, topkSets[0])

start = timeit.default_timer()

highestMMR = exactMMR(topkSets[1])

print("exact mmr:", highestMMR)

delta = 0.05* highestMMR

#delta = 289.18

#delta = 0
print("delta=", delta)

theta = highestMMR - delta
print("theta:", theta)

stop = timeit.default_timer()
exact_time = stop - start
print('Time for exact MMR: ', exact_time)

print("init_j", init_j)
print("Set pruning count for 1 top-k set:", int(count))
print("total sets combination for first top-k:", totalComb)

i = 2
topkSets.pop(0)
countGenTopk = 0
setcount = 0
while exactMMR(topkSets[i - 1]) > theta and init_j < numberofSample:
    lastMax = exactMMR(topkSets[i - 1])
    print("last max",lastMax)
    setcount = setcount +1
    print("number of final top-k sets so far:", setcount)
    if flag == True:
        break
    Ubounds.pop(topkSets[i - 1])
    Lbounds.pop(topkSets[i - 1])

    topkSets[i] = generateTopkSet(i, lastMax)

    #print("mmr:", exactMMR(topkSets[i]))

    i = i + 1
    countGenTopk = countGenTopk + 1

start = timeit.default_timer()

finalres = {}
for k,v in topkSets.items():
    if v!= -1:
        finalres[k] = v

print("final top-k sets", finalres)



'''
print("number of topk set = ", len(finalres))
fileName = "imdb10k-topksets-delta0-001.pickle"
with open(fileName, 'wb') as handle:
    pickle.dump(finalres, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("file saved in : ", fileName)
'''


file_to_write2 = open("airbnb-1k-topksets-delta0-02.pickle", "wb")

pickle.dump(finalres, file_to_write2)



# set_prob, item_prob = leximin(list(topkSets.values()))
# set_prob, item_prob = set_cover(instance,list(topkSets.values()))
# set_prob, item_prob = heuristic_leximin(list(topkSets.values()))

stop = timeit.default_timer()
lex_time = stop - start
# print('Time for leximin: ', lex_time)
# print(set_prob, item_prob)
print("init_j", init_j)

myset = set(seen_items)
uniqueSeenItems = list(myset)

lenuniqueRecords = len(uniqueSeenItems)
print("number of seen records:", lenuniqueRecords)

print("number of final topk sets", len(finalres))

pruneRecord = (numberofSample - lenuniqueRecords) / numberofSample

print("record pruning percentage:", pruneRecord * 100)

print("num times genTopk clled = ", countGenTopk)
print(" set pruning condition count:", int(count))
print("total sets:", total_sets)

print("total sets combination so far top-k:", totalComb)

setPrune = (total_sets - totalComb) / total_sets

print("prune percentage for sets:", setPrune * 100)

stop1 = timeit.default_timer()

run_time = stop1 - start1
print('running time: ', run_time)