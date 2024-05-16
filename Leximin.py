from itertools import chain
import gurobipy as grb
import numpy as np

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
                itemCountDic[item] =  itemCountDic[item] - 1
            for j in range(m):
                if i != j and P[j] !=0:
                    P[j] = P[j] + P[i] / reduced_m
            P[i] = 0
            #print("now sum = ", sum(P))

    item_probs = {}

    nonzero_prob = {}

    for i in range(m):
        nonzero_prob[panel_items[i]] = P[i]

    for i in instance:
        p = 0
        for item, k in nonzero_prob.items():
            if i in item:
                p = p + k

        item_probs[i] = p

    print("panel prob = ",nonzero_prob)
    print("item prob = ", item_probs)
    print("sum = ",sum(P), "min of item prob = ", min(item_probs.values()))

    return nonzero_prob, item_probs

# optimal leximin implemendated in Gurobi

def leximin(panel_items):

    m = grb.Model()
    # Variables for the output probabilities of the different panels
    lambda_p = [m.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.) for _ in panel_items]
    # To avoid numerical problems, we formally minimize the largest downward deviation from the fixed probabilities.
    x = m.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.)
    m.addConstr(grb.quicksum(lambda_p) == 1)  # Probabilities add up to 1
    for item in instance:
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
    for i in instance:
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


# Greedy leximin implemendation

def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    #if elements != universe:
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
    setProb = 1/l
    for sett in subsets:
        if sett in cover:
            probs[tuple(sett)] = setProb
        else:
            probs[tuple(sett)] = 0
    #print(cover)

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
    for i,v in probs.items():
        if v != 0:
            nonzeroprobs [i] = v
    print("number of non zero sets:")
    print(len(nonzeroprobs))
    print("non zero set probs:")
    print(nonzeroprobs)
    return item_probs, nonzeroprobs



instance = ["i1","i2","i3","i5"]

panel_items = [("i1","i2","i3"),("i1","i3","i5"), ("i1","i2","i5")]
#print(heuristic_leximin(panel_items))

print(leximin(panel_items))
# print(set_cover(instance,panel_items))



