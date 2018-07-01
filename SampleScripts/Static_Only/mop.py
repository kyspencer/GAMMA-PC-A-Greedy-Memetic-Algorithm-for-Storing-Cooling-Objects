# mop.py
#   This file contains functions inherent to multi-objective problems (MOP).
#   Author: Kristina Spencer
#   Date: March 31, 2016

import numpy as np
import random


def main():
    print('This python script includes: ')
    print('  - MOproblem: a class that maintains multi-objective problem.')
    print('  - calcfits(): a module to calculate fitness values.')
    print('  - dom(): a module to determine dominance between two solutions,')
    print('  - fnds(): a module to perform the fast-nondominated-sort algorithm,')


class MOproblem:
    # This class maintains the overall multi-objective problem.
    def __init__(self, pop, items):
        self.pop = pop
        self.items = items
        self.nobj = 3
        self.weights = np.zeros((len(items), 1))
        self.heights = np.zeros((len(items), 1))
        self.makematrix()

    def makematrix(self):
        # This function initializes the weight and height matrices.
        for j in range(len(self.items)):
            self.weights[j, 0] = self.items[j].getweight()
            self.heights[j, 0] = self.items[j].getheight()

    def calcfits(self, x, y):
        # This module calculates the fitness values for a solution.
        # To run this module, enable the following line:
        # import numpy as np
        fitvals = np.zeros(self.nobj)
        # Objective Function 1 min. # of bins
        fitvals[0] = np.sum(y)
        # Objective Function 2 min. H of bins
        his = np.dot(x, self.heights)
        fitvals[1] = np.amax(his)
        # Objective Function 3 min. Average Bin weight
        wis = np.dot(x, self.weights)
        fitvals[2] = np.average(wis, weights=wis.astype(bool))
        bin_heights = np.reshape(his, len(self.items))
        bin_weights = np.reshape(wis, len(self.items))
        return fitvals, bin_heights, bin_weights

    def getweightmatrix(self):
        return self.weights

    def getheightmatrix(self):
        return self.heights


def calcfits(x, y, items):
    # This module calculates the fitness values for a solution
    # independent of a problem.
    fitvals = np.zeros(3)
    # Objective Function 1 min. # of bins
    fitvals[0] = np.sum(y)
    # Objective Function 2 min. H of bins
    n = len(y)
    his = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if x[i, j] == 1:
                his[i] = his[i] + items[j].getheight()
    fitvals[1] = np.amax(his)
    # Objective Function 3
    r = 0.03
    bi = int(fitvals[0])
    wis = np.zeros(bi)
    for i in range(bi):
        bnm = 0
        for j in range(n):
            cj = items[j].getweight()
            bnm = bnm + cj * x[i, j]
        wis[i] = bnm # / ((1 + r)**i)
    # fitvals[2] = np.amax(wis)
    fitvals[2] = np.average(wis, weights=wis.astype(bool))
    return fitvals


def dom(u, v):
    # dom(u, v) determines if fitness value u dominates fitness value v
    # to find the Pareto set
    nobj = 3
    equal = 0
    for i in range(nobj):
        if u[i] == v[i]:
            equal += 1
    u1, u2, u3 = u[0], u[1], u[2]
    v1, v2, v3 = v[0], v[1], v[2]
    return equal != nobj and u1 <= v1 and u2 <= v2 and u3 <= v3


def fnds(setp):
    # This module performs the fast-non-dominated-sort described in Deb(2002).
    numsol = len(setp)
    fronts = []
    sp = []
    fhold = []
    nps = []
    for p in range(numsol):
        shold = []
        np = 0
        for q in range(numsol):
            if setp[p] != setp[q]:
                if dom(setp[p].getfits(), setp[q].getfits()):
                    shold.append(setp[q])
                if dom(setp[q].getfits(), setp[p].getfits()):
                    np += 1
        sp.append(shold)
        nps.append(np)
        if np == 0:
            fhold.append(setp[p])
            setp[p].updaterank(1)
    fronts.append(fhold)            # Pareto set
    i = 0
    while fronts[i] != []:
        q = []
        for j in range(numsol):
            if setp[j] in fronts[i]:
                for k in range(numsol):
                    if setp[k] in sp[j]:
                        nps[k] -= 1
                        if nps[k] == 0:
                            setp[k].updaterank(i+2)
                            q.append(setp[k])
        fronts.append(q)
        i += 1
    return setp, fronts


if __name__ == '__main__':
    main()
