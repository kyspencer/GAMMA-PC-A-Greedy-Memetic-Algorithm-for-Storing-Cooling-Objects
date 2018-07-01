# binpacking.py
#   This file contains a heuristic to translate a given combination of items
#   into a bin packing mapping. Function ed performs the necessary encoding
#   and decoding based on Dahmani (2014).
#
#   The algorithm from Dahmani has been modified here for the cooling cookie
#   problem.
#   Author: Kristina Spencer

from __future__ import print_function
import numpy as np
import random
import sys
from math import ceil
from operator import attrgetter


class BPP:
    # This class groups the bin packing problem information.
    def __init__(self, n, cap, coolrack, items):
        self.n = int(n)         # Number of items to sort
        self.cap = cap          # Max. bin capacity
        self.rack = coolrack    # Max. cooling rack capacity
        self.items = items      # list of item objects
        self.tbatch = 600       # Each batch of cookies takes 600 seconds to bake
        self.nbatches = max(items.values(), key=attrgetter('batch')).batch
        self.lb = 0             # initialize lower bound
        self.calclowerbound()

    def calclowerbound(self):
        # This function calculates theoretical lower bound for the number of
        # bins. It assumes this is the total weight of all items divided by
        # the max weight of a bin.
        minbins = ceil(float(self.n) / self.cap)
        self.lb = int(minbins)

    def getub(self):
        # Returns the upper bound (bin capacity)
        return self.cap

    def getitems(self):
        # Returns the list of items to pack
        return self.items

    def getlb(self):
        # Returns the theoretical lower bound
        return self.lb


def ed(permutation, tfill, bpp):
    # This function follows the algorithm given in Dahmani (2014).
    #  - permutation = list of item indices
    #  - tfill = n x 1 matrix, times each bin gets filled
    #  - bpp = instance of class BPP
    #   The heuristic used is randomly chosen as indicated in the paper.
    lambba = random.randint(0, 2)
    if lambba == 0:
        x, y, tfill = ll(permutation, tfill, bpp)
    elif lambba == 1:
        x, y, tfill = dp(permutation, tfill, bpp)
    else:
        x, y, tfill = combo(permutation, tfill, bpp)
    return x, y, tfill


def ll(permutation, tfill, bpp):
    # This module decodes a given chromosome "permutation" using
    # the least loaded strategy. It returns x and y, a loading pattern
    # combination. The item index corresponds to the column j + 1.
    m = int(bpp.getlb())    # initialize lower bound on bins
    items = bpp.getitems()
    n, c, x, y, r = initial(m, bpp)
    # Go through permutation one item at a time.
    for j in range(n):
        cj = permutation[j]
        item = items.get(cj)
        # Find box i to put j in:
        i = llmove(n, m, r, item, tfill)
        # Add cookie j to box i
        m, x, y, r, tfill = addtobin(i, cj, m, x, y, r, c, item, tfill)
    return x, y, tfill


def llmove(n, m, r, item, tfill):
    i_rlowtohigh = np.argsort(r)
    # This module performs the sorting for module ll.
    for j in range(m):
        # Find open bin with max. residual value, moving backward thru i_rlowtohigh
        lli = i_rlowtohigh[(n - 1) - j]
        pack = packable(r[lli], tfill[lli], item)
        if all(pack) is True:
            return lli
    # If least loaded bin won't fit item, need to open new bin.
    if all(pack) is False:
        return m


def dp(permutation, tfill, bpp):
    # This module decodes a given chromosome "permutation" using
    # the dot product strategy. It returns x and y, a loading pattern
    # combination. The item index corresponds to the column j + 1.
    m = 1                       # open one bin
    items = bpp.getitems()
    n, c, x, y, r = initial(m, bpp)
    t_range, onrack = get_coolrack_variation(tfill, bpp)
    # Set weights for the dot product array
    weights = [1.0 / c, 1.0 / bpp.rack]     # 1/boxcap, 1/coolrackcap
    # Go through permutation one item at a time.
    for j in range(n):
        cj = permutation[j]
        item = items.get(cj)
        # Find box i to put j in:
        i = dpmove(m, r, item, tfill, weights, t_range, onrack)
        # Add cookie j to box i
        m, x, y, r, tfill = addtobin(i, cj, m, x, y, r, c, item, tfill)
        t_range, onrack = update_coolrack_variation(i, tfill, t_range, onrack)
    return x, y, tfill


def dpmove(m, r, item, tfill, weights, t_range, onrack):
    # This module performs the sorting for module dp.
    # Form the dot product array
    dparray = np.zeros(m)
    for i in range(m):
        pack = packable(r[i], tfill[i], item)
        if all(pack) is True:
            tk = np.where(np.array(t_range) == tfill[i])[0]
            # Filling early will reduce onrack for all after time[tk]
            maxonrack_fromtk = max(onrack[tk[0]:])
            dparray[i] = weights[0] * r[i] + weights[1] * maxonrack_fromtk
    # Find the max. dot product value
    maxdp = np.amax(dparray)
    if maxdp == 0:
        i = m
    else:
        i = np.argmax(dparray)
    return i


def get_coolrack_variation(tfill, bpp):
    # This module initializes the onrack values as a function of time.
    # Set up t_range: combine tfill values and bake times
    t_range = [(b + 1) * bpp.tbatch for b in range(bpp.nbatches + 1)]
    tfill_unique = list(np.unique(tfill))
    t_range.extend(tfill_unique)
    t_range = list(set(t_range))
    t_range.sort()
    if 0 in t_range:
        t_range.remove(0)
    # Set up space values
    n_b = bpp.n / bpp.nbatches
    onrack = [n_b]
    for tk in range(1, len(t_range)):
        if t_range[tk] % bpp.tbatch == 0:
            if onrack[-1] == bpp.n:
                onrack.append(onrack[-1])
            else:
                onrack.append(onrack[-1] + n_b)
        else:
            onrack.append(onrack[-1])
    return t_range, onrack


def update_coolrack_variation(i, tfill, t_range, onrack):
    # This module updates the onrack values as a function of time, assuming
    # removal of one cookie.
    # Determine if box i is newly opened or already established:
    if tfill[i] in t_range:
        tk = np.where(np.array(t_range) == tfill[i])[0]
        for t in range(tk[0], len(t_range)):
            onrack[t] -= 1
    else:
        # Add tfill[i] to t_range and update onrack
        if tfill[i] > t_range[-1]:
            t_range.append(tfill[i])
            onrack.append(onrack[-1] - 1)
        else:
            tklist = np.where(np.array(t_range) >= tfill[i])[0]
            t_range.insert(tklist[0], tfill[i])
            onrack.insert(tklist[0], onrack[tklist[0] - 1])
            for t in range(tklist[0], len(t_range)):
                onrack[t] -= 1
    return t_range, onrack


def addtobin(i, j, m, x, y, r, c, item, tfill):
    # If i = m, then a new bin needs to be opened.
    if i == m:
        m += 1
        x[m - 1, j] = 1
        y[m - 1] = 1
        r[m - 1] = c - 1
        # If tfill was not assigned at i = m-1...
        if tfill[m - 1] == 0:
            tfill[m - 1] = item.getbatch() * 600 + 150
        # If tfill was assigned at i = m-1...
        else:
            pack = packable(r[m - 1], tfill[m - 1], item)
            if all(pack) is not True:
                tfill[m - 1] = item.getbatch() * 600 + 150
    else:
        x[i, j] = 1
        r[i] -= 1
    return m, x, y, r, tfill


def combo(permutation, tfill, bpp):
    # This module decodes the permutation based on the suggested
    # combination of ll and dp strategies in Dahmani (2014).
    split = 0.30
    m = int(bpp.getlb())    # initialize lower bound on bins
    items = bpp.getitems()
    n, c, x, y, r = initial(m, bpp)
    switch = int(round(n*split))
    # Set weights for the dot product array
    weights = [1.0 / c, 1.0 / bpp.rack]  # 1/boxcap, 1/coolrackcap
    t_range, onrack = get_coolrack_variation(tfill, bpp)
    # Perform least loaded moves before dot product moves.
    for j in range(n):
        cj = permutation[j]
        item = items.get(cj)
        if j < switch:
            i = llmove(n, m, r, item, tfill)
        else:
            i = dpmove(m, r, item, tfill, weights, t_range, onrack)
        m, x, y, r, tfill = addtobin(i, cj, m, x, y, r, c, item, tfill)
        t_range, onrack = update_coolrack_variation(i, tfill, t_range, onrack)
    return x, y, tfill


def initial(m, bpp):
    # This module initializes the elements common to all strategies.
    n = bpp.n                           # Number of cookies to sort
    c = int(bpp.getub())                # initialize max capacity
    x = np.zeros((n, n), dtype=np.int)  # initialize x
    y = np.zeros(n, dtype=np.int)       # initialize y
    for i in range(m):                  # initialize y
        y[i] = 1
    r = np.zeros(n, dtype=np.int)       # initialize capacities
    for i in range(m):                  # initialize r (residual matrix)
        r[i] = c
    return n, c, x, y, r


def packable(rone, tfilli, item):
    # This module checks to see if object j can fit inside bin i at time tfilli
    # Capacity constraint
    rc = rone - 1
    cappack = (rc >= 0)
    # Time constraint
    #   tbatch = 10 min = 600 s
    timepack = (item.getbatch() * 600 < tfilli)
    return cappack, timepack


def coordarrays(solution):
    # This function makes sure that tfill matches the number of open bins
    # and that open bins actually have something in them.
    n = solution.n
    # Match up matrices
    for i in range(n):
        if sum(solution.x[i, :]) == 0:
            solution.y[i] = 0
        else:
            solution.y[i] = 1
        if solution.y[i] == 0:
            solution.tfill[i] = 0
    # Check for empty bins
    solution.initopenbins()
    emptyrows = np.all(np.equal(solution.y[solution.openbins:], 0))
    if len(solution.vlrep) > solution.openbins or not emptyrows:
        solution = remove_empty_bins(n, solution)
    return solution


def remove_empty_bins(n, solution):
    # Remove empty bins
    for i in range(n):
        if solution.y[i] == 0:
            # Calculate s
            s = findrowspace(n, i, solution.gety())
            print(i, s)
            if s != None:
                # Move rows down by s rows
                for i2 in range(i, n - s):
                    if solution.y[i2 + s] == 0:
                        break
                    solution.y[i2] = 1
                    solution.y[i2 + s] = 0
                    solution.tfill[i2] = solution.tfill[i2 + s]
                    solution.tfill[i2 + s] = 0.0
                    if solution.vlrep[i2] != []:
                        print(' coordarrays: added bin')
                        solution.vlrep.insert(i2, [])
                        checkformismatch(solution)
                    for j in range(n):
                        if solution.x[i2 + s, j] == 1:
                            solution.moveitem(i2 + s, j, i2)
    solution.removeunneededbins()
    return solution


def findrowspace(n, istart, y):
    # This function determines the number of rows until the next open bin
    for i2 in range(istart + 1, n):
        if y[i2] == 1:
            s = i2 - istart
            return s
    return None


def repackitems(m, x, y, tfill, bpp, t, binitems):
    # This module repacks items from existing bins into new bins at
    # time t. Only the dot product strategy is used to try to minimize
    # the number of extra bins added. binitems is a list of (i, j) tuples
    # denoting where cookie j is currently located in the x-matrix.
    # Set weights for the dot product array
    weights = [1, 371.5]                # 1 item/item, avg. temp. during cooling
    m, c, y, tfill, r = initializerepack(m, x, y, tfill, bpp, t)
    for i, j in binitems:
        item = bpp.items.get(j)
        # Delete cookie j from bin i
        x[i, j] = 0
        # Add cookie to new bin
        inew = dpmove(m, r, item, tfill, weights)
        m, x, y, r, tfill = addtobin(inew, j, m, x, y, r, c, item, tfill)
    return x, y, tfill


def initializerepack(m, x, y, tfill, bpp, t):
    # This function initializes the variables needed by repackitems_newbin
    # It sets the variables so that available bins meet the criteria tfill[i]
    # is less than time t.
    c = int(bpp.getub())                # Bin capacity
    # Find bins that meet criteria
    ilist = np.where(tfill[:m] <= t)
    r = np.zeros(bpp.n, dtype=np.int)   # initialize capacities
    # If ilist is empty, open new bin:
    if not np.greater_equal(len(ilist[0]), 1):
        y, tfill, r, m = openonebin(m, y, tfill, r, c, t)
    # If other bins can accept items:
    else:
        # Edit residual capacity array
        for i in ilist[0]:
            r[i] = c
            for j in range(len(y)):
                if x[i, j] == 1:
                    r[i] -= 1
        # If all boxes are full, open new bin
        if np.all(np.equal(r, 0)):
            y, tfill, r, m = openonebin(m, y, tfill, r, c, t)
    return m, c, y, tfill, r


def openonebin(m, y, tfill, r, c, t):
    y[m] = 1        # open one bin
    r[m] = c        # open one bin
    tfill[m] = t    # open bin at time t
    m += 1
    return y, tfill, r, m


def initialrepack(x, y, starti, endi, bpp):
    # Fix this!
    # This function initializes variables needed for repack
    if y[endi] == 1:        # initialize lower bound on bins
        m = endi
    else:
        m = int(bpp.getlb())
    h = int(bpp.getub())    # initialize max length
    w = int(bpp.getwbin())  # initialize max weight
    n = len(y)              # get number of items
    r = np.zeros((len(y), 2), dtype=np.int)  # initialize capacities
    for i in range(starti, endi):
        r[i, :] = [w, h]
    # Get list of items to repack
    packitems = []
    for i in range(starti, endi):
        if i >= m:
            y[i] = 0
        for j in range(n):
            if x[i, j] == 1:
                packitems.append(j + 1)
                x[i, j] = 0
    return m, h, w, n, x, y, r, packitems


def checkformismatch(solution, out=sys.stdout):
    # This function identifies if the given solution does not have an x-matrix
    # and a variable length representation that match.
    x = solution.getx()
    vlrep = solution.getvlrep()
    for i in range(len(vlrep)):
        for j in vlrep[i]:
            if x[i, j] != 1:
                out.write('Error: Solution {0:d} is not coordinated on item {1:d}.'
                          .format(solution.getid(), j))


def xycheck(m, x, y):
    # This function verifies that the number of open bins in x and y agree
    n = len(y)
    itembins = np.sum(x, axis=1)
    for i in range(n):
        if (itembins[i] > 0 and y[i] == 1) is False:
            if (itembins[i] == 0 and y[i] == 0) is False:
                print('Solution', m, 'has an open bin error: bin', i)


def main():
    # Get info to create items
    # n = eval(input('Please enter the number of items to be sorted: \n'))
    # folder = input('Please enter the name of the folder where your input file is: \n')
    # datafile = input('Please enter the name of the input file: \n')
    n = 1000
    folder = 'tests/'
    datafile = 'Cookies1000.txt'
    random.seed(50)

    # Create item objects and initialize a bin packing problem class
    data = folder + datafile
    binc, binh, items = makeitems(data)
    bpp = BPP(n, 24, 300, items)

    # Input items into the heuristic in the way they were ordered in
    # the datafile. Each "gene" should correspond to an item's index.
    solid = 1                       # give this solution an id number
    chromosome = list(range(1, n + 1))
    x, y = ed(solid, chromosome, bpp)
    np.savetxt(folder + str(solid) + '_x.txt', x, fmt='%i', header='Item Location Matrix x:')
    np.savetxt(folder + str(solid) + '_y.txt', y, fmt='%i', header='Bins Used Matrix y:')


if __name__ == '__main__':
    main()
