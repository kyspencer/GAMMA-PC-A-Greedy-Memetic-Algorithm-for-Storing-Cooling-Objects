# solutions.py
#   This python file contains modules save solutions.
#   Author: Kristina Spencer
#   Date: March 31, 2016

import binpacking as bp
import mop
import random
from constraints import concheck
from numpy import zeros
from operator import itemgetter


def main():
    print('This file saves solutions in a bpp optimization.')


def process(idnum, t, chromosome, bpp, items):
    # Process solutions (decode, check constraints, calculate
    # fitness values, make solution object)
    x, y = bp.ed(idnum, chromosome, bpp, items)
    concheck(idnum, x, bpp)
    fit = mop.calcfits(x, y, items)
    a = MultiSol(idnum, chromosome, x, y, t, fit, 0, 0.0)
    return a


def oldnew(archive, q, genes):
    # This module checks the new generation to see if its
    # members already exist or need to be created.
    #  - archive is the set of all current solutions
    #  - q is the new generation
    #  - genes is only the chromosome portion of q
    #  - members is the number of individuals in a gen.
    new = []
    archgenes = []
    for m in range(len(archive)):
        archgenes.append(archive[m].getgenes())
    k = 0
    for p in range(len(genes)):
        count = archgenes.count(genes[p])
        if count == 0:
            new.append(p)
            del q[k]
            k -= 1
        k += 1
    return new, q


def reduce(archive, p, q):
    # This module keeps the length of the archive below 1000 individual
    # solutions to save computer memory during runtime.
    #   - archive is the list of all solutions
    #   - p is the parent generation
    #   - q is the next generation
    from operator import attrgetter
    if len(archive) > 1200:
        archive.sort(key=attrgetter('rank'))
        k = 1000
        for m in range(k, len(archive)):
            if archive[k] in p:
                k += 1
            elif archive[k] in q:
                k += 1
            else:
                del archive[k]
        archive.sort(key=attrgetter('index'))
    return archive


class Sol:
    def __init__(self, index, chromosome, x, y, t, fitvals, prank):
        self.index = index
        self.genes = chromosome
        self.n = len(self.genes)
        self.x = x
        self.y = y
        self.t = int(t)
        self.fits = fitvals
        self.rank = prank

    def getindex(self):
        return self.index

    def updateid(self, idnum):
        self.index = idnum

    def getgenes(self):
        return self.genes

    def getx(self):
        return self.x

    def gety(self):
        return self.y

    def getgen(self):
        return self.t

    def getfits(self):
        return self.fits
        
    def getrank(self):
        return self.rank
        
    def updaterank(self, prank):
        self.rank = prank


class MultiSol(Sol):
    def __init__(self, index, chromosome, x, y, t, fitvals, prank, idist):
        Sol.__init__(self, index, chromosome, x, y, t, fitvals, prank)
        self.fit0 = fitvals[0]
        self.fit1 = fitvals[1]
        self.fit2 = fitvals[2]
        self.cd = idist

    def getbins(self):
        return self.fit0

    def getmaxh(self):
        return self.fit1

    def getavgw(self):
        return self.fit2

    def getcd(self):
        return self.cd

    def updatecd(self, idist):
        self.cd = idist


class GAMMASol(Sol):
    def __init__(self, index, x, y, vlrep, t, chrom=None, prank=0):
        self.cd = 0.0                           # Initialize crowded distance value
        self.vlrep = vlrep
        self.bin_weights = zeros(len(y))        # Initialize bin weight array
        self.bin_heights = zeros(len(y))        # Initialize bin height array
        fitvals = zeros(3)                      # Initialize fitness vector
        if not chrom:
            chrom = self.vlrep2chrom(vlrep)
        Sol.__init__(self, index, chrom, x, y, t, fitvals, prank)
        self.openbins = 0
        self.initopenbins()

    def updatefitvals(self, fitvals):
        self.fits = fitvals
        self.fit0 = fitvals[0]
        self.fit1 = fitvals[1]
        self.fit2 = fitvals[2]

    def set_weights(self, weights):
        if len(weights) != len(self.bin_weights):
            print('Error!  The length of the bin weight array is not ', self.n)
        self.bin_weights = weights

    def set_heights(self, heights):
        if len(heights) != len(self.bin_heights):
            print('Error!  The length of the bin height array is not ', self.n)
        self.bin_heights = heights

    def swapitems(self, i1, j1, i2, j2):
        # This function swaps the cookies j1 and j2 between boxes i1 and i2
        # Swap in the x-matrix
        self.x[i1, j1] = 0
        self.x[i2, j1] = 1
        self.x[i2, j2] = 0
        self.x[i1, j2] = 1
        # Swap in the variable length representation
        self.vlrep[i1].remove(j1)
        self.vlrep[i2].append(j1)
        self.vlrep[i2].remove(j2)
        self.vlrep[i1].append(j2)
        # Resort the bins to keep js in order
        self.vlrep[i1].sort()
        self.vlrep[i2].sort()

    def moveitem(self, i, j, inew):
        # This function moves cookie j2 from box i2 to box i
        # Move in variable length representation
        self.vlrep[i].remove(j)
        self.vlrep[inew].append(j)
        # Move in x-matrix
        self.x[i, j] = 0
        self.x[inew, j] = 1
        # Resort bin inew to keep js in order
        self.vlrep[inew].sort()
        # Check y-matrix
        if not self.vlrep[i]:
            self.closebin(i)
            if inew > i:
                inew -= 1
        if self.y[inew] == 0:
            self.y[inew] = 1

    def opennewbin(self, i, j):
        # This function moves cookie j from box i into box inew at time t
        inew = len(self.vlrep)
        # Move in x-matrix:
        self.x[i, j] = 0
        self.x[inew, j] = 1
        # Open new box in y-matrix:
        self.y[inew] = 1
        # Open new box in vlrep
        self.vlrep[i].remove(j)
        self.vlrep.append([j])
        self.openbins = len(self.vlrep)

    def closebin(self, i):
        # This function closes bin i after it has been emptied
        if self.vlrep[i] == []:
            del self.vlrep[i]
            # Move to close empty rows
            for imove in range(i, self.n - 1):
                self.y[imove] = self.y[imove + 1]
                self.x[imove, :] = self.x[imove + 1, :]
            self.y[-1] = 0
            self.x[-1, :] = 0
        self.initopenbins()

    def vlrep2chrom(self, vlrep):
        # This function reforms vlrep into the chromosome representation
        chrom = list(vlrep[0])
        for i in range(1, len(vlrep)):
            chrom.extend(list(vlrep[i]))
        return chrom

    def removeunneededbins(self):
        # This function removes empty bins from the end of the vlrep list
        for i in range(self.openbins, self.n):
            if len(self.vlrep) == self.openbins:
                break
            if self.vlrep[self.openbins] == []:
                del self.vlrep[self.openbins]
            else:
                print('Error: y does not match vlrep in solution', self.index)

    def initopenbins(self):
        # This function determines the number of open bins based on the y-matrix
        self.openbins = int(sum(self.y))

    def getbins(self):
        return self.fit0

    def getmaxh(self):
        return self.fit1

    def get_heights(self):
        return self.bin_heights

    def getavgw(self):
        return self.fit2

    def get_weights(self):
        return self.bin_weights

    def getvlrep(self, i=None):
        if i:
            return self.vlrep[i]
        elif i == 0:
            return self.vlrep[0]
        else:
            return self.vlrep

    def getopenbins(self):
        return self.openbins

    def getcd(self):
        return self.cd

    def updatecd(self, idist):
        self.cd = idist


class PSOSol(Sol):
    # This subclass of solutions is built specifically for a PSO algorithm.
    def __init__(self, index, chromosome, x, y, t, fitvals, prank, niche):
        Sol.__init__(self, index, chromosome, x, y, t, fitvals, prank)
        self.fit0 = fitvals[0]
        self.fit1 = fitvals[1]
        self.fit2 = fitvals[2]
        self.niche = niche
        self.vlrep = []
        self.binws = []  # individual bin weights
        self.binhs = []  # individual bin heights

    def makevlrep(self, vlrep, items):
        # Note: vlrep has item index in it, which is j + 1
        self.vlrep = list(vlrep)
        for i in range(len(self.vlrep)):
            weight = 0
            height = 0
            for j in range(len(self.vlrep[i])):
                index = self.vlrep[i][j]
                weight += items[index - 1].getweight()
                height += items[index - 1].getheight()
            self.binws.append(weight)
            self.binhs.append(height)

    def getpbest(self):
        # This function returns the most filled bin
        # Randomly chooses "most filled" from weight or height
        wmaxindex, wmax = max(enumerate(self.binws), key=itemgetter(1))
        hmaxindex, hmax = max(enumerate(self.binhs), key=itemgetter(1))
        wmaxbin = self.vlrep[wmaxindex]
        hmaxbin = self.vlrep[hmaxindex]
        binitems = random.choice([wmaxbin, hmaxbin])
        return binitems

    def getbins(self):
        return self.fit0

    def getmaxh(self):
        return self.fit1

    def getavgw(self):
        return self.fit2

    def getniche(self):
        return self.niche

    def updateniche(self, nichecount):
        self.niche = nichecount

    def getvlrep(self):
        return self.vlrep


if __name__ == '__main__':
    main()
