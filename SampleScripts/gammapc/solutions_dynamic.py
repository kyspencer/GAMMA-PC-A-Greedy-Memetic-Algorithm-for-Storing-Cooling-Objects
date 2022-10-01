# solutions.py
#   This python file contains modules save solutions.
#   Author: Kristina Spencer
#   Date: March 31, 2016

from . import binpacking_dynamic as bp
import random
from numpy import allclose, sum, zeros
from operator import itemgetter


def oldnew(archive, q, genes):
    # This module checks the new generation to see if its
    # members already exist or need to be created.
    #  - archive is the set of all stored solutions
    #  - q is formerly found solutions belonging to the next generation
    #  - genes is a list of tuples (chromosome, tfill)
    # Collect one list of keys and one list of chromosomes
    indices = []
    archgenes = []
    for key, solution in archive.items():
        indices.append(key)
        archgenes.append(solution.getgenes())
    # Go through genes to find old solutions
    k = 0
    for p in range(len(genes)):
        count = archgenes.count(genes[k][0])
        if count > 0:
            i = archgenes.index(genes[k][0])
            # If tfill is same:
            if allclose(archive.get(indices[i]).gettfill(), genes[k][1]):
                q.append(archive.get(indices[i]))
                # Remove genes[p]
                del genes[k]
                k -= 1
        k += 1
    return genes, q


class Sol:
    def __init__(self, index, chromosome, tfill):
        self.index = index
        self.genes = chromosome
        self.tfill = tfill
        self.n = len(self.genes)
        self.fits = 0
        self.rank = 0

    def edit_tfilli(self, i, t):
        # This function changes self.tfill[i] to t
        self.tfill[i] = t

    def getid(self):
        return self.index

    def updateid(self, idnum):
        self.index = idnum

    def getgenes(self):
        return self.genes

    def gettfill(self):
        return self.tfill

    def getfits(self):
        return self.fits

    def updaterank(self, prank):
        self.rank = prank
        
    def getrank(self):
        return self.rank


class CookieSol(Sol):
    def __init__(self, index, x, y, vlrep, tfill):
        chrom = self.vlrep2chrom(vlrep)
        Sol.__init__(self, index, chrom, tfill)
        self.x = x
        self.y = y
        self.vlrep = vlrep
        self.tavail = zeros(len(self.y))    # Initialize tavail
        self.q0bins = zeros(len(self.y))    # Initialize initial heat in bins
        self.fits = zeros(3)
        self.cd = 0
        self.openbins = 0
        self.initopenbins()

    def updatefitvals(self, fitvals):
        self.fits = fitvals
        self.fit0 = fitvals[0]
        self.fit1 = fitvals[1]
        self.fit2 = fitvals[2]

    def settavailable(self, tavail):
        self.tavail = tavail

    def setq0bins(self, q0bins):
        self.q0bins = q0bins

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

    def opennewbin(self, i, j, t):
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
        # Open time t
        self.tfill[inew] = t

    def closebin(self, i):
        # This function closes bin i after it has been emptied
        if self.vlrep[i] == []:
            del self.vlrep[i]
            # Move to close empty rows
            for imove in range(i, self.n - 1):
                self.y[imove] = self.y[imove + 1]
                self.tfill[imove] = self.tfill[imove + 1]
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

    def initopenbins(self):
        # This function determines the number of open bins based on the y-matrix
        self.openbins = int(sum(self.y))

    def removeunneededbins(self):
        # This function removes empty bins from the end of the vlrep list
        for i in range(self.openbins, self.n):
            if len(self.vlrep) == self.openbins:
                break
            if self.vlrep[self.openbins] == []:
                del self.vlrep[self.openbins]
            else:
                print('Error: y does not match vlrep in solution', self.index)

    def getx(self):
        return self.x

    def gety(self):
        return self.y

    def gettavail(self):
        return self.tavail

    def getq0bins(self):
        return self.q0bins

    def getbins(self):
        return self.fits[0]

    def getavgheat(self):
        return self.fits[1]

    def getmaxreadyt(self):
        return self.fits[2]

    def updatecd(self, idist):
        self.cd = idist

    def getcd(self):
        return self.cd

    def getvlrep(self, i=None):
        if i:
            return self.vlrep[i]
        elif i == 0:
            return self.vlrep[0]
        else:
            return self.vlrep

    def getopenbins(self):
        return self.openbins


class MultiSol(Sol):
    def __init__(self, index, chromosome, tfill, bpp):
        Sol.__init__(self, index, chromosome, tfill)
        self.bpp = bpp
        self.x, self.y, self.tfill = bp.ed(self.genes, self.tfill, self.bpp)
        self.tavail = zeros(len(self.y))    # Initialize tavail
        self.q0bins = zeros(len(self.y))    # Initialize initial heat in bins
        self.fits = zeros(3)
        self.cd = 0
        self.vlrep = []
        self.openbins = 0
        self.initopenbins()
        self.makevlrep()

    def makevlrep(self):
        # This function transforms the x-matrix into the variable
        # length representation of the loading pattern
        for i in range(self.openbins):
            bini = []
            for j in range(self.n):
                if self.x[i, j] == 1:
                    bini.append(j)
            self.vlrep.append(bini)

    def updatefitvals(self, fitvals):
        self.fits = fitvals
        self.fit0 = fitvals[0]
        self.fit1 = fitvals[1]
        self.fit2 = fitvals[2]

    def settavailable(self, tavail):
        self.tavail = tavail

    def setq0bins(self, q0bins):
        self.q0bins = q0bins

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
        if self.y[inew] == 0:
            self.y[inew] = 1

    def repackitemsatt(self, t, binitems):
        # This function adds bin(s) at time t
        # binitems is a list of (i, j) tuples denoting where
        # cookie j is currently located in the x-matrix.
        self.x, self.y, self.tfill = bp.repackitems(self.openbins, self.x,
                                                    self.y, self.tfill,
                                                    self.bpp, t, binitems)
        self.vlrep = []
        self.initopenbins()
        self.makevlrep()

    def opennewbin(self, i, j, t):
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
        # Open time t
        self.tfill[inew] = t

    def closebin(self, i):
        # This function closes bin i after it has been emptied
        if not self.vlrep[i]:
            del self.vlrep[i]
            # Move to close empty rows
            for imove in range(i, self.n - 1):
                self.y[imove] = self.y[imove + 1]
                self.tfill[imove] = self.tfill[imove + 1]
                self.x[imove, :] = self.x[imove + 1, :]
        self.initopenbins()

    def initopenbins(self):
        self.openbins = int(sum(self.y))

    def removeunneededbins(self):
        # This function removes unneeded bins from the vlrep
        self.initopenbins()
        if len(self.vlrep) > self.openbins:
            for i in range(self.openbins, self.n):
                if len(self.vlrep) == self.openbins:
                    break
                if self.vlrep[self.openbins] == []:
                    del self.vlrep[self.openbins]
                else:
                    print('Error: y does not match vlrep in solution', self.index)

    def getx(self):
        return self.x

    def gety(self):
        return self.y

    def gettavail(self):
        return self.tavail

    def getq0bins(self):
        return self.q0bins

    def getbins(self):
        return self.fit0

    def getavgheat(self):
        return self.fit1

    def getmaxreadyt(self):
        return self.fit2

    def updatecd(self, idist):
        self.cd = idist

    def getcd(self):
        return self.cd

    def getvlrep(self, i=None):
        if i:
            return self.vlrep[i]
        elif i == 0:
            return self.vlrep[0]
        else:
            return self.vlrep

    def getopenbins(self):
        return self.openbins


class PSOSol(Sol):
    # This subclass of solutions is built specifically for a PSO algorithm.
    def __init__(self, index, chromosome, x, y, tfill, fitvals, niche):
        Sol.__init__(self, index, chromosome, x, y, tfill)
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
    print('This file saves solutions in a bpp optimization.')
