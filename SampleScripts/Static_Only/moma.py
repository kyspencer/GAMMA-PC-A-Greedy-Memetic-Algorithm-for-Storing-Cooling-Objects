# moma.py
#   This python script automates MOMA as described in Ishibuchi (2009).
#   Author: Kristina Yancey Spencer
#    i - bin index
#    j - item index
#    k - algorithm index
#    m - solution index
#    t - generation index

from __future__ import print_function
import binpacking as bp
import constraints
import ga
import mop
import numpy as np
import os
import outformat as outf
import random
import solutions as sols
from datetime import datetime
from glob import glob
from items import makeitems
from operator import attrgetter


def moma(n, folder, datafile):
    existing_files = glob(folder + '*.out')
    filename = folder + 'run%d.out' % (len(existing_files) + 1)
    data = folder + datafile

    # Initialize algorithm
    pop = 100       # members per gen.
    end = 25000     # function evaluations or 250 gen.
    outf.startout(filename, n, end, data, 'MOMA')
    startt = datetime.now()
    print('                         ', startt)
    print('*******************************************************************************')
    print('     Method: MOMA\n')
    print('     data: ', datafile)
    print('*******************************************************************************')
    binc, binh, items = makeitems(data)
    bpp = bp.BPP(n, binc, binh, items)
    moop = mop.MOproblem(pop, items)
    gen = Generation(n, pop, end, items, bpp, moop)
    gen.initialp()
    gen.makeq()

    # NSGA-II - Local search
    while not gen.endgen():
        gen.rungen()
        outf.genout(filename, gen.gett(), pop, gen.getq(), [gen.getarch(), []])

    # Get final nondominated set
    aslimit = 75
    r, allfronts = fnds(gen.getarch())
    if len(allfronts[0]) > aslimit:
        gen.fittruncation(allfronts[0], aslimit)
    ndset = approx(gen.getarch())

    # Make output
    see(ndset, folder)
    import csv
    with open(folder + 'ApproximationSet.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, dialect='excel', delimiter=',')
        csvwriter.writerow(['Solution ID', 'f[1]', 'f[2]', 'f[3]'])
        for m in ndset:
            csvwriter.writerow([m.getindex(), m.getbins(), m.getmaxh(), m.getavgw()])
    outf.endout(filename)
    print('This algorithm has completed successfully.')


class Generation:
    def __init__(self, n, popsize, end, items, bpp, moop):
        self.n = int(n)
        self.pop = int(popsize)
        self.end = int(end)
        self.items = items
        self.bpp = bpp
        self.moop = moop
        self.t = 0
        self.idnum = 0
        self.p = []
        self.q = []
        self.archive = []   # Solution Archive
        self.newgenes = []  # List of new genes
        self.funkeval = 0

    def rungen(self):
        self.t += 1
        #if self.t % 50 == 0:
        gent = datetime.now()
        print('                         ', gent)
        print('t = ', self.t)
        print('Selecting parent population...')
        self.makep()
        print('GA operation: binary selection...')
        if self.t == 1:
            self.q = ga.binsel(self.p, self.pop, 'elitism')
        else:
            self.q = ga.binsel(self.p, self.pop, 'cco')
        print('GA operation: crossover...')
        self.newgenes = ga.xover(self.q, self.pop, 0.8)
        print('GA operation: mutation...')
        self.evoloperators()
        if self.t % 10 == 0:
            self.localsearch()
        print(self.funkeval, 'function evaluations have been performed.\n')

    def initialp(self):
        # This function creates the initial set of chromosomes for gen. 0
        # Create list of items, i.e. chromosomes
        chromosomes = []
        for j in range(self.n):
            chromosomes.append(self.items[j].getindex())
        # Generate popsize random combinations of the chromosomes
        random.seed(52)
        self.newgenes = []
        for i in range(self.pop):
            x = random.sample(chromosomes, len(chromosomes))
            self.newgenes.append(x)

    def makep(self):
        r = self.p + self.q
        r, fronts = fnds(r)
        print(' There are currently ', len(fronts[0]), 'solutions in the Approximate Set.')
        fronts[0] = self.cleanarchive(fronts[0])
        if self.t == 1:
            self.p = r
        else:
            self.fill(fronts)

    def fill(self, fronts):
        # This module fills the parent population based on the crowded
        # comparison operator.
        self.p = []
        k = 0
        fronts[k] = cda(fronts[k], 3)
        while (len(self.p) + len(fronts[k])) < self.pop:
            self.p = self.p + fronts[k]
            k += 1
            if fronts[k] == []:
                needed = self.pop - len(self.p)
                for n in range(needed):
                    if n <= len(fronts[0]):
                        m = random.choice(fronts[0])
                    else:
                        m = random.choice(fronts[1])
                    nls = 5
                    mneighbors = self.paretols(m, 5, retrieve=True)
                    while mneighbors == []:
                        nls += 5
                        mneighbors = self.paretols(m, nls, retrieve=True)
                    fronts[k].append(random.choice(mneighbors))
            fronts[k] = cda(fronts[k], 3)
        fronts[k].sort(key=attrgetter('cd'), reverse=True)
        fillsize = self.pop - len(self.p)
        for l in range(fillsize):
            self.p.append(fronts[k][l])

    def evoloperators(self):
        # This module decides if a generation will evolve by chromosome
        # or by bin packing operators.
        rannum = random.random()
        if rannum < 1:  #0.5:    # chromosome operators
            self.newgenes = ga.mutat(self.n, self.newgenes, self.pop)
            # Form into solutions of the next generation
            self.makeq()
        else:               # bin packing operators
            self.makeqbybins()

    def makeq(self):
        if self.t == 0:
            new = range(self.pop)
        else:
            new, self.q = sols.oldnew(self.archive, self.q, self.newgenes)
        # Make new solutions in q
        for m in new:
            x, y = bp.ed(self.idnum, self.newgenes[m], self.bpp)
            self.addnewsol(x, y, self.newgenes[m])

    def makeqbybins(self):
        # This function is almost the same as makeq() except modified for
        # the bin packing operators.
        # Sort out what solutions were modified by the crossover operation
        new, self.q = sols.oldnew(self.archive, self.q, self.newgenes)
        # First go through what's left in q
        for m in range(len(self.q)):
            qx = self.q[m].getx()
            qy = self.q[m].gety()
            x, y = self.binmutation(qx, qy)
            if np.all(np.equal(x, qx)) is False:     # Indicates new solution
                del self.q[m]
                newchrom = self.updatechrom(x)
                self.addnewsol(x, y, newchrom)
        # Then operate on new solutions
        for m in new:
            newx, newy = bp.ed(self.idnum, self.newgenes[m], self.bpp)
            x, y = self.binmutation(newx, newy)
            if np.all(np.equal(x, newx)) is False:
                self.newgenes[m] = self.updatechrom(x)
            self.addnewsol(x, y, self.newgenes[m])

    def addnewsol(self, x, y, chromosome):
        # This function makes a new solution out of x and y and archives it
        constraints.concheck(self.idnum, x, self.bpp)
        constraints.xycheck(self.idnum, x, y)
        fit = self.moop.calcfits(x, y)
        self.updatefe()
        newsol = sols.MultiSol(self.idnum, chromosome, x, y, self.t, fit, 0, 0.0)
        self.q.append(newsol)
        self.archive.append(newsol)
        self.updateid()

    def cleanarchive(self, approxfront):
        # This function removes unwanted solutions from the archive
        # and calls the local search function.
        aslimit = 75    # Limit to size of approximation set
        if len(approxfront) > aslimit:
            approxfront = self.ccotruncation(approxfront, aslimit)
        # Every 50 generations check the archive
        if self.t % 50 == 0:
            set = [m for m in self.archive if m.getrank() == 1]
            irrelevant, archfronts = fnds(set)
            # Truncate approximate set
            if len(archfronts[0]) > aslimit:
                self.fittruncation(archfronts[0], aslimit)
        self.archive = sols.reduce(self.archive, self.p, self.q)
        return approxfront

    def ccotruncation(self, approxfront, limit):
        # This function performs a recurrent truncation of the
        # approximate set based on cco.
        print(' Removing superfluous solutions from the front.')
        # Sort approxfront by the crowded distance indicator
        approxfront.sort(key=attrgetter('cd'))
        # Remove solutions with the lowest distance first
        nremove = len(approxfront) - limit
        for m in range(nremove):
            if approxfront[m] in self.archive:
                self.archive.remove(approxfront[m])
        del approxfront[:nremove]
        approxfront.sort(key=attrgetter('index'))
        print(' There are now', len(approxfront), 'solutions in the Approximate Set.')
        return approxfront

    def fittruncation(self, approxfront, limit):
        # This function removes solutions from the approximate set that have the same
        # fitness values.
        print('Removing superfluous solutions from the archive.')
        nremove = len(approxfront) - limit
        clusters = self.getclusters(approxfront)
        nrm = 0
        for k in range(len(clusters[0])):
            cdremove = clusters[0][0].getcd()
            for c in range(len(clusters)):
                if nrm < nremove:
                    if len(clusters[c]) > 1:
                        if clusters[c][0].getcd() <= cdremove:
                            if clusters[c][0] not in self.p or self.q:
                                self.archive.remove(clusters[c][0])
                                nrm += 1
                            clusters[c].remove(clusters[c][0])
        print('There are now', len(approxfront) - nrm, 'solutions in the Approximate Set.')

    def getclusters(self, front):
        # This function sorts a front into clusters of solutions
        # Each cluster has the same number of bins in the solution
        front.sort(key=attrgetter('fit0'))
        clusters = []
        m1 = 0
        while m1 < len(front):
            numbins = front[m1].getbins()
            fitlist = []
            for m2 in range(len(front)):
                if front[m2].getbins() == numbins:
                    fitlist.append(front[m2])
                    m1 += 1
            # Not a cluster if a solution is by itself
            if len(fitlist) > 1:
                fitlist.sort(key=attrgetter('cd'))
                clusters.append(fitlist)
        orderedbylen = sorted(clusters, key=len, reverse=True)
        return orderedbylen

    def localsearch(self):
        # This function performs the Pareto dominance local search with
        # the probability of search increasing from 0 to 0.2 over the course
        # of the calculations.
        probls = 0  # Probability of local search
        # Update probability for progress in calculations
        if self.funkeval > 0.9 * self.end:
            probls = 0.2
        else:
            probls = 0.2 * float(self.funkeval) / (0.9 * self.end)
        nls = 10
        for m in range(self.pop):
            ranls = random.random()
            if ranls <= probls:
                self.paretols(self.p[m], nls)

    def paretols(self, m, nls, retrieve=False):
        # This function finds up to nls neighbors to check if they dominate
        # solution m of the new generation.
        x = m.getx()
        y = m.gety()
        neighbors = []
        for k in range(nls):
            newx, newy = self.binmutation(x, y)
            newfit = self.moop.calcfits(newx, newy)
            self.updatefe()
            # If we find a neighbor that is nondominated by solution m, add to q.
            if not mop.dom(m.getfits(), newfit):
                neighbors = self.addneighbor(newx, newy, newfit, neighbors)
        if retrieve is True:
            return neighbors

    def addneighbor(self, x, y, fit, neighbors):
        # This function creates a new solution based on local search and
        # adds it to the neighbor list.
        print(' Local search found another possible solution:', self.idnum)
        newgenes = self.updatechrom(x)
        newsol = sols.MultiSol(self.idnum, newgenes, x, y, self.t, fit, 0, 0.0)
        neighbors.append(newsol)
        self.q.append(newsol)
        self.archive.append(newsol)
        self.updateid()
        return neighbors

    def binmutation(self, x, y):
        # This function performs bin-specific mutations at a mutation
        # probability of 0.3
        prob = 1    #0.3
        ranmute = random.random()
        if ranmute < prob:
            ranm1 = random.random()
            if ranm1 < 0.25:
                x, y = self.partswap(x, y)
            elif ranm1 < 0.50:
                x, y = self.mergebins(x, y)
            elif ranm1 < 0.75:
                x, y = self.splitbin(x, y)
            else:
                x, y = self.itemswap(x, y)
        return x, y

    def itemswap(self, x, y):
        # This function performs a 2-item swap:
        #   with an item from the fullest bin
        #   with an item from the emptiest bin
        #   with an item from a random bin
        openbins = self.findopenbins(y)
        # Select random bins
        i1 = self.getmaxminranbin(x, openbins)
        i2 = self.getrandsecondbin(i1, x, openbins)
        try:
            j1, j2 = self.getitemstoswap(x, i1, i2)
            x[i1, j1] = 0
            x[i1, j2] = 1
            x[i2, j2] = 0
            x[i2, j1] = 1
        except:
            binitems1 = self.finditemsinbin(x, i1)
            if len(binitems1) != 1:
                j1 = random.choice(binitems1) - 1
                x[i1, j1] = 0
                i2 = openbins[-1] + 1
                x[i2, j1] = 1
                y[i2] = 1
        return x, y

    def getitemstoswap(self, x, i1, i2):
        # This function returns two items in bins i1 and i2 that
        # can be feasibly swapped.
        # Find items in bin i1 and bin i2
        binitems1 = self.finditemsinbin(x, i1)
        binitems2 = self.finditemsinbin(x, i2)
        for count1 in range(len(binitems1)):
            for count2 in range(len(binitems2)):
                # Select random items in chosen bins
                # Remember binitems is list of item indices, where index = j + 1
                j1 = random.choice(binitems1) - 1
                j2 = random.choice(binitems2) - 1
                # Check for feasibility
                newbin1 = list(binitems1)
                newbin1.remove(j1 + 1)
                newbin1.append(j2 + 1)
                check1 = constraints.bincheck(newbin1, self.bpp)
                newbin2 = list(binitems2)
                newbin2.remove(j2 + 1)
                newbin2.append(j1 + 1)
                check2 = constraints.bincheck(newbin2, self.bpp)
                # violation if bincheck returns True
                truthvalue = check1 or check2
                # Stop module and return values if found items to swap
                if truthvalue is False:
                    return j1, j2
        return False

    def partswap(self, x, y):
        # This function swaps parts of two bins with each other
        openbins = self.findopenbins(y)
        # Pick two random bins
        i1 = random.choice(openbins)
        binitems1 = self.finditemsinbin(x, i1)
        while len(binitems1) == 1:
            i1 = random.choice(openbins)
            binitems1 = self.finditemsinbin(x, i1)
        i2 = self.getrandsecondbin(i1, x, openbins)
        # Pick two points to swap after
        binitems2 = self.finditemsinbin(x, i2)
        j1 = random.randrange(1, len(binitems1))
        j2 = random.randrange(1, len(binitems2))
        # Check for violations
        newbin1 = binitems1[:j1] + binitems2[j2:]
        newbin2 = binitems2[:j2] + binitems1[j1:]
        violat1 = constraints.bincheck(newbin1, self.bpp)
        violat2 = constraints.bincheck(newbin2, self.bpp)
        if violat1 or violat2 is True:
            self.splitbin(x, y)
        # If no violations, swap items:
        else:
            for index in newbin1[j1:]:
                x[i1, index - 1] = 1
                x[i2, index - 1] = 0
            for index in newbin2[j2:]:
                x[i2, index - 1] = 1
                x[i1, index - 1] = 0
        return x, y

    def mergebins(self, x, y):
        # This function merges together the two least filled bins
        i1, i2 = self.gettwominbins(x, y)
        old1 = self.finditemsinbin(x, i1)
        old2 = self.finditemsinbin(x, i2)
        newbin = old1 + old2
        violation = constraints.bincheck(newbin, self.bpp)
        if violation is True:
            self.splitbin(x, y)
        else:
            # Add items in bin2 to bin1
            for index in old2:
                x[i1, index - 1] = 1
            # Move up other bins (overwriting bin2)
            for i in range(i2, self.n - 1):
                y[i] = y[i + 1]
                for j in range(self.n):
                    x[i, j] = x[i + 1, j]
            # Very last bin will now be empty
            y[-1] = 0
            x[-1, :] = 0
        return x, y

    def splitbin(self, x, y):
        # This function splits either a random or the fullest bin into two
        openbins = self.findopenbins(y)
        # Get bin number to split
        ransplit = random.random()
        if ransplit < 0.5:
            i1 = random.choice(openbins)
        else:
            i1 = self.getmaxbin(x)
        # Get items in bin, check to make sure bin has more than one item
        binitems = self.finditemsinbin(x, i1)
        while len(binitems) == 1:
            i1 = random.choice(openbins)
            binitems = self.finditemsinbin(x, i1)
        # Get random place to split bin
        jsplit = random.randrange(1, len(binitems))
        newbin = list(binitems[jsplit:])
        i2 = openbins[-1] + 1
        for index in newbin:
            x[i1, index - 1] = 0
            x[i2, index - 1] = 1
        y[i2] = 1
        return x, y
        
    def findopenbins(self, y):
        # This function gathers the open bin indices into a list.
        # "open" is indicated if y[i] = 1
        openbins = []
        for i in range(self.n):
            if y[i] == 1:
                openbins.append(i)
        return openbins

    def getmaxminranbin(self, x, bins):
        # This function randomly returns one bin number from the list "bins"
        #   max: the fullest bin
        #   min: the emptiest bin
        #   ran: a random bin
        rannum = random.random()
        if rannum < (1/3):
            bini = self.getmaxbin(x)
        elif rannum < (2/3):
            bini = self.getminbin(x, bins)
        else:
            bini = random.choice(bins)
        return bini

    def getmaxbin(self, x):
        # This function returns the fullest bin index.
        binheights = np.dot(x, self.moop.getheightmatrix())
        bini = np.argmax(binheights)
        return bini

    def getminbin(self, x, openbins):
        # This function returns the emptiest bin index.
        binheights = np.dot(x, self.moop.getheightmatrix())
        lastopenbin = openbins[-1]
        bini = np.argmin(binheights[:lastopenbin])
        return bini

    def gettwominbins(self, x, y):
        # This function returns the indices of the two emptiest bins.
        openbins = self.findopenbins(y)
        lastopenbin = openbins[-1]
        fill_level = np.zeros(lastopenbin)
        for i in range(lastopenbin):
            fill_level[i] = len(self.finditemsinbin(x, i))
        binindices = fill_level.argsort()[:2]
        i1 = binindices[0]
        i2 = binindices[1]
        return i1, i2

    def getrandsecondbin(self, i1, x, openbins):
        # This function returns a second random bin that is not
        # bin i1.
        i2 = random.choice(openbins)
        binitems2 = self.finditemsinbin(x, i2)
        while i2 == i1 or len(binitems2) == 1:
            i2 = random.choice(openbins)
            binitems2 = self.finditemsinbin(x, i2)
        return i2
        
    def finditemsinbin(self, x, bini):
        # This function transforms the data in the x-matrix into the variable
        # length representation in "bin".
        #   bini: index of bin
        #   vlrepi: list containing the item index in bin, where index = j + 1
        vlrepi = []
        for j in range(self.n):
            if x[bini, j] == 1:
                vlrepi.append(j + 1)
        return vlrepi

    def updatechrom(self, x):
        # This function converts a given x-matrix into a chromosome
        # representation
        newchrom = []
        for i in range(self.n):
            for j in range(self.n):
                if x[i, j] == 1:
                    newchrom.append(j+1)
        return newchrom

    def updatefe(self):
        # This function keeps track of the number of function evaluations.
        self.funkeval += 1

    def endgen(self):
        # Complete if number of function evaluations > end
        return self.funkeval > self.end

    def updateid(self):
        self.idnum += 1

    def getid(self):
        return self.idnum

    def gett(self):
        return self.t

    def getp(self):
        return self.p

    def getq(self):
        return self.q

    def getarch(self):
        return self.archive


def fnds(setp):
    # This module performs the fast-non-dominated-sort described in Deb(2002).
    # To run this module, enable the following line:
    # from mop import dom
    numsol = len(setp)
    fronts = []
    sp = []
    fhold = []
    nps = []
    for p in range(numsol):
        shold = []
        nump = 0
        for q in range(numsol):
            if setp[p] != setp[q]:
                if mop.dom(setp[p].getfits(), setp[q].getfits()):
                    shold.append(setp[q])
                if mop.dom(setp[q].getfits(), setp[p].getfits()):
                    nump += 1
        sp.append(shold)
        nps.append(nump)
        if nump == 0:
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


def cco(i, p, j, q):
    # This module guides the selection process using the crowded distance
    # assignment.
    prank = p.getrank()
    qrank = q.getrank()
    if prank < qrank:
        best = i
    elif prank > qrank:
        best = j
    else:
        pid = p.getcd()
        qid = q.getcd()
        if pid > qid:
            best = i
        else:
            best = j
    return best


def cda(front, nobj):
    # This module performs the calculation of the crowding distance
    # assignment.
    front = rmdupl(front)
    l = len(front)
    fdist = []
    indexes = []
    for i in range(l):
        fdist.append(0)
        indexes.append([i, front[i].getbins(), front[i].getmaxh(), front[i].getavgw()])
    for o in range(nobj):
        indexes.sort(key=lambda pair: pair[o+1])
        fdist[indexes[0][0]] = 1000000  # represent infinity
        fdist[indexes[l-1][0]] = 1000000
        for i in range(1, l-1):
            fdist[indexes[i][0]] += (indexes[i+1][o+1]-indexes[i-1][o+1]) / \
                                    (indexes[l-1][o+1]-indexes[0][o+1])
    for p in range(l):
        idist = fdist[p]
        front[p].updatecd(idist)
    return front


def rmdupl(front):
    # This module takes a front produced by sorting a generation and
    # removes any duplicate individuals.
    r = 0
    for p in range(len(front)):  # remove duplicates
        if front.count(front[r]) > 1:
            del front[r]
            r -= 1
        r += 1
    return front


def approx(archive):
    # This module finds the final approximate set.
    numsol = len(archive)
    ndset = []
    for p in range(numsol):
        nump = 0    # number of sol. that dominate p
        for q in range(numsol):
            if archive[p] != archive[q]:
                if mop.dom(archive[q].getfits(), archive[p].getfits()):
                    nump += 1
        if nump == 0:
            ndset.append(archive[p])
    ndset = rmdupl(ndset)
    return ndset


def see(ndset, folder):
    # see prints out a file to show how x and y for a gene
    # To run this module, enable the following line:
    # import numpy as np
    pathname = folder + 'variables/'
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    for m in ndset:
        x, y = m.getx(), m.gety()
        solid = str(m.getindex())
        np.savetxt(pathname + solid + '_x.txt', x, fmt='%i', header='Item Location Matrix x:')
        np.savetxt(pathname + solid + '_y.txt', y, fmt='%i', header='Bins Used Matrix y:')


def main():
    # n = eval(input('Please enter the number of items to be sorted: \n'))
    # folder = input('Please enter the name of the folder where your input file is: \n')
    # file = input('Please enter the name of the input file: \n')
    # moma(n, folder, file)
    moma(500, '/Users/gelliebeenz/Documents/Python/ObjectiveMethod/Static/MOMA/SBSBPP500/Experiment04/',
         'SBSBPP500_run4.txt')


if __name__ == '__main__':
    main()
