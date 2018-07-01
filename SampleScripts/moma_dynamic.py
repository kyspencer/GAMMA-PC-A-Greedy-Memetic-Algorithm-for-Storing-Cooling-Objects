# moma.py
#   This python script automates MOMA as described in Ishibuchi (2009).
#   Author: Kristina Yancey Spencer
#    i - bin index
#    j - item index
#    k - algorithm index
#    m - solution index
#    g - generation index

from __future__ import print_function
import random
import coolcookies
import csv
import ga
import h5py
import mooproblem as mop
import numpy as np
import outformat as outf
import solutions_dynamic as sols
import sys
from binpacking_dynamic import BPP, coordarrays
from copy import deepcopy
from datetime import datetime
from glob import glob
from operator import attrgetter
from os import mkdir, path


def moma(n, folder, datafile):
    existing_files = glob(folder + '*.out')
    filename = folder + 'run%d.out' % (len(existing_files) + 1)
    data = folder + datafile

    # Initialize algorithm
    batchsize = 100
    boxcap = 24
    rackcap = 300
    fillcap = 8
    pop = 100           # members per gen.
    end = 25000         # function evaluations
    # batchsize = 6
    # boxcap = 8
    # rackcap = 15
    # fillcap = 2
    # pop = 50            # members per gen.
    # end = 750           # function evaluations
    outf.startout(filename, n, end, data, 'MOMA')
    startt = datetime.now()
    print('                         ', startt)
    print('*******************************************************************************')
    print('     Method: MOMA\n')
    print('     data: ', datafile)
    print('*******************************************************************************')
    cookies = coolcookies.makeobjects(n, batchsize, data)
    moop = mop.MOCookieProblem(n, boxcap, rackcap, fillcap, cookies)
    bpp = BPP(n, boxcap, rackcap, cookies)
    gen = Generation(n, pop, end, cookies, bpp, moop)
    gen.initialp(folder + 'seed.txt')
    gen.makeq()

    # NSGA-II - Local search
    while not gen.endgen():
        gen.rungen()
        outf.genout(filename, gen.gett(), pop, gen.getq(), gen.getfnts())

    # Make output
    ndset = gen.finalapproxset()
    savefitvals(ndset, folder)
    savexys(ndset, folder)
    see(ndset, folder)
    outf.endout(filename)
    print('This algorithm has completed successfully.')


class Generation:
    def __init__(self, n, popsize, end, cookies, bpp, moop):
        self.n = int(n)             # Number of cookies to be sorted
        self.pop = int(popsize)     # Size of a generation population
        self.end = int(end)         # Number of function evaluations to perform
        self.items = cookies        # Dictionary of objects to be sorted
        self.bpp = bpp
        self.moop = moop
        self.idnum = 0              # Solution id number counter
        self.g = 0                  # Generation number counter
        self.p = []
        self.q = []
        self.fronts = []
        self.archive = {}           # Solution Archive
        self.newgenes = []          # List of new genes
        self.funkeval = 0           # Function evaluation counter

    def rungen(self):
        self.g += 1
        # Print out generation info:
        if self.g % 50 == 0:
            gent = datetime.now()
            print('                         ', gent)
        print(' ')
        print('gen = ', self.g)
        # Create the parent pool:
        self.makep()
        if self.g == 1:
            self.q = ga.binsel(self.p, self.pop, 'elitism')
        else:
            self.q = ga.binsel(self.p, self.pop, 'cco')
        # Use the mixed-variable specific modules:
        self.q, self.newgenes = ga.xover_mv(self.q, self.pop, 0.8)
        self.q, self.newgenes = ga.mutat_mv(self.q, self.newgenes, self.pop, 0.3)
        self.makeq()
        if self.g % 10 == 0:
            self.localsearch()
        print(self.funkeval, 'function evaluations have been performed.\n')

    def initialp(self, seedfile):
        # This module creates an initial generation randomly
        random.seed(self.getseedvalue(seedfile))
        chromosomes = range(self.n)
        # Generate popsize random combinations of the chromosomes
        for i in range(self.pop):
            x = random.sample(chromosomes, len(chromosomes))    # chromosome representation
            tfill = self.initialtfill()                         # time variable
            self.newgenes.append((x, tfill))

    def makep(self):
        print('Selecting parent population...')
        r = self.p + self.q
        r, self.fronts = fnds(r)
        print('There are currently {0} solutions in the Approximate Set.'.
              format(len(self.fronts[0])))
        if self.g == 1:
            self.p = r
        else:
            self.fill()

    def fill(self):
        # This module fills the parent population based on the crowded
        # comparison operator.
        self.p = []
        k = 0
        self.fronts[k] = cda(self.fronts[k], 3)
        while (len(self.p) + len(self.fronts[k])) < self.pop:
            self.p = self.p + self.fronts[k]
            k += 1
            if self.fronts[k] == []:
                self.findmoresols(k)
            self.fronts[k] = cda(self.fronts[k], 3)
        self.fronts[k].sort(key=attrgetter('cd'), reverse=True)
        fillsize = self.pop - len(self.p)
        for l in range(fillsize):
            self.p.append(self.fronts[k][l])

    def findmoresols(self, k):
        # This module locates more solutions if too many were removed
        # during truncation procedures
        needed = self.pop - len(self.p)
        for n in range(needed):
            # Pick random solution in first two fronts
            if n <= len(self.fronts[0]):
                m = random.choice(self.fronts[0])
            else:
                m = random.choice(self.fronts[1])
            # Find a nondominated solution neighbor
            nls = 5
            mneighbors = self.paretols(m, 5, retrieve=True)
            while mneighbors == []:
                nls += 5
                mneighbors = self.paretols(m, nls, retrieve=True)
            self.fronts[k].append(random.choice(mneighbors))

    def makeq(self):
        if self.g != 0:
            self.newgenes, self.q = sols.oldnew(self.archive, self.q,
                                                self.newgenes)
        print('Found ', len(self.newgenes), 'new solutions.')
        # Make new solutions in q
        for m in range(len(self.newgenes)):
            self.addnewsol(self.newgenes[m])
        self.cleanarchive()

    def addnewsol(self, newgenes):
        # This function creates a new solution from an individual newgenes.
        # Step 1: Make new solution
        newsol = sols.MultiSol(self.idnum, newgenes[0], newgenes[1], self.bpp)
        # Step 2: Check & Fix feasibility of solution
        newsol = self.moop.calcfeasibility(newsol)
        checkformismatch(newsol)
        # Step 3: Calculate fitness values of solution
        fits = self.moop.calcfits(newsol)
        newsol.updatefitvals(fits)
        self.updateid()
        self.updatefe()
        self.q.append(newsol)
        self.archive[newsol.index] = newsol

    def cleanarchive(self):
        # This function removes unwanted solutions from the archive
        # and calls the local search function.
        self.reduce(1000)
        aslimit = 100    # Limit to size of approximation set
        # Every 50 generations check the archive
        if self.g % 50 == 0 or (self.end - self.funkeval) < 500:
            approxset = [m for k, m in self.archive.items() if m.getrank() == 1]
            keys = [k for k, m in self.archive.items() if m.getrank() == 1]
            # Truncate approximate set
            if len(approxset) > aslimit:
                self.fittruncation(keys, approxset, aslimit)

    def reduce(self, klim):
        # This module keeps the length of the archive below 1000 individual
        # solutions to save computer memory during runtime.
        #   - archive is the list of all solutions
        if len(self.archive) > round(klim * 1.2):
            while len(self.archive) > klim:
                rankdel = max(self.archive.values(), key=attrgetter('rank')).rank
                # Any solution with a rank greater than or equal to 3, easy remove
                if rankdel >= 3:
                    self.archive = {k: v for k, v in self.archive.items()
                                    if v.rank != rankdel}
                # Solutions in front 0 - 2 also need to consider crowding distance
                else:
                    worstfront = {k: v for k, v in self.archive.items()
                                  if v.rank == rankdel}
                    cddel = min(worstfront.values(), key=attrgetter('cd')).cd
                    delete = {k: v for k, v in worstfront.items()
                              if v.cd == cddel}
                    for key, solution in delete.items():
                        del self.archive[key]

    def fittruncation(self, keys, approxfront, limit):
        # This function removes solutions from the approximate set that have
        # the same fitness values.
        print('Removing superfluous solutions from the archive.')
        nremove = len(approxfront) - limit          # Number of solutions to remove
        clusters = self.getclusters(approxfront)
        nrm = 0
        # Remove one solution from each cluster before repeating
        for k in range(len(clusters[0])):
            if nrm == nremove:
                break
            cdremove = clusters[0][0].getcd()
            for c in range(len(clusters)):
                if nrm == nremove:
                    break
                if len(clusters[c]) > 1:
                    if clusters[c][0].getcd() <= cdremove:
                        if clusters[c][0] not in self.p or self.q:
                            keyval = approxfront.index(clusters[c][0])
                            del self.archive[keys[keyval]]
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
        print('Local Search...')
        self.fronts[0].sort(key=attrgetter('cd'), reverse=True)
        # Set probability for progress in calculations
        if self.funkeval > 0.9 * self.end:
            probls = 0.2
        else:
            probls = 0.2 * float(self.funkeval) / (0.9 * self.end)
        nls = 10        # Number of neighbors to search
        for m in range(self.pop):
            ranls = random.random()
            if ranls <= probls:
                if m < len(self.fronts[0]):
                    self.paretols(self.fronts[0][m], nls)
                else:
                    self.paretols(self.p[m], nls)

    def paretols(self, m, nls, retrieve=False):
        # This function finds up to nls neighbors to check if they dominate
        # solution m of the new generation.
        neighbors = []
        for k in range(nls):
            copy = deepcopy(m)
            copy = self.binmutation(copy)
            # Fix matrices
            copy.updateid(100000)
            copy = self.moop.calcfeasibility(copy)
            checkformismatch(copy)
            newfit = self.moop.calcfits(copy)
            self.updatefe()
            # If we find a neighbor that is nondominated by solution m, add to q.
            if not mop.dom2(m.getfits(), newfit):
                print(' Local search found another possible solution:', self.idnum)
                copy.updatefitvals(newfit)
                copy.updateid(self.idnum)
                self.updateid()
                self.q.append(copy)
                self.archive[copy.index] = copy
                neighbors.append(copy)
        if retrieve is True:
            return neighbors

    def binmutation(self, solution):
        # This function performs bin-specific mutations
        prob = 1
        ranmute = random.random()
        if ranmute < prob:
            ranm1 = random.random()
            if ranm1 < 0.25:
                solution = self.partswap(solution)
            elif ranm1 < 0.50:
                solution = self.mergebins(solution)
            elif ranm1 < 0.75:
                solution = self.splitbin(solution)
            else:
                solution = self.itemswap(solution)
        return solution

    def itemswap(self, solution):
        # This function performs a 2-item swap:
        #   with an item from the fullest bin
        #   with an item from the emptiest bin
        #   with an item from a random bin
        y = solution.gety()
        vlrep = solution.getvlrep()
        openbins = self.findopenbins(y)
        # Select random bins
        i1 = self.getmaxminranbin(vlrep)
        i2 = self.getrandsecondbin(i1, vlrep, solution.gettfill(), openbins)
        try:
            j1, j2 = self.getitemstoswap(solution, i1, i2)
            solution.swapitems(i1, j1, i2, j2)
        except:
            binitems1 = vlrep[i1]
            if len(binitems1) != 1:
                j1 = random.choice(binitems1)
                tnew = self.items.get(j1).getbatch() * 600 + 150
                solution.opennewbin(i1, j1, tnew)
        return solution

    def getitemstoswap(self, solution, i1, i2):
        # This function returns two items in bins i1 and i2 that
        # can be feasibly swapped.
        tfill = solution.gettfill()
        vlrep = solution.getvlrep()
        # Find items in bin i1 and bin i2
        binitems1 = vlrep[i1]
        binitems2 = vlrep[i2]
        # Find item in bin i1 that can move to i2
        for count1 in range(len(binitems1)):
            j1 = random.choice(binitems1)
            # Determine if can be swapped
            bool1 = self.moop.cookiedonebaking(j1, tfill[i2])
            if bool1 is True:
                for count2 in range(len(binitems2)):
                    j2 = random.choice(binitems2)
                    # Determine if can be swapped
                    bool2 = self.moop.cookiedonebaking(j2, tfill[i1])
                    # Stop module and return values if found items to swap
                    if bool2 is True:
                        return j1, j2
        return False

    def partswap(self, solution):
        # This function swaps parts of two bins with each other
        y = solution.gety()
        tfill = solution.gettfill()
        vlrep = solution.getvlrep()
        openbins = self.findopenbins(y)
        # Pick two random bins
        i1 = self.getrandombin(vlrep)
        i2 = self.getrandsecondbin(i1, vlrep, tfill, openbins)
        binitems1 = vlrep[i1]
        binitems2 = vlrep[i2]
        # Perform the swap, or if fail, move to splitting a bin
        try:
            p1, p2 = self.getpointtoswap(binitems1, tfill[i1], binitems2, tfill[i2])
            movetobin2 = list(binitems1[p1:])
            movetobin1 = list(binitems2[p2:])
            for j in movetobin2:
                solution.moveitem(i1, j, i2)
            for j in movetobin1:
                solution.moveitem(i2, j, i1)
        except:
            self.splitbin(solution)
        return solution

    def getpointtoswap(self, binitems1, t1, binitems2, t2):
        # This function returns two points to perform the swap on
        # Retrieve boolean lists
        bool1 = self.moop.packatt(binitems1, t2)
        bool2 = self.moop.packatt(binitems2, t1)
        # Find starting point for bin 1:
        start1 = self.findstartforswap(bool1)
        if start1 == len(bool1):
            return False
        else:
            p1 = random.randrange(start1, len(binitems1))
        # Find starting point for bin 2:
        start2 = self.findstartforswap(bool2)
        if start2 == len(bool2):
            return False
        else:
            p2 = random.randrange(start2, len(binitems2))
        # Check for capacity violations
        newbin1 = binitems1[:p1] + binitems2[p2:]
        newbin2 = binitems2[:p2] + binitems1[p1:]
        if len(newbin1) >= self.moop.boxcap or len(newbin2) >= self.moop.boxcap:
            return False
        else:
            return p1, p2

    def findstartforswap(self, boollist):
        # This function returns the index after which all values are True
        start = 1
        for k in range(len(boollist) - 1, 0, -1):
            if boollist[k] is False:
                start = k + 1
                return start
        return start

    def mergebins(self, solution):
        # This function merges together the two least filled bins
        vlrep = solution.getvlrep()
        tfill = solution.gettfill()
        try:
            i1, i2 = self.gettwominbins(vlrep, tfill)
            # Add items in bin2 to bin1
            for j in vlrep[i2]:
                solution.moveitem(i2, j, i1)
            solution.y[i2] = 0
        except:
            self.splitbin(solution)
        return solution

    def splitbin(self, solution):
        # This function splits either a random or the fullest bin into two
        vlrep = solution.getvlrep()
        # Get bin number to split
        ransplit = random.random()
        if ransplit < 0.5:
            i1 = self.getrandombin(vlrep)
        else:
            i1 = self.getmaxbin(vlrep)
        # Get random place to split bin
        binitems = vlrep[i1]
        jsplit = random.randrange(1, len(binitems))
        # Open new bin
        newbin = list(binitems[jsplit:])
        j1 = newbin[-1]
        tnew = self.items.get(j1).getbatch() * 600 + 150
        solution.opennewbin(i1, j1, tnew)
        del newbin[-1]
        if newbin == []:
            return solution
        # Move items to new bin
        i2 = solution.openbins - 1
        for index in newbin:
            solution.moveitem(i1, index, i2)
        return solution
        
    def findopenbins(self, y):
        # This function gathers the open bin indices into a list.
        # "open" is indicated if y[i] = 1
        openbins = []
        for i in range(self.n):
            if y[i] == 1:
                openbins.append(i)
        return openbins

    def getmaxminranbin(self, vlrep):
        # This function randomly returns one bin number from the list "bins"
        #   max: the fullest bin
        #   min: the emptiest bin
        #   ran: a random bin
        rannum = random.random()
        if rannum < (1/3):
            bini = self.getmaxbin(vlrep)
        elif rannum < (2/3):
            bini = self.getminbin(vlrep)
        else:
            bini = self.getrandombin(vlrep)
        return bini

    def getrandombin(self, vlrep):
        # This function returns a random bin with more than one item in it
        bins = range(len(vlrep))
        bini = random.choice(bins)
        while len(vlrep[bini]) <= 1:
            bini = random.choice(bins)
        return bini

    def getmaxbin(self, vlrep):
        # This function returns the index of the fullest bin.
        bincapacity = np.zeros(len(vlrep))
        for i in range(len(vlrep)):
            bincapacity[i] = len(vlrep[i])
        bini = np.argmax(bincapacity)
        return bini

    def getminbin(self, vlrep):
        # This function returns the index of the emptiest bin.
        bincapacity = np.zeros(len(vlrep))
        for i in range(len(vlrep)):
            bincapacity[i] = len(vlrep[i])
        bini = np.argmin(bincapacity)
        return bini

    def gettwominbins(self, vlrep, tfill):
        # This function returns the indices of the two emptiest bins.
        bincapacity = np.zeros(len(vlrep))
        for i in range(len(vlrep)):
            bincapacity[i] = len(vlrep[i])
        binindices = bincapacity.argsort()
        i1 = binindices[0]
        # Find a bin that works time-wise with bin i1
        for bi in range(1, len(vlrep)):
            i2 = binindices[bi]
            bool1 = self.moop.packatt(vlrep[i2], tfill[i1])
            if all(bool1) is True:
                # Check for capacity violations
                newbinlen = bincapacity[0] + bincapacity[bi]
                if newbinlen <= self.moop.boxcap:
                    return i1, i2
                else:
                    return False
        return False

    def getrandsecondbin(self, i1, vlrep, tfill, openbins):
        # This function returns a second random bin that is not
        # bin i1 and that items in bin i1 can be moved to
        i2 = random.choice(openbins)
        boollist = self.moop.packatt(vlrep[i1], tfill[i2])
        while i2 == i1 or len(vlrep[i2]) == 1 or any(boollist) is False:
            i2 = random.choice(openbins)
            boollist = self.moop.packatt(vlrep[i1], tfill[i2])
        return i2

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

    def getseedvalue(self, seedfile):
        # This function retrieves the seed number from seedfile.
        infile = open(seedfile, 'r')
        seed = int(infile.readline())
        infile.close()
        return seed

    def initialtfill(self):
        # This function generates a random tfill matrix.
        tbox0 = random.uniform(700, 1200)
        # Use either random number of bins or x * nbatches
        rannum = random.random()
        if rannum < 0.5:
            fillbins = random.randint(self.bpp.lb, self.n)
        else:
            # Vary x between dozen and two dozen in a box
            expert_highfill = int(self.n / min(24, self.moop.boxcap))
            expert_lowfill = int(self.n / min(12, self.moop.boxcap -
                                              (self.moop.boxcap % 3)))
            xbake = random.randint(expert_highfill, expert_lowfill)
            fillbins = xbake * self.moop.nbatches
        # Find tintervals
        tintervals = (600 * (self.moop.nbatches + 1) - tbox0) / (max(fillbins - 1, 1))
        tfill = np.zeros(self.n, dtype=np.float)
        for i in range(fillbins):
            tfill[i] = round(tintervals * i + tbox0, 1)
        return tfill

    def endgen(self):
        # Complete if number of function evaluations > end
        return self.funkeval > self.end

    def finalapproxset(self):
        # This function finds the final approximate set.
        numsol = len(self.archive)
        ndset = []
        keys = [k for k, v in self.archive.items()]
        for p in range(numsol):
            np = 0
            u = self.archive.get(keys[p])
            for q in range(numsol):
                v = self.archive.get(keys[q])
                if u != v:
                    # If v dominates u, increase domination count
                    if mop.dom1(v.getfits(), u.getfits()):
                        np += 1
            if np == 0:
                ndset.append(u)
        return ndset

    def updateid(self):
        self.idnum += 1

    def getid(self):
        return self.idnum

    def gett(self):
        return self.g

    def getp(self):
        return self.p

    def getq(self):
        return self.q

    def getfnts(self):
        return self.fronts

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
                if mop.dom1(setp[p].getfits(), setp[q].getfits()):
                    shold.append(setp[q])
                if mop.dom1(setp[q].getfits(), setp[p].getfits()):
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
        indexes.append([i, front[i].getbins(), front[i].getavgheat(), front[i].getmaxreadyt()])
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


def savefitvals(ndset, folder):
    # This function saves the fitness values of the Approximation Set to
    # a csv file.
    with open(folder + 'ApproximationSet.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, dialect='excel', delimiter=',')
        csvwriter.writerow(['Solution ID', 'f[1] (B)', 'f[2] (W)', 'f[3] (s)'])
        for m in ndset:
            csvwriter.writerow([m.getid(), m.getbins(), m.getavgheat(),
                                m.getmaxreadyt()])


def savexys(ndset, folder):
    # This function saves the x, y, and tfill matrices into a .h5 file for later
    names = []
    xs = []
    ys = []
    tfills = []
    for m in range(len(ndset)):
        index, x, y = ndset[m].getid(), ndset[m].getx(), ndset[m].gety()
        tfill = ndset[m].gettfill()
        names.append(str(index))
        xs.append(x)
        ys.append(y)
        tfills.append(tfill)
    h5f = h5py.File(folder + 'xymatrices.h5', 'w')
    xgrp = h5f.create_group('xmatrices')
    ygrp = h5f.create_group('yarrays')
    tgrp = h5f.create_group('tfills')
    for m in range(len(names)):
        xgrp.create_dataset(names[m], data=xs[m])
        ygrp.create_dataset(names[m], data=ys[m])
        tgrp.create_dataset(names[m], data=tfills[m])
    h5f.close()


def see(ndset, folder):
    # see prints out a file to show how x and y for a gene
    dirname = folder + 'variables/'
    if not path.exists(dirname):
        mkdir(dirname)
    for m in ndset:
        x, y, tfill = m.getx(), m.gety(), m.gettfill()
        solid = str(m.getid())
        np.savetxt(dirname + solid + '_x.txt', x, fmt='%i', header='Item Location Matrix x:')
        np.savetxt(dirname + solid + '_y.txt', y, fmt='%i', header='Bins Used Matrix y:')
        np.savetxt(dirname + solid + '_tfill.txt', tfill, fmt='%G',
                   header='Time of Bins Used Matrix tfill:')


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


def main():
    # n = eval(input('Please enter the number of items to be sorted: \n'))
    # folder = input('Please enter the name of the folder where your input file is: \n')
    # file = input('Please enter the name of the input file: \n')
    # moma(n, folder, file)
    moma(1000, '/Users/gelliebeenz/Documents/Python/ObjectiveMethod/TimeDependent/'
               'MOMA/Cookies1000/Experiment01/', 'Cookies1000.txt')


if __name__ == '__main__':
    main()
