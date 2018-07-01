# nsgaii.py
#    Program to run NSGA-II on dynamic benchmark problems
#    Author: Kristina Spencer
#    g - generation index
#    i - bin index
#    j - item index
#    k - algorithm index
#    m - solution index
#    t - time index

from __future__ import print_function
import coolcookies
import csv
import ga
import h5py
import mooproblem as mop
import numpy as np
import outformat as outf
import random
import solutions_dynamic as sols
import sys
from binpacking_dynamic import BPP
from datetime import datetime
from glob import glob
from operator import attrgetter
from os import mkdir, path


def nsgaii(n, folder, datafile):
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
    # pop = 50        # members per gen.
    # end = 750       # function evaluations
    outf.startout(filename, n, end, data, 'NSGA-II')
    startt = datetime.now()
    print('                         ', startt)
    print('*******************************************************************************')
    print('     Method: NSGA-II\n')
    print('     data: ', datafile)
    print('*******************************************************************************')
    cookies = coolcookies.makeobjects(n, batchsize, data)
    moop = mop.MOCookieProblem(n, boxcap, rackcap, fillcap, cookies)
    bpp = BPP(n, boxcap, rackcap, cookies)
    gen = Generation(n, pop, end, cookies, bpp, moop)
    gen.initialp(folder + 'seed.txt')
    gen.makeq()

    # NSGA-II
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
        self.newgenes = []
        self.fronts = []
        self.archive = {}
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
        self.q, self.newgenes = ga.xover_mv(self.q, self.pop, 0.9)
        self.q, self.newgenes = ga.mutat_mv(self.q, self.newgenes, self.pop, 0.3)
        self.makeq()
        print(self.funkeval, 'function evaluations have been performed.\n')

    def initialp(self, seedfile):
        # This module creates an initial generation randomly
        random.seed(self.getseedvalue(seedfile))
        chromosomes = range(self.n)
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
            self.p = fill(self.fronts, self.pop)

    def makeq(self):
        if self.g != 0:
            self.newgenes, self.q = sols.oldnew(self.archive, self.q,
                                                self.newgenes)
        print('Found ', len(self.newgenes), 'new solutions.')
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
        # This function keeps the size of the archive below 1000 and
        # the size of the approximation set below 100.
        self.reduce(1000)
        aslimit = 100       # Limit to size of approximation set
        # Every 50 generations check the archive
        if self.g % 50 == 0 or (self.end - self.funkeval) < 500:
            approxset = [m for k, m in self.archive.items() if m.getrank() == 1]
            keys = [k for k, m in self.archive.items() if m.getrank() == 1]
            # Truncate approximate set
            if len(approxset) > aslimit:
                self.fittruncation(keys, approxset, aslimit)

    def reduce(self, k):
        # This module keeps the length of the archive below 1000 individual
        # solutions to save computer memory during runtime.
        #   - archive is the list of all solutions
        if len(self.archive) > round(k * 1.2):
            while len(self.archive) > k:
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

    def endgen(self):
        # Complete if number of function evaluations > end
        if self.g < 3:
            return False
        return self.funkeval > self.end

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

    def updatefe(self):
        # This function keeps track of the number of function evaluations.
        self.funkeval += 1

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
        np = 0
        for q in range(numsol):
            if setp[p] != setp[q]:
                if mop.dom1(setp[p].getfits(), setp[q].getfits()):
                    shold.append(setp[q])
                if mop.dom1(setp[q].getfits(), setp[p].getfits()):
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


def fill(fronts, pop):
    # This module fills the parent population based on the crowded
    # comparison operator.
    p = []
    k = 0
    fronts[k] = cda(fronts[k], 3)
    while (len(p) + len(fronts[k])) <= pop:
        p = p + fronts[k]
        k += 1
        fronts[k] = cda(fronts[k], 3)
    fronts[k].sort(key=attrgetter('cd'), reverse=True)
    fillsize = pop - len(p)
    for l in range(fillsize):
        p.append(fronts[k][l])
    return p


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
        np.savetxt(dirname + solid + '_x.txt', x, fmt='%i',
                   header='Item Location Matrix x:')
        np.savetxt(dirname + solid + '_y.txt', y, fmt='%i',
                   header='Bins Used Matrix y:')
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
    n = eval(input('Please enter the number of items to be sorted: \n'))
    folder = input('Please enter the name of the folder where your input file is: \n')
    datafile = input('Please enter the name of the input file: \n')
    nsgaii(n, folder, datafile)


if __name__ == '__main__':
    main()
