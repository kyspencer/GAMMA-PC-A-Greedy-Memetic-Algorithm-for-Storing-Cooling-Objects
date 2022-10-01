# momad.py
#   This files runs the multi-objective memetic algorithm based on
#   decomposition (MOMAD).
#   Author: Kristina Yancey Spencer
#   Date: June 1, 2016
#   i - bin index
#   j - item index
#   k - subproblem index
#   m - solution index
#   t - generation index
#   w - subproblem summing index

from __future__ import print_function
import binpacking as bp
from gammapc import ga, outformat as outf
import mop
import numpy as np
import os
import random
import solutions as sol
from datetime import datetime
from glob import glob
from items import makeitems
from operator import attrgetter
import constraints


def momad(n, folder, file):
    existing_files = glob(folder + '*.out')
    filename = folder + 'run%d.out' % (len(existing_files) + 1)
    data = folder + file

    # Initialize Algorithm
    end = 25000  # function evaluations
    method = 'MOMAD'
    outf.startout(filename, n, end, data, method)
    startt = datetime.now()
    print('                         ', startt)
    print('*******************************************************************************')
    print('     Method: MOMAD \n')
    print('     data: ', file)
    print('*******************************************************************************')
    if n <= 600:
        nso = n
    else:
        nso = 600  # number of single-objective functions
    binc, binh, items = makeitems(data)
    bpp = bp.BPP(n, binc, binh, items)
    moop = MOproblem(nso, items)
    gen = MOMADgen(n, bpp, moop, items, end)
    gen.initialize()

    # MOMAD
    while not gen.endgen():
        gen.rungen()
        fronts = []
        fronts.append(gen.getpe())
        fronts.append([])
        outf.genout(filename, gen.gett(), nso, gen.getpl(), fronts)

    see(gen.pe, folder)
    import csv
    with open(folder + 'ApproximationSet.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, dialect='excel', delimiter=',')
        csvwriter.writerow(['Solution ID', 'f[1]', 'f[2]', 'f[3]'])
        for m in gen.pe:
            csvwriter.writerow([m.getindex(), m.getbins(), m.getmaxh(), m.getavgw()])
    outf.endout(filename)
    print('This algorithm has completed successfully.')


class MOproblem:
    # This class maintains the overall multi-objective problem.
    def __init__(self, nso, items):
        self.nso = nso      # number of single-objective subproblems
        self.nobj = 3       # number of objective functions
        self.items = items
        self.lambdas = []
        self.weights = np.zeros((len(items), 1))
        self.heights = np.zeros((len(items), 1))
        self.makematrix()   # make N_so subproblems

    def makematrix(self):
        # This function selects N weight vectors and forms the
        # N_so subproblems. It also initializes the weight and
        # height matrices.
        random.seed(68)
        for k in range(self.nso):
            high = 1
            lambdak = np.zeros(self.nobj)
            for w in range(self.nobj - 1):
                lambdak[w] = random.uniform(0, high)
                high -= lambdak[w]
            lambdak[self.nobj-1] = 1 - np.sum(lambdak)
            self.lambdas.append(lambdak)
        for j in range(len(self.items)):
            self.weights[j, 0] = self.items[j].getweight()
            self.heights[j, 0] = self.items[j].getheight()

    def subfit(self, k, chrom, bpp, method):
        # This function calculates the fitness value for subprob. k
        # Options: 'ed', 'll', 'dp', 'combo'
        if method == 'ed':
            x, y = bp.ed(0, chrom, bpp, self.items)
        else:
            x, y = bp.ll(chrom, bpp.getwbin(), bpp.getub(), bpp.getlb(), self.items)
        subfit = np.dot(self.lambdas[k], self.calcfits(x, y))
        return x, y, subfit

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
        return fitvals


class MOMADgen:
    def __init__(self, n, bpp, moop, items, end):
        self.n = int(n)
        self.t = 0
        self.pl = []      # current solutions for subproblems
        self.pp = []      # solutions to undergo PLS
        self.pe = []      # external pop. of nondominated solutions
        self.end = int(end)
        self.bpp = bpp
        self.moop = moop
        self.idnum = 0
        self.items = items
        self.funkeval = 0
        self.chrom = []
        self.makechroms()

    def rungen(self):
        self.t += 1
        print('t = ', self.t)
        print('Step 2: Pareto Local Search')
        maxmum = 10
        self.pls(maxmum)
        self.pp = []
        print('Step 3: Perturb, Local Search, and Update Populations')
        for k in range(self.moop.nso):
            print('     Perturbing P_L member', k)
            xksquiggly = self.perturb(self.pl[k])
            yk = self.localsh(k, xksquiggly)
            self.update1(yk)
            self.pe = self.update2(yk, self.pe)
            if yk in self.pe:
                self.pp = self.update2(yk, self.pp)
        print(self.funkeval, 'function evaluations have been performed.\n')

    def updatefe(self):
        # This function keeps track of the number of function evaluations.
        self.funkeval += 1

    def endgen(self):
        # Step 1 of MOMAD
        return self.funkeval > self.end

    def makechroms(self):
        for j in range(self.n):
            self.chrom.append(self.items[j].getindex())

    def initialize(self):
        # This function is Step 0 in MOMAD. It initializes P_L, P_P, and P_E.
        print('Initialization starting...')
        # Step 0.1: Initialization of P_L
        for k in range(self.moop.nso):
            print('     Searching for P_L member', k)
            xkso = self.soga(k, 5, 5)
            fit = self.moop.calcfits(xkso.getx(), xkso.gety())
            xk = sol.MultiSol(self.idnum, xkso.getgenes(), xkso.getx(), xkso.gety(), 0, fit, 0, 0.0)
            self.updateid()
            self.pl.append(xk)
        print('Step 0.1: Initialize P_L: complete')
        # Step 0.2: Initialization of  P_p and P_e
        self.pl, fronts = mop.fnds(self.pl)
        self.pp = fronts[0]
        self.pe = self.pp
        print('Step 0.2: Initalize P_P and P_E: complete')
        print(self.funkeval, 'function evaluations were performed in Step 0.\n')

    def pls(self, maxmum):
        # This function performs the Pareto Local Search from Ke(2014).
        # Step 2 in MOMAD.
        # input: pp, pl, pe, maxmum
        # output: pl, pe
        m = 0
        while self.pp != [] and m < maxmum:
            palpha = []
            for x in range(len(self.pp)):
                print('     Searching near solution', self.pp[x].getindex())
                nofx = self.neighborhood(self.pp[x], 50)
                for y in range(len(nofx)):
                    self.update1(nofx[y])
                    if mop.dom(nofx[y].getfits(), self.pp[x].getfits()):
                        self.pe = self.update2(nofx[y], self.pe)
                        if nofx[y] in self.pe:
                            palpha = self.update2(nofx[y], palpha)
            self.pp = list(palpha)
            m += 1

    def perturb(self, xk):
        # This function performs the pertubation in Step 3.1 of MOMAD.
        alpha = random.random()
        if alpha == 0:
            alpha = random.random()
        blocklen = int(alpha * self.n)
        splitj = self.n - blocklen
        cut1 = xk.getgenes()[:splitj]
        cut2 = xk.getgenes()[splitj:]
        cut3 = random.sample(cut2, len(cut2))
        newgene = cut1 + cut3
        x, y = bp.ed(self.idnum, newgene, self.bpp, self.items)
        fit = self.moop.calcfits(x, y)
        xksquiggly = sol.MultiSol(self.idnum, newgene, x, y, self.t, fit, 0, 0.0)
        self.updateid()
        self.updatefe()
        return xksquiggly

    def localsh(self, k, xk):
        # This function performs the local search in Step 3.2 of MOMAD.
        # Uses reduced variable neighborhood search
        # kmax = 1
        m = 0
        while m < 25:
            neighbor = self.neighborhood(xk, 1)
            yk = neighbor[0]
            x1, y1, sofit1 = self.moop.subfit(k, yk.getgenes(), self.bpp, 'ed')
            x2, y2, sofit2 = self.moop.subfit(k, xk.getgenes(), self.bpp, 'ed')
            self.updatefe()
            if sofit1 < sofit2:
                xk = yk
            m += 1
        return xk

    def update1(self, y):
        # This function performs the Update1 algorithm from Ke(2014).
        # input: pl, y
        # output: pl
        replacement = False
        palpha = list(self.pl)
        while palpha != [] and replacement is False:
            x = random.choice(palpha)
            k = self.pl.index(x)
            palpha.remove(x)
            x1, y1, sofit1 = self.moop.subfit(k, y.getgenes(), self.bpp, 'ed')
            x2, y2, sofit2 = self.moop.subfit(k, x.getgenes(), self.bpp, 'ed')
            if sofit1 < sofit2:
                self.pl[k] = y
                replacement = True
            self.updatefe()

    def update2(self, y, setp):
        # This function performs the Update2 algorithm from Ke(2014).
        # input: setp, y
        # output: setp
        palpha = list(setp)
        palpha.append(y)
        palpha, fronts = mop.fnds(palpha)
        setp = fronts[0]  # Nondominated front for palpha is now setp
        # The last step effectively removes all solutions in setp that
        # are dominated by y and adds y to setp if nondominated
        return setp

    def soga(self, k, mems, numgen):
        # Initialize single-objective solutions using genetic algorithm
        q = []
        newgenes = []
        for t in range(numgen):
            print('         Subgeneration', t)
            p, newgenes = self.makesop(t, mems, q, newgenes)
            if t != 0:
                q = ga.binsel(p, mems, 'elitism')
                newgenes = ga.xover(q, mems, 0.9)
                newgenes = ga.mutat(self.n, newgenes, mems)
            q = self.makesoq(t, mems, k, p, q, newgenes)
        q = sorank(q)
        return q[0]

    def makesop(self, t, mems, q, newgenes):
        # Part of soga(), makes P
        if t == 0:
            p = []
            for m in range(mems):
                a = random.sample(self.chrom, self.n)
                newgenes.append(a)
        else:
            p = q
        return p, newgenes

    def makesoq(self, t, mems, k, p, q, newgenes):
        # Part of soga(), makes Q
        if t == 0:
            new = range(mems)
        else:
            new, q = sol.oldnew(p, q, newgenes)
        for m in new:
            x, y, fit = self.moop.subfit(k, newgenes[m], self.bpp, 'll')
            constraints.concheck(m, x, self.bpp)
            q.append(sol.Sol(m, newgenes[m], x, y, 0, fit, 0))
            self.updatefe()
        q = sorank(q)
        return q

    def neighborhood(self, m, hoodsize):
        # This function takes solution m and finds all of its neighbors.
        # input: m
        # output: the set of m's neighbors
        # Limiting neighborhood to 100 neighbors to reduce computationload
        neighbors = []
        while len(neighbors) < hoodsize:
            newgene = m.getgenes()
            a = random.randrange(0, len(newgene))
            b = random.randrange(0, len(newgene))
            newgene[a], newgene[b] = newgene[b], newgene[a]
            x, y = bp.ed(self.idnum, newgene, self.bpp, self.items)
            fit = self.moop.calcfits(x, y)
            a = sol.MultiSol(self.idnum, newgene, x, y, self.t, fit, 0, 0.0)
            neighbors.append(a)
            self.updateid()
            self.updatefe()
        return neighbors

    def updateid(self):
        self.idnum += 1

    def gett(self):
        return self.t

    def getpl(self):
        return self.pl

    def getpe(self):
        return self.pe


def sorank(setp):
    # Fast sorting to obtain single-objective ranks
    numsol = len(setp)
    setp.sort(key=attrgetter('fits'))
    prank = 1
    for m in range(numsol-1):
        setp[m].updaterank(prank)
        if setp[m].getfits() != setp[m+1].getfits():
            prank += 1
    setp[-1].updaterank(prank)
    return setp


def see(ndset, folder):
    # see prints out a file to show how x and y for a gene
    # To run this module, enable the following line:
    # import numpy as np
    dir = folder + 'variables/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    for m in ndset:
        x, y = m.getx(), m.gety()
        solid = str(m.getindex())
        np.savetxt(dir + solid + '_x.txt', x, fmt='%i', header='Item Location Matrix x:')
        np.savetxt(dir + solid + '_y.txt', y, fmt='%i', header='Bins Used Matrix y:')


def main():
    n = eval(input('Please enter the number of items to be sorted: \n'))
    folder = input('Please enter the name of the folder where your input file is: \n')
    file = input('Please enter the name of the input file: \n')
    momad(n, folder, file)

if __name__ == '__main__':
    main()