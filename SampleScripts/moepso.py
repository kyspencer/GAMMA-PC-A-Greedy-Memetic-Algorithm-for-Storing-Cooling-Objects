# moepso.py
#   This files runs the multi-objective evolutionary particle swarm
#   optimization (MOEPSO) algorithm.
#   Author: Kristina Yancey Spencer
#   Date: June 15, 2016
#   i - bin index
#   j - item index
#   m - solution index
#   t - generation index

from __future__ import print_function
import binpacking as bp
import mop
import numpy as np
import os
import outformat as outf
import random
import solutions as sol
from datetime import datetime
from glob import glob
from items import makeitems
from operator import attrgetter
import constraints


def moepso(n, folder, inputfile):
    existing_files = glob(folder + '*.out')
    filename = folder + 'run%d.out' % (len(existing_files) + 1)
    data = folder + inputfile

    # Initialize Algorithm
    pop = 500    # swarm members
    end = 25000  # function evaluations
    method = 'MOEPSO'
    outf.startout(filename, n, end, data, method)
    startt = datetime.now()
    print('                         ', startt)
    print('*******************************************************************************')
    print('     Method: MOEPSO \n')
    print('     data: ', inputfile)
    print('*******************************************************************************')
    binc, binh, items = makeitems(data)
    bpp = bp.BPP(n, binc, binh, items)
    moop = mop.MOproblem(pop, items)
    gen = MOEPSOgen(n, pop, end, bpp, moop, items)

    # MOEPSO generations
    while not gen.endgen():
        gen.rungen()
        fronts = [gen.getgbest(), []]
        outf.genout(filename, gen.gett(), pop, gen.getp(), fronts)

    see(gen.gbest, folder, 'variables/')
    import csv
    with open(folder + 'ApproximationSet.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, dialect='excel', delimiter=',')
        csvwriter.writerow(['Solution ID', 'f[1]', 'f[2]', 'f[3]'])
        for m in gen.gbest:
            csvwriter.writerow([m.getindex(), m.getbins(), m.getmaxh(), m.getavgw()])
    outf.endout(filename)
    print('This algorithm has completed successfully.')


class Particle:
    # This class groups together particle information.
    def __init__(self, n, x, y, chromosome):
        self.n = n
        self.x = x
        self.y = y
        self.id = 0      # connects to associated solution id
        self.chrom = chromosome
        self.vlrep = []  # variable length representation of chromosome
        self.transform()

    def transform(self):
        # This function transforms x and y into the variable
        # length representation.
        # The numbers in vlrep correspond to item indices.
        for i in range(self.n):
            if self.y[i] == 1:
                bini = []
                for j in range(self.n):
                    if self.x[i, j] == 1:
                        bini.append(j+1)
                self.vlrep.append(bini)

    def addbin(self, vid, bpp, items):
        # This function inserts the "best bin" chosen by the PSO operator.
        # - inserts "best bin" into x and y
        # - repacks the rest of the bins and makes new vlrep
        if vid not in self.vlrep:
            # move other bins down one in matrix
            for i in range(self.n - 1, 0, -1):
                self.x[i, :] = self.x[i-1, :]
            # align first bin in x, y with best bin
            for j in range(self.n):
                self.x[0, j] = 0
                if (j+1) in vid:
                    self.x[0, j] = 1
            # remove duplicates from list
            for i in range(len(self.vlrep)):
                for itemid in vid:
                    if itemid in self.vlrep[i]:
                        self.x[i+1, itemid-1] = 0
            self.x, self.y = bp.repack(self.x, self.y, 1, self.n, bpp, items)
            self.vlrep = []
            self.transform()

    def partswap(self, bpp):
        # This function picks two bins randomly and swaps part of their contents.
        i1 = random.randrange(len(self.vlrep))
        while len(self.vlrep[i1]) == 1:
            i1 = random.randrange(len(self.vlrep))
        i2 = random.randrange(len(self.vlrep))
        while i2 == i1 or len(self.vlrep[i2]) == 1:
            i2 = random.randrange(len(self.vlrep))
        j1 = random.randrange(1, len(self.vlrep[i1]))
        j2 = random.randrange(1, len(self.vlrep[i2]))
        newbin1 = self.vlrep[i1][:j1] + self.vlrep[i2][j2:]
        newbin2 = self.vlrep[i2][:j2] + self.vlrep[i1][j1:]
        violat1 = constraints.bincheck(newbin1, bpp)
        violat2 = constraints.bincheck(newbin2, bpp)
        if violat1 or violat2 is True:
            self.splitbin()
        else:
            self.vlrep[i1], self.vlrep[i2] = newbin1, newbin2
            for j in self.vlrep[i1][j1:]:
                self.x[i1, j-1] = 1
                self.x[i2, j-1] = 0
            for j in self.vlrep[i2][j2:]:
                self.x[i2, j-1] = 1
                self.x[i1, j-1] = 0

    def mergebins(self, bpp):
        # This function merges the two least filled bins into 1 bin.
        # Note: vlrep has item index in it, which is j + 1
        fill_level = np.zeros(len(self.vlrep))
        for i in range(len(self.vlrep)):
            fill_level[i] = len(self.vlrep[i])
        binindices = fill_level.argsort()[:2]
        i1 = binindices[0]
        i2 = binindices[1]
        newbin = self.vlrep[i1] + self.vlrep[i2]
        violation = constraints.bincheck(newbin, bpp)
        if violation is True:
            rannum = random.random()
            if rannum < 0.5:
                self.partswap(bpp)
            else:
                self.splitbin()
        else:
            for j in self.vlrep[i2]:
                self.x[i1, j-1] = 1
            for i in range(i2, self.n-1):
                self.y[i] = self.y[i+1]
                for j in range(self.n):
                    self.x[i, j] = self.x[i+1, j]
            self.y[-1] = 0
            self.x[-1, :] = 0
            self.vlrep[i1].extend(self.vlrep[i2])
            del self.vlrep[i2]

    def splitbin(self):
        # This function splits a random bin into two separate bins.
        isplit = random.randrange(len(self.vlrep))
        while len(self.vlrep[isplit]) == 1:
            isplit = random.randrange(len(self.vlrep))
        jsplit = random.randrange(1, len(self.vlrep[isplit]))
        newbin = list(self.vlrep[isplit][jsplit:])
        for j in newbin:
            self.x[isplit, j-1] = 0
            self.x[len(self.vlrep), j-1] = 1
        self.y[len(self.vlrep)] = 1
        self.vlrep.append(newbin)
        del self.vlrep[isplit][jsplit:]

    def updatechrom(self):
        newchrom = []
        for i in range(len(self.vlrep)):
            for j in range(len(self.vlrep[i])):
                newchrom.append(self.vlrep[i][j])
        self.chrom = newchrom

    def updateid(self, idnum):
        # This function connects the swarm member to a solution id.
        self.id = idnum

    def getindex(self):
        return self.id

    def getx(self):
        return self.x

    def gety(self):
        return self.y

    def getgenes(self):
        return self.chrom

    def getvlrep(self):
        return self.vlrep


class MOEPSOgen:
    # This class maintains each generation evolution for the MOEPSO.
    # The swarm is made up of the particles, and p is the corresponding solutions
    def __init__(self, n, pop, end, bpp, moop, items):
        self.n = n
        self.t = 0
        self.pop = pop
        self.end = end
        self.bpp = bpp
        self.moop = moop
        self.p = []
        self.swarm = []
        self.pbest = []     # Archive of personal best solutions
        self.gbest = []     # Archive of all non-dominated solutions
        self.gblimit = 50   # Fixed sized of G_best
        self.items = items
        self.idnum = 0
        self.funkeval = 0

    def rungen(self):
        if self.t == 0:
            newgenes = self.initialize()
            for m in range(self.pop):
                x, y = bp.ed(m, newgenes[m], self.bpp, self.items)
                self.swarm.append(Particle(self.n, x, y, newgenes[m]))
        self.t += 1
        print('t = ', self.t)
        self.fitevalarch()
        self.psooperator()
        self.mutation()
        self.heuristic()
        print(self.funkeval, 'function evaluations have been performed.\n')

    def initialize(self):
        # This module initializes MOEPSO based on Liu(2007).
        # input: items, number of swarm members
        # output: initial swarm of pop solutions
        print('Initializing swarm')
        random.seed(83)
        # Initialize chromosomes for swarm
        newgenes = []
        for m in range(self.pop):
            ran1 = random.random()
            if ran1 < 0.5:      # all random sequence
                randomlist = random.sample(self.items, self.n)
                newgenes.append(self.getchroms(randomlist))
            else:               # sorted sequence
                ran2 = random.choice([0, 1])
                if ran2 == 0:
                    sortedlist = sorted(self.items, key=attrgetter('weight'))
                    newgenes.append(self.getchroms(sortedlist))
                else:
                    sortedlist = sorted(self.items, key=attrgetter('height'))
                    newgenes.append(self.getchroms(sortedlist))
                ran3 = random.random()
                if ran3 < 0.5:  # partial random, partial sorted
                    ran4 = random.randrange(0, self.n)
                    cut1 = newgenes[m][:ran4]
                    cut2 = random.sample(newgenes[m][ran4:], (self.n - ran4))
                    newgenes[m] = cut1 + cut2
        # Initialize p
        for m in range(self.pop):
            self.p.append([])
        return newgenes

    def getchroms(self, items):
        # This function produces a list containing only the chromosome IDs
        # from a given list of items, in the order given in the call.
        chromosomes = []
        for j in range(self.n):
            chromosomes.append(items[j].getindex())
        return chromosomes

    def fitevalarch(self):
        # Step 2: evaluation and archiving
        print('Step 2: Evaluation and Archiving')
        if self.t == 1:
            new = range(self.pop)
        else:
            new = self.oldnew()
        # Evaluation of fitness values
        print(' - Evaluation')
        for m in new:
            fit = self.moop.calcfits(self.swarm[m].getx(), self.swarm[m].gety())
            self.updatefe()
            constraints.concheck(self.idnum, self.swarm[m].getx(), self.bpp)
            self.p[m] = sol.PSOSol(self.idnum, self.swarm[m].getgenes(), self.swarm[m].getx(),
                                   self.swarm[m].gety(), self.t, fit, 0, 0)
            self.swarm[m].updateid(self.idnum)
            self.updateid()
        # Archiving: Personal Best Archive
        self.archive()

    def oldnew(self):
        archive = self.p + self.pbest + self.gbest
        genes = []
        for m in range(self.pop):
            genes.append(self.swarm[m].getgenes())
        new = []
        archgenes = []
        for m in range(len(archive)):
            archgenes.append(archive[m].getgenes())
        for p in range(self.pop):
            count = archgenes.count(genes[p])
            if count == 0:
                new.append(p)
        return new

    def archive(self):
        print(' - Updating Pbest')
        self.updatepbest()
        # Archiving: Global Best Archive (Approximate Set)
        print(' - Archiving Global Best')
        pgsolutions = sorted(self.pbest + self.gbest, key=attrgetter('index'))
        pgsolutions, fronts = mop.fnds(pgsolutions)
        self.gbest = fronts[0]
        self.gbest = rmdupl(self.gbest)
        print(' - Updating niche counts')
        self.dynsharing()
        if len(self.gbest) > self.gblimit:
            self.nichetruncation()

    def updatepbest(self):
        # This function updates the population Pbest
        if self.t == 1:
            self.pbest = list(self.p)
            for m in range(self.pop):
                self.pbest[m].makevlrep(self.swarm[m].getvlrep(), self.items)
        else:
            for m in range(self.pop):
                if self.p[m].getindex() != self.pbest[m].getindex():
                    # See if the new sol. m dominates the current personal best
                    if mop.dom(self.p[m].getfits(), self.pbest[m].getfits()):
                        self.pbest[m] = self.p[m]
                        self.pbest[m].makevlrep(self.swarm[m].getvlrep(), self.items)
                        print('     Swarm member', m, 'found new personal best!')
                    else:
                        if not mop.dom(self.pbest[m].getfits(), self.p[m].getfits()):
                            pbran = random.random()
                            if pbran < 0.5:
                                self.pbest[m] = self.p[m]
                                self.pbest[m].makevlrep(self.swarm[m].getvlrep(), self.items)

    def dynsharing(self):
        # This function calculates the dynamic sharing scheme Tan(2003).
        dij = []
        for m in range(len(self.gbest)):
            dij.append(self.calcdist(m))
        dmax = np.amax(dij)
        dmin = np.amin(dij)
        davg = (dmax + dmin) / 2
        sigmashare = (self.pop ** (1 / (1 - self.moop.nobj))) * (davg / 2)
        for m in range(len(self.gbest)):
            sh_dij = np.zeros((len(self.gbest)-1, 1))
            for j in range(len(sh_dij)):
                if dij[m][j] < sigmashare:
                    sh_dij[j, 0] = 1 - (dij[m][j] / sigmashare)
            mi = np.sum(sh_dij)
            self.gbest[m].updateniche(mi)

    def calcdist(self, m):
        # This function calculates the distance between solutions in gbest.
        dij = np.zeros((len(self.gbest) - 1, 1))
        mfitvals = self.gbest[m].getfits()
        for j in range(len(dij)):
            jfitvals = np.zeros(len(mfitvals))
            if j < m:
                jfitvals = self.gbest[j].getfits()
            if j >= m:
                jfitvals = self.gbest[j+1].getfits()
            dij[j, 0] = np.linalg.norm(mfitvals - jfitvals)
        return dij

    def nichetruncation(self):
        # This function performs a recurrent truncation of Gbest based on
        # niche count.
        # Sort by niche count, low to high
        self.gbest.sort(key=attrgetter('niche'))
        # Remove solutions with the highest niche count first
        m = len(self.gbest) - 1
        while len(self.gbest) > self.gblimit:
            # Count how many solutions have the same niche count
            count = 0
            niche = self.gbest[m].getniche()
            for m2 in range(m, 0 , -1):
                if self.gbest[m2].getniche() != niche:
                    break
                count += 1
            # Remove solutions with niche count
            for k in range(count - 1):
                if len(self.gbest) == self.gblimit:
                    break
                del self.gbest[m-1]
                m -= 1
            # Recalculate niche and resort list
            self.dynsharing()
            self.gbest.sort(key=attrgetter('niche'))
        # Reset Gbest to normal sorting order
        self.gbest.sort(key=attrgetter('index'))

    def psooperator(self):
        # Step 3: PSO operator
        print('Step 3: PSO operator')
        for m in range(self.pop):
            pid = self.pbest[m].getpbest()
            pgd = self.getgbestbin()
            vid = list(random.choice([pid, pgd]))
            self.swarm[m].addbin(vid, self.bpp, self.items)

    def getgbestbin(self):
        # This function sends back the most filled bin of the global best
        # solution, which is chosen by tournament niche method.
        i = random.randrange(0, len(self.gbest))
        j = random.randrange(0, len(self.gbest))
        nichei = self.gbest[i].getniche()
        nichej = self.gbest[j].getniche()
        if nichei < nichej:
            gbestsol = self.gbest[i]
        elif nichej < nichei:
            gbestsol = self.gbest[j]
        else:
            gbestsol = random.choice([self.gbest[i], self.gbest[j]])
        pgd = gbestsol.getpbest()
        return pgd

    def mutation(self):
        # Step 4: Mutation operator
        print('Step 4: Mutation')
        for m in range(self.pop):
            ranm1 = random.random()
            if ranm1 < 0.33:
                self.swarm[m].partswap(self.bpp)
            elif ranm1 < 0.66:
                self.swarm[m].mergebins(self.bpp)
            else:
                self.swarm[m].splitbin()

    def heuristic(self):
        # Step 5: Solution encoding
        print('Step 5: Heuristic for solution encoding')
        for m in range(self.pop):
            self.swarm[m].updatechrom()

    def endgen(self):
        # Step 1 of MOEPSO
        return self.funkeval >= self.end

    def updatefe(self):
        # This function keeps track of the number of function evaluations.
        self.funkeval += 1

    def updateid(self):
        self.idnum += 1

    def gett(self):
        return self.t

    def getp(self):
        return self.p

    def getgbest(self):
        return self.gbest


def see(ndset, folder, flag):
    # see prints out a file to show how x and y for a gene
    # To run this module, enable the following line:
    # import numpy as np
    pathname = folder + flag
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    for m in ndset:
        x, y = m.getx(), m.gety()
        solid = str(m.getindex())
        np.savetxt(pathname + solid + '_x.txt', x, fmt='%i', header='Item Location Matrix x:')
        np.savetxt(pathname + solid + '_y.txt', y, fmt='%i', header='Bins Used Matrix y:')
        vlrep = m.getvlrep()
        outf.printvlrep(pathname + solid + '_vlrep.txt', vlrep)


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


def main():
    n = eval(input('Please enter the number of items to be sorted: \n'))
    folder = input('Please enter the name of the folder where your input file is: \n')
    inputfile = input('Please enter the name of the input file: \n')
    moepso(n, folder, inputfile)


if __name__ == '__main__':
    main()