# nsgaii.py
#    Program to run NSGA-II on benchmark problems
#    Author: Kristina Spencer
#    Date: March 10, 2016
#    i - bin index
#    j - item index
#    k - algorithm index
#    m - solution index
#    t - generation ind#x

from __future__ import print_function
import binpacking as bp
import datetime
import ga
import numpy as np
import os
import solutions as sols
from items import makeitems
from mop import dom
from operator import attrgetter


def nsgaii(n, folder, file):
    import outformat as outf
    from glob import glob
    existing_files = glob(folder + '*.out')
    filename = folder + 'run%d.out' % (len(existing_files) + 1)
    data = folder + file

    # Initialize algorithm
    pop = 100       # members per gen.
    end = 25000     # function evaluations or 250 gen.
    outf.startout(filename, n, end, data, 'NSGA-II')
    startt = datetime.datetime.now()
    print('                         ', startt)
    print('*******************************************************************************')
    print('     Method: NSGA-II\n')
    print('     data: ', file)
    print('*******************************************************************************')
    binc, binh, items = makeitems(data)
    bpp = bp.BPP(n, binc, binh, items)
    gen = Generation(n, pop, end, items, bpp)
    gen.initialp()
    gen.makeq()

    # NSGA-II
    while not gen.endgen():
        gen.rungen()
        outf.genout(filename, gen.gett(), pop, gen.getq(), gen.getfnts())

    # Make output
    ndset = approx(gen.getarch())
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
    def __init__(self, n, popsize, end, items, bpp):
        self.n = int(n)
        self.pop = int(popsize)
        self.end = int(end)
        self.t = 0
        self.items = items
        self.bpp = bpp
        self.idnum = 0
        self.p = []
        self.q = []
        self.fronts = []
        self.archive = []

    def rungen(self):
        self.t += 1
        if self.t % 50 == 0:
            gent = datetime.datetime.now()
            print('                         ', gent)
        print('t = ', self.t)
        self.makep()
        if self.t == 1:
            self.q = ga.binsel(self.p, self.pop, 'elitism')
        else:
            self.q = ga.binsel(self.p, self.pop, 'cco')
        self.newgenes = ga.xover(self.q, self.pop, 0.9)
        self.newgenes = ga.mutat(self.n, self.newgenes, self.pop)
        self.makeq()

    def initialp(self):
        chromosomes = []
        for j in range(self.n):
            chromosomes.append(self.items[j].getindex())
        import random
        random.seed(52)
        self.newgenes = []
        for i in range(self.pop):
            x = random.sample(chromosomes, len(chromosomes))
            self.newgenes.append(x)

    def makep(self):
        r = self.p + self.q
        r, self.fronts = fnds(r)
        print('There are currently ', len(self.fronts[0]), 'solutions in the Approximate Set.\n')
        if self.t == 1:
            self.p = r
        else:
            self.p = fill(self.fronts, self.pop)

    def makeq(self):
        if self.t == 0:
            new = range(self.pop)
        else:
            new, self.q = sols.oldnew(self.archive, self.q, self.newgenes)
        print('Found ', len(new), 'new solutions. \n')
        for m in new:
            newsol = sols.process(self.idnum, self.t, self.newgenes[m], self.bpp, self.items)
            self.q.append(newsol)
            self.archive.append(newsol)
            self.updateid()
        self.archive = sols.reduce(self.archive, self.p, self.q)

    def endgen(self):
        if self.t < 3:
            return False
        maxid = self.q[self.pop - 1].getindex()
        return maxid > self.end

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


def fill(fronts, pop):
    # This module fills the parent population based on the crowded
    # comparison operator.
    # To run this module, enable the following line:
    # from operator import attrgetter
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
    # To run this module, enable the following line:
    # from mop import dom
    numsol = len(archive)
    ndset = []
    for p in range(numsol):
        np = 0
        for q in range(numsol):
            if archive[p] != archive[q]:
                if dom(archive[q].getfits(), archive[p].getfits()):
                    np += 1
        if np == 0:
            ndset.append(archive[p])
    return ndset


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
    nsgaii(n, folder, file)


if __name__ == '__main__':
    main()
