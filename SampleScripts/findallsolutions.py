# nsgaii.py
#    Program to find solutions to the toy dynamic benchmark.
#    Author: Kristina Spencer
#    g - generation index
#    i - bin index
#    j - item index
#    k - algorithm index
#    m - solution index
#    t - time index

from __future__ import print_function
import csv
import h5py
import numpy as np
import operator as op
import random
import sys
from copy import deepcopy
from functools import reduce
from itertools import combinations
from operator import attrgetter
from os import mkdir, path

from gammapc import coolcookies, grasp, mooproblem as mop


def main():
    # The toy problem has 96 cookies in it, batches of 24.
    # Each box can hold 10 cookies, and the cooling rack can hold 30 cookies.
    # This algorithm will find the True Pareto Front by searching all of the
    # possibilities.
    n = 24
    batchsize = 6
    boxcap = 8
    rackcap = 15
    fillcap = 2
    nbatches = 4
    beta = 5
    folder = '/Users/gelliebeenz/Documents/Python/ObjectiveMethod/' \
             'TimeDependent/ToyProblem/'
    datafile = 'Cookies24.txt'
    data = folder + datafile
    cookies = coolcookies.makeobjects(n, batchsize, data)
    moop = mop.MOCookieProblem(n, boxcap, rackcap, fillcap, cookies)

    # Fill boxes with all possibilities using bpp
    archive = findallvlrep(n, nbatches, moop)
    p = len(archive)
    # Find best solutions
    print('FNDS of combinations')
    chunks = int(len(archive) / 500)
    nextround = []
    for chnk in range(chunks):
        print('     Chunk', chnk)
        archive[500*chnk: 500*(chnk + 1)], fronts = fnds(archive[500*chnk: 500*(chnk + 1)])
        for m in range(len(fronts[0])):
            r, rcl_t = getresiduals(fronts[0][m], moop)
            solution = ls_time(fronts[0][m], rcl_t, moop)
            if not np.all(np.equal(fronts[0][m], solution.getfits())):
                solution.updateid(p)
                p += 1
                fronts[0][m] = solution
        nextround.extend(fronts[0])
    archive[500*chunks - 1: -1], fronts = fnds(archive[500*chunks - 1: -1])
    nextround.extend(fronts[0])
    nextround, fronts = fnds(nextround)
    printapproxset(fronts)
    # Find fill times that optimize objectives 2 and 3
    print('Local Search from Approximate Set')
    finalround = list(fronts[0])
    numls = len(fronts[0])
    print(' - beginning local search on objective #2')
    for m in range(numls):
        p, neighbors = ls2(p, finalround[m], numls, moop, beta)
        finalround.extend(neighbors)
    print(' - beginning local search on objective #3')
    for m in range(len(finalround)):
        p, neighbors = ls3(p, finalround[m], numls, moop)
        finalround.extend(neighbors)
    print('Final FNDS')
    finalround, finalfronts = fnds(finalround)
    printapproxset(finalfronts)

    # Make output
    savefitvals(finalfronts[0], folder)
    savexys(finalfronts[0], folder)
    see(finalfronts[0], folder)
    print('This algorithm has completed successfully.')


def settfill0(n, nbatches, nbins):
    if nbins >= nbatches:
        tintervals = (600 * (nbatches + 2) - 1400) / nbins
    else:
        tintervals = 600 * nbatches / nbins
    tfill0 = np.zeros(n)
    for i in range(nbins):
        tfill0[i] = tintervals * i + 1400
    return tfill0


def findallvlrep(n, nbatches, moop):
    # This algorithm tries to find every combination of vlrep
    p = 0
    archive = []
    nbin_divisible = [3, 4]
    for nbins in nbin_divisible:
        tfill0 = settfill0(n, nbatches, nbins)
        possible_openspots = int(10 - 30 / nbins)
        # Find all combinations with set number of bins
        more_combos = getcombos(nbins - 1, n, moop, tfill0, possible_openspots)
        print('Found {0} different loading combinations.'.format(len(more_combos)))
        # Make into solutions
        for q in range(len(more_combos)):
            newsol = CookieSol(n, p, more_combos[q], tfill0)
            newsol = makereal(newsol, moop)
            archive.append(newsol)
            p += 1
    return archive


def getcombos(i, n, moop, tfill0, possible_openspots):
    # This recursive function retrieves all the possible variations of
    # bin combinations.
    # i is the bin we're finding combos for, nbins is the total number
    # of bins, n is the number of cookies
    if i == 0:
        packing_combos = []
        cookedlist = moop.cookedbyt(range(n), tfill0[i])
        # Find all possible combinations of cookies in bin i
        # for opspot in range(possible_openspots + 1):
        options = list(combinations(cookedlist, moop.boxcap - possible_openspots))
        for g in range(len(options)):
            packing_combos.append([options[g]])
    else:
        packing_combos = []
        combo_options = getcombos(i-1, n, moop, tfill0, possible_openspots)
        for c in range(len(combo_options)):
            combo_c = list(combo_options[c])
            notboxed = [x for x in range(n)
                        if sum(y.count(x) for y in combo_c) != 1]
            cookedlist = moop.cookedbyt(notboxed, tfill0[i])
            # Find all possible combinations of cookies in bin i
            # for opspot in range(possible_openspots + 1):
            options = list(combinations(cookedlist, moop.boxcap - possible_openspots))
            add = min(len(options), int(max(100000/len(combo_options), 1)))
            for g in range(add):
                packing_combos.append(list(combo_c)+[options[g]])
    print(' --> Filled bin {0}: {1} combinations'.format(i, len(packing_combos)))
    return packing_combos


def ncr(n, r):
    # Number of combinations calculator
    # n - number of possibilities
    # r - sample size
    r = min(r, n-r)
    if r == 0:
        return 1
    numer = reduce(op.mul, range(n, n-r, -1))
    denom = reduce(op.mul, range(1, r+1))
    return numer//denom


def test_domination(solution, neighbor):
    u = solution.getfits()
    v = neighbor.getfits()
    if mop.dom2(v, u):
        return neighbor
    else:
        return solution


def ls_time(solution, rcl_t, moop):
    # This function seeks to find a better time to fill bins
    # Start by finding the dynamic residual matrix for the cooling rack
    neighbor = deepcopy(solution)
    tfill = neighbor.gettfill()
    # tcheck = get_times_to_check(tfill, neighbor.gettavail(), moop)
    # r_cr = getdynamicresidual(tcheck, neighbor, moop)
    i_tlowtohigh = list(np.argsort(tfill[:neighbor.openbins], axis=0))
    for i in i_tlowtohigh:
        neighbor, r_cr = find_new_tfilli_2(i, neighbor, rcl_t, moop)
    # Check if modified solution is nondominated
    neighbor = makereal(neighbor, moop)
    winner = test_domination(solution, neighbor)
    return winner


def ls2(p, solution, numls, moop, beta):
    # Heuristic to locate a better solution in terms of the second objective:
    # minimizing the weighted average initial heat in a box
    k = 0
    neighbors = []
    searchfrom = solution
    while k < numls:
        k, coolneighbor, rcl_t = ls2_loading(k, searchfrom, moop, beta)
        if coolneighbor:
            coolneighbor = ls_time(coolneighbor, rcl_t, moop)
            coolneighbor.updateid(p)
            p += 1
            neighbors.append(coolneighbor)
            searchfrom = coolneighbor
        else:
            k = numls
    return p, neighbors


def ls2_loading(k, searchfrom, moop, beta):
    # This function finds the restricted candidate list and tries to move
    # cookies toward more favorable configurations to minimize the weighted avg
    u = searchfrom.getfits()
    r, rcl_t = getresiduals(searchfrom, moop)
    copy = deepcopy(searchfrom)
    hotbins = np.argsort(searchfrom.getq0bins())
    for s in range(searchfrom.openbins):
        i = hotbins[-s - 1]
        vlrep = copy.getvlrep()
        # If there is only one item in the box, no point in moving
        if len(vlrep[i]) < 2:
            return k, None, rcl_t
        rcl_j = ls2_makercl(i, vlrep, beta)
        k, newsol, rcl_t = search_rclj(k, i, copy, u, r, rcl_j, rcl_t, moop)
        if newsol:
            return k, newsol, rcl_t
    # If a nondominated solution wasn't found, return nothing
    return k, None, rcl_t


def ls2_makercl(i, vlrep, beta):
    # This function returns the restricted candidate list for local search 2
    # Restricted candidate list
    binkeys = list(vlrep[i])
    avglen = averageLen(vlrep)
    nrcl_min = min(len(binkeys) - 1, beta)
    nrcl = max(len(binkeys) - avglen, nrcl_min)
    rcl_j = random.sample(binkeys, nrcl)
    return rcl_j


def find_new_tfilli(i, neighbor, tcheck, r_cr, moop):
    vlrep = neighbor.getvlrep()
    # Set old tfill[i] for comparison
    tfilli_old = neighbor.tfill[i]
    tcheck_s = np.where(tcheck >= tfilli_old)[0]
    for s in tcheck_s:
        leftover_room = r_cr[s] - len(vlrep[i])
        if -len(vlrep[i]) <= leftover_room <= 0:
            # Don't do anything if not changing tfill[i]
            if tfilli_old == tcheck[s]:
                return neighbor, r_cr
            # Make new tfill[i]
            else:
                neighbor, r_cr = set_new_tfilli(i, s, neighbor, tcheck, r_cr, moop)
                return neighbor, r_cr
        elif leftover_room < -len(vlrep[i]):
            # Don't do anything if not changing tfill[i]
            if tfilli_old == tcheck[s - 1]:
                return neighbor, r_cr
            # Make new tfill[i]
            else:
                neighbor, r_cr = set_new_tfilli(i, s - 1, neighbor, tcheck, r_cr, moop)
                return neighbor, r_cr
    # If didn't find boundary, move a bit later
    s = np.where(tcheck == neighbor.gettavail()[i])[0]
    neighbor, r_cr = set_new_tfilli(i, s[0] - 1, neighbor, tcheck, r_cr, moop)
    return neighbor, r_cr


def set_new_tfilli(i, s, neighbor, tcheck, r_cr, moop):
    # Edit tfill[i]
    neighbor.edit_tfilli(i, tcheck[s])
    # Get min time cookie from box i is on cooling rack
    t_min = get_box_tmin(neighbor.getvlrep()[i], moop)
    # Recalculate r_CR values at times between t_min and new tfill[i]
    qlist = np.where(t_min <= tcheck)[0]
    for q in qlist:
        nrackitems = countonrack(tcheck[q], neighbor.getvlrep(),
                                 neighbor.gettfill(), moop)
        r_cr[q] = moop.coolrack - nrackitems
    return neighbor, r_cr


def find_new_tfilli_2(i, neighbor, rcl_t, moop):
    # This function determines a new time for box i to be filled and updates
    # the RCLTime instance.
    vlrep = neighbor.getvlrep()
    tfill = neighbor.gettfill()
    told = tfill[i]
    t, rcl_t = get_feasible_tfilli(rcl_t, vlrep[i], moop)
    if t:
        neighbor.edit_tfilli(i, t)
        # Adapt Greedy Function
        rcl_t.adapt_changetime(told, t, len(vlrep[i]))
    return neighbor, rcl_t


def get_feasible_tfilli(rcl_t, vlrepi, moop):
    # This function locates a new value for tfill[i] that doesn't violate rack
    # or fill limits
    tmin = get_box_tmin(vlrepi, moop)
    args = [tmin, 'hload', len(vlrepi)]
    # Find new time for box i
    t_new, p_t, rcl_t = find_new_time_value(rcl_t, *args)
    if not t_new:
        return None, rcl_t
    kappa = 0                                           # Counter to exit loop
    # Check if possible to fill in period
    while rcl_t.res_fill[p_t] < 1:
        if kappa == 10:
            return None, rcl_t
        # If not possible, find new time value
        t_new, p_t, rcl_t = find_new_time_value(rcl_t, *args)
        kappa += 1
    # If returning t_new to open bin, reduce fill capacity by 1
    rcl_t.res_fill[p_t] -= 1
    return t_new, rcl_t


def get_box_tmin(vlrepi, moop):
    # Find minimum time for box i
    boxi_contents = {k: v for k, v in moop.cookies.items() if k in vlrepi}
    maxbatch = max(boxi_contents.values(), key=attrgetter('batch')).batch
    tmin = maxbatch * 600
    return tmin


def find_new_time_value(rcl_t, *args):
    # This module retrieves a new time value and also returns which period
    # it belongs to
    t_new = rcl_t.get_new_t(args[0], mode=args[1], nmove=args[2])
    if not t_new:
        return None, None, rcl_t
    # If the new time value is beyond the current fill periods, extend
    while t_new > rcl_t.t_t[-1]:
        rcl_t.extend_fill_periods()
    # Find the period containing t_new
    tlist = np.where(t_new > np.array(rcl_t.t_t))[0]
    return t_new, tlist[-1], rcl_t


def ls3(p, solution, numls, moop):
    # Heuristic to locate a better solution in terms of the third objective:
    # minimizing the maximum time to move to store front.
    k = 0
    neighbors = []
    searchfrom = solution
    while k < numls:
        k, coolneighbor, rcl_t = ls3_rcl(k, numls, searchfrom, moop)
        if coolneighbor:
            coolneighbor = ls_time(coolneighbor, rcl_t, moop)
            coolneighbor.updateid(p)
            p += 1
            neighbors.append(coolneighbor)
            searchfrom = coolneighbor
        else:
            k = numls
    return p, neighbors


def ls3_rcl(k, numls, searchfrom, moop):
    # This function finds the restricted candidate list for bin i and tries to
    # move cookies to find a new nondominated solution. If unsuccessful, moves
    # to a new bin
    u = searchfrom.getfits()
    r, rcl_t = getresiduals(searchfrom, moop)
    copy = deepcopy(searchfrom)
    latebins = np.argsort(searchfrom.gettavail(), axis=0)
    for s in range(searchfrom.openbins):
        if k == numls:
            return k, None, rcl_t
        i = latebins[-s - 1]
        binkeys = list(copy.getvlrep(i=i))
        # If there is only one item in the box, no point in moving
        if len(binkeys) < 2:
            return k, None, rcl_t
        # Restricted candidate list
        n_rclj = int(0.5 * len(binkeys))
        rcl_j = binkeys[-n_rclj - 1: -1]
        k, newsol, rcl_t = search_rclj(k, i, copy, u, r, rcl_j, rcl_t, moop)
        if newsol:
            return k, newsol, rcl_t
    # If a nondominated solution wasn't found, return nothing
    return k, None, rcl_t


def search_rclj(k, i, solution, u, r, rcl_j, rcl_t, moop):
    # This function moves cookies into new boxes until either it finds a new
    # nondominated solution or it runs out of candidates from this solution
    for m in range(len(rcl_j)):
        k += 1
        j = random.choice(rcl_j)
        rcl_j.remove(j)
        r, rcl_t, solution = lsmove(i, j, r, rcl_t, solution, moop)
        # Check if modified solution is nondominated
        solution = makereal(solution, moop)
        v = solution.getfits()
        if not mop.dom2(u, v):
            return k, solution, rcl_t
    return k, None, rcl_t


def lsmove(i, j, r, rcl_t, solution, moop):
    # This function determines where cookie j should move to
    m = solution.getopenbins()
    tfill = solution.gettfill()
    # Gather bin options and pick new bin for the move
    ilist = ls3options(m, moop.cookies.get(j), r, rcl_t, tfill)
    inew = random.choice(ilist)
    # Open a new bin or move cookie to a new bin
    if inew == solution.openbins:
        t, rcl_t = get_feasible_tfilli(rcl_t, [j], moop)
        if not t:
            try:
                solution.moveitem(i, j, ilist[1])
            except:
                return r, rcl_t, solution
        else:
            solution.opennewbin(i, j, round(t, 1))
            r[inew, 0] = moop.boxcap
            r[inew, 1] = rcl_t.adapt_greedy_function_newbin(t)
    else:
        solution.moveitem(i, j, inew)
        r[i, 1], r[inew, 1] = rcl_t.adapt_movebins(tfill[i], tfill[inew])
    r = update_spaceresiduals(r, i, inew)
    return r, rcl_t, solution


def ls3options(m, cookie, r, rcl_t, tfill):
    # This function retrieves a candidate list for moving a cookie.
    bcookiej = cookie.getbatch()            # cookie batch number
    tmax = rcl_t.get_tmax(bcookiej * 600, 1)
    i_rlowtohigh = np.argsort(r[:m], axis=0)
    # This module performs the sorting for module ll.
    for j in range(m):
        # Find open bin with max. residual value, moving backward thru i_rlowtohigh
        lsi = i_rlowtohigh[-1 - j, 0]
        if tfill[lsi] <= tmax:
            pack = packable(r[lsi, 0], r[lsi, 1], tfill[lsi], cookie)
            if pack:
                return [m, lsi]
    # If least loaded bin won't fit item, need to open new bin.
    return [m]


def getresiduals(solution, moop):
    # This function calculates the residual matrix associated with a given
    # dynamic bin packing loading. The first column represents the open box
    # capacities, and the second column represents the maximum number of
    # cookies that can be added to the cooling rack right before tfill_i
    n = solution.n
    vlrep = solution.getvlrep()
    coolrack = moop.coolrack
    r = np.zeros((n, 2), dtype=np.int)
    # Set box capacity residuals
    for i in range(len(vlrep)):
        r[i, 0] = moop.boxcap - len(vlrep[i])
        r[i, 1] = coolrack
    # Set cooling rack capacity residuals
    n_b = n // moop.nbatches
    rcl_t = grasp.RCLtime(coolrack, moop.fillcap, n_b, moop.tbatch, moop.nbatches)
    r[:len(vlrep), 1] = rcl_t.initialize_withtfill(len(vlrep), vlrep,
                                                   solution.gettfill())
    return r, rcl_t


def update_spaceresiduals(r, i, inew):
    # This function updates the space residual r after a cookie moves
    # from box i to box inew
    # Update r: box capacity
    r[i, 0] += 1
    r[inew, 0] -= 1
    return r


def countonrack(t, vlrep, tfill, moop):
    # Cookies from boxes filled after t might be on rack
    timecheckindices = np.where(tfill > t)
    nrackitems = 0
    for i in timecheckindices[0]:
        for j in vlrep[i]:
            onrack = moop.rackij(t, tfill[i], moop.cookies.get(j))
            if onrack == 1:
                nrackitems += 1
    return nrackitems


def getdynamicresidual(tcheck, neighbor, moop):
    # This function returns a residual matrix such that
    # rCR_s0 = tcheck[s]
    # rCR_s1 = open spots on cooling rack
    len_s = len(tcheck)
    r_crlist = []
    for s in range(len_s):
        r_cr0 = tcheck[s]
        nrackitems = countonrack(r_cr0, neighbor.getx(),
                                 neighbor.gettfill(), moop)
        r_cr1 = moop.coolrack - nrackitems
        r_crlist.append(r_cr1)
    r_cr = np.array(r_crlist, dtype=np.int)
    return r_cr


def get_times_to_check(tfill, tavail, moop):
    # This function creates a numpy array of times to check the
    # cooling rack. It combines current tfill times, times where
    # cookies are being removed from the oven, and the t_available matrix.
    tfill_unique = np.unique(tfill)
    tbakes = [(bj + 1) * moop.tbatch for bj in range(moop.nbatches)]
    t_innout = np.concatenate((tfill_unique, np.array(tbakes), tavail))
    tcheck = np.unique(t_innout)
    return tcheck[1:]


def packable(rone, rtwo, tfilli, cookie):
    # This module checks to see if object j can fit inside bin i at time tfilli
    # Capacity constraints
    r1 = rone - 1
    r2 = rtwo - 1
    # cappack =
    # Time constraint
    #   tbatch = 10 min = 600 s
    t_cook = cookie.getbatch() * 600
    return r1 >= 0 and r2 >= 0 and t_cook < tfilli


def picktime_fromcoolrack(solution, j, r, moop):
    # This function returns the first time that cookie exists on the
    # cooling rack based on the time intervals used to check capacity
    # j - cookie to open new box with
    m = solution.openbins
    tfill = solution.gettfill()
    bk = moop.cookies.get(j).getbatch()
    tmin = bk * moop.tbatch
    rlist = np.where(tfill > bk * moop.tbatch)[0]
    i_tlowtohigh = np.argsort(tfill[:m], axis=0)
    t_low, t_high = get_trange(m, tmin, tfill, r, i_tlowtohigh, rlist)
    tpack = random.uniform(t_low, t_high)
    nrack = countonrack(tpack, solution.getvlrep(), tfill, moop)
    while nrack >= moop.coolrack:
        t_high = tpack
        if round(t_low, -1) == round(t_high, -1):
            return False
        tpack = random.uniform(t_low, t_high)
        nrack = countonrack(tpack, solution.getvlrep(), tfill, moop)
    return tpack


def get_trange(m, tmin, tfill, r, i_tlowtohigh, rlist):
    # Find one time interval too late for cookie j to be packed away
    # Returns t_undermax, t_overmax
    tfill_unique = np.unique(tfill)
    tdiff = np.absolute(np.ediff1d(tfill_unique))
    rlist = np.where(tfill_unique > tmin)[0]
    for imax in rlist:
        if r[imax, 1] - 1 < 0:
            optionb = tfill_unique[imax] - np.max(tdiff)
            return max(tmin, optionb), tfill_unique[imax]
    # If didn't find a barrier, return another option
    return tmin + random.uniform(0, 600), tfill_unique[-1]


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


def printapproxset(fronts):
    print('The Pareto Front has {0} solutions in it.'.format(len(fronts[0])))
    print('Frnt Member  Solution ID     f[1] (B)    f[2] (W)        f[3] (s)')
    print('------------------------------------------------------------------')
    for m in range(len(fronts[0])):
        print("{0:11d}  {1:11d}     {2:8.0f}    {3:8.2f}    {4:12.1f}".
              format(m, fronts[0][m].getid(), fronts[0][m].getbins(),
                     fronts[0][m].getavgheat(), fronts[0][m].getmaxreadyt()))


class CookieSol:
    def __init__(self, n, index, vlrep_in, tfill):
        self.n = n
        self.idnum = index
        self.vlrep = []
        for i in range(len(vlrep_in)):
            self.vlrep.append(list(vlrep_in[i]))
        self.tfill = tfill
        self.openbins = len(vlrep_in)
        self.x = np.zeros((n, n), dtype=np.int)     # initialize x
        self.y = np.zeros(n, dtype=np.int)          # initialize y
        self.tavail = np.zeros(n)                   # initialize tavail
        self.q0bins = np.zeros(n)
        self.makexandy()
        self.rank = 0

    def makexandy(self):
        for i in range(len(self.vlrep)):
            self.y[i] = 1
            for j in self.vlrep[i]:
                self.x[i,j] = 1

    def edit_tfilli(self, i, t):
        # This function changes self.tfill[i] to t
        self.tfill[i] = t

    def updatefitvals(self, fitvals):
        self.fitvals = fitvals
        self.fit0 = fitvals[0]
        self.fit1 = fitvals[1]
        self.fit2 = fitvals[2]

    def settavailable(self, tavail):
        self.tavail = tavail

    def setq0bins(self, q0bins):
        self.q0bins = q0bins

    def moveitem(self, i, j, inew):
        # This function moves cookie j2 from box i2 to box i
        if self.y[inew] == 0:
            self.y[inew] = 1
            self.openbins += 1
        # Move in x-matrix
        self.x[i, j] = 0
        self.x[inew, j] = 1
        # Move in variable length representation
        self.vlrep[i].remove(j)
        self.vlrep[inew].append(j)
        # Resort bin inew to keep js in order
        self.vlrep[inew].sort()

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

    def initopenbins(self):
        for i in range(self.n - 1, 0, -1):
            if self.y[i] == 1:
                self.openbins = i + 1
                break

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
                    print('Error: y does not match vlrep in solution', self.idnum)

    def getid(self):
        return self.idnum

    def updateid(self, idnum):
        self.idnum = idnum

    def getx(self):
        return self.x

    def gety(self):
        return self.y

    def gettfill(self):
        return self.tfill

    def gettavail(self):
        return self.tavail

    def getq0bins(self):
        return self.q0bins

    def getvlrep(self, i=None):
        if i:
            return self.vlrep[i]
        elif i == 0:
            return self.vlrep[0]
        else:
            return self.vlrep

    def getfits(self):
        return self.fitvals

    def getbins(self):
        return self.fit0

    def getavgheat(self):
        return self.fit1

    def getmaxreadyt(self):
        return self.fit2

    def getopenbins(self):
        return self.openbins

    def updaterank(self, prank):
        self.rank = prank

    def getrank(self):
        return self.rank


def makereal(solution, moop):
    # This function checks the feasibility of a solution and calculates fitness
    # values.
    solution = moop.calcfeasibility(solution)
    checkformismatch(solution)
    fits = moop.calcfits(solution)
    solution.updatefitvals(fits)
    return solution


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


def averageLen(lst):
    # Calculates the average length of lists inside a list, returns integer value
    lengths = [len(i) for i in lst]
    return 0 if len(lengths) == 0 else (int(sum(lengths) / len(lengths)))


main()