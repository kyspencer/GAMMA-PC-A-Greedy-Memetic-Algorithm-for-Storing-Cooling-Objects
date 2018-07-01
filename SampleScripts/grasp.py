# grasp.py
#   This script implements the GRASP heuristic for the dynamic bin packing
#   problem.
#   Author: Kristina Yancey Spencer

from __future__ import print_function
import numpy as np
import random
import solutions_dynamic as solmaker
import sys
from copy import deepcopy
from itertools import combinations
from math import ceil, sqrt
from operator import attrgetter


class BPP:
    # This class groups the bin packing problem information and performs
    # the GRASP operations.
    def __init__(self, n, cookies, moop):
        self.beta = 5               # Cardinality restriction
        self.n = int(n)             # Number of cookies to sort
        self.cookies = cookies      # dictionary of item objects
        self.moop = moop            # Multiobjective problem class
        self.lb = 0                 # initialize lower bound
        self.calclowerbound()

    def generate_newsol(self, index, p_ls1, p_ls2, *args):
        # This module creates an instance of a NewSolution class and
        # performs the generate_newsol procedure
        newbie = NewSolution(self.beta, self.n, self.cookies, self.moop)
        newsol = newbie.make_newsol(index, *args)
        newsol = self.checkandfit(newsol)
        p = index + 1       # ID number for first neighbor
        rannum = random.random()
        if rannum < p_ls1:
            if newsol.getopenbins() > self.lb:
                p, neighbors = self.ls1(p, 1, newsol)
            else:
                p, neighbors = self.bin_mutation(p, 1, newsol)
        elif rannum < p_ls2:
            p, neighbors = self.ls2(p, 1, newsol)
        else:
            p, neighbors = self.ls3(p, 1, newsol)
        if neighbors:
            winner = self.test_domination(newsol, neighbors[0])
            return p, winner
        return p, newsol

    def checkandfit(self, solution):
        # This function checks the feasibility of a solution and calculates fitness
        # values.
        solution = self.moop.calcfeasibility(solution)
        checkformismatch(solution.getx(), solution.getvlrep())
        fits = self.moop.calcfits(solution)
        solution.updatefitvals(fits)
        return solution

    def test_domination(self, solution, neighbor):
        # This function determines if neighbor dominates solution.
        u = solution.getfits()
        v = neighbor.getfits()
        if dom2(v, u):
            return neighbor
        else:
            return solution

    def ls_time(self, solution, rcl_t):
        # This function seeks to find a better time to fill bins
        # Start by finding the dynamic residual matrix for the cooling rack
        neighbor = deepcopy(solution)
        tfill = neighbor.gettfill()
        i_tlowtohigh = list(np.argsort(tfill[:neighbor.openbins], axis=0))
        for i in i_tlowtohigh:
            neighbor, rcl_t = self.find_new_tfilli(i, neighbor, rcl_t)
        # Check if modified solution is nondominated
        neighbor = self.checkandfit(neighbor)
        winner = self.test_domination(solution, neighbor)
        return winner

    def find_new_tfilli(self, i, solution, rcl_t):
        # This function determines a new time for box i to be filled and updates
        # the RCLTime instance
        vlrep = solution.getvlrep()
        tfill = solution.gettfill()
        told = tfill[i]
        tmin = self.get_box_tmin(vlrep[i])
        kwargs = {'mode': 'hload', 'nmove': len(vlrep[i]), 'told': told}
        t, rcl_t = self.get_feasible_tfilli(rcl_t, tmin, **kwargs)
        if t:
            solution.edit_tfilli(i, t)
            # Adapt Greedy Function
            rcl_t.adapt_changetime(told, t, len(vlrep[i]))
        return solution, rcl_t

    def get_feasible_tfilli(self, rcl_t, tmin, **kwargs):
        # This function locates a new value for tfill[i] that doesn't violate
        # rack or fill limits
        # Find new time for box i
        t_new, p_t, rcl_t = self.find_new_time_value(rcl_t, tmin, **kwargs)
        if not t_new:
            return None, rcl_t
        kappa = 0                               # Counter to exit loop
        # Check if possible to fill in period
        while rcl_t.res_fill[p_t] < 1:
            if kappa == 10:
                return None, rcl_t
            # If not possible, find new time value
            t_new, p_t, rcl_t = self.find_new_time_value(rcl_t, tmin, **kwargs)
            if not t_new:
                return None, rcl_t
            kappa += 1
        # If returning t_new to open bin, reduce fill capacity by 1
        rcl_t.res_fill[p_t] -= 1
        return t_new, rcl_t

    def get_box_tmin(self, vlrepi):
        # Find minimum time for box i
        boxi_contents = {k: v for k, v in self.cookies.items() if k in vlrepi}
        maxbatch = max(boxi_contents.values(), key=attrgetter('batch')).batch
        tmin = maxbatch * 600
        return tmin

    def find_new_time_value(self, rcl_t, tmin, **kwargs):
        # This module retrieves a new time value and also returns which period
        # it belongs to
        t_new = rcl_t.get_new_t(tmin, **kwargs)
        if not t_new:
            return None, None, rcl_t
        t_p = self.find_t_in_fill_periods(t_new, rcl_t)
        return t_new, t_p, rcl_t

    def find_t_in_fill_periods(self, t, rcl_t):
        # If the new time value is beyond the current fill periods, extend
        while t > rcl_t.t_t[-1]:
            rcl_t.extend_fill_periods()
        # Find the period containing t_new
        tlist = np.where(t >= np.array(rcl_t.t_t))[0]
        return tlist[-1]

    def ls1(self, p, numls, solution):
        # Heuristic to locate a better solution in terms of the first objective:
        # minimizing the number of bins in use
        k = 0
        neighbors = []
        searchfrom = solution
        while k < numls:
            coolneighbor, rcl_t = self.ls1_loading(searchfrom)
            if coolneighbor:
                k += 1
                coolneighbor = self.ls_time(coolneighbor, rcl_t)
                coolneighbor.updateid(p)
                p += 1
                neighbors.append(coolneighbor)
                searchfrom = coolneighbor
            else:
                k = numls
        return p, neighbors

    def ls2(self, p, numls, solution):
        # Heuristic to locate a better solution in terms of the second objective:
        # minimizing the weighted average initial heat in a box
        # p - current id number for new solution
        # numls - number of neighbors to find during local search
        # Returns updated p and list of neighbors
        k = 0
        neighbors = []
        searchfrom = solution
        while k < numls:
            k, coolneighbor, rcl_t = self.ls2_loading(k, searchfrom)
            if coolneighbor:
                coolneighbor = self.ls_time(coolneighbor, rcl_t)
                coolneighbor.updateid(p)
                p += 1
                neighbors.append(coolneighbor)
                searchfrom = coolneighbor
            else:
                k = numls
        return p, neighbors

    def ls3(self, p, numls, solution):
        # Heuristic to locate a better solution in terms of the third objective:
        # minimizing the maximum time to move to store front.
        k = 0
        neighbors = []
        searchfrom = solution
        while k < numls:
            k, coolneighbor, rcl_t = self.ls3_loading(k, searchfrom)
            if coolneighbor:
                coolneighbor = self.ls_time(coolneighbor, rcl_t)
                coolneighbor.updateid(p)
                p += 1
                neighbors.append(coolneighbor)
                searchfrom = coolneighbor
            else:
                k = numls
        return p, neighbors

    def ls1_loading(self, searchfrom):
        # This function attempts to empty the least filled bin and move its
        # cookies into available boxes.
        u = searchfrom.getfits()
        vlrep = searchfrom.getvlrep()
        r, rcl_t = self.getresiduals(vlrep, searchfrom.gettfill())
        copy = deepcopy(searchfrom)
        half = len(vlrep) // 2
        for iloop in range(half):
            # Find the emptiest bin's index number
            lengths = [len(i) for i in copy.getvlrep()]
            i = np.argmin(np.array(lengths))
            copy, r, rcl_t = self.empty_bin(i, copy, r, rcl_t)
            # If a nondominated solution wasn't found, return nothing
            copy = self.checkandfit(copy)
            v = copy.getfits()
            if not dom2(u, v):
                return copy, rcl_t
        return None, rcl_t

    def empty_bin(self, i, copy, r, rcl_t):
        # This function moves items in box i to other boxes
        for j in list(copy.getvlrep()[i]):
            # Find rcl_bins
            tfill = copy.gettfill()
            rcl_bins = self.ls1_makercl(i, j, r, rcl_t, tfill)
            if len(rcl_bins) == 0:
                return copy, r, rcl_t
            # Pick random bin
            inew = random.choice(rcl_bins)
            # Move cookie to new bin
            copy.moveitem(i, j, inew)
            r = self.update_spaceresiduals(r, i, inew)
            r[i, 1], r[inew, 1] = rcl_t.adapt_movebins(tfill[i], tfill[inew])
        return copy, r, rcl_t

    def ls1_makercl(self, iold, j, r, rcl_t, tfill):
        # This function returns the restricted candidate list for cookie
        # j to move into based on the dot product strategy
        # Set weights for the dot product array (1/boxcap, 1/coolrackcap)
        weights = [1.0 / self.moop.boxcap, 1.0 / self.moop.coolrack]
        # The cookie should not move into a box that is filled until after
        # it is done baking
        tmin = self.cookies.get(j).getbatch() * 600
        tmax = rcl_t.get_tmax(tmin, 1)
        options_byt = [i for i in range(self.n) if tfill[i] > tmin]
        if tfill[iold] != tmin:
            options_byt.remove(iold)
        # Form dot product array
        dparray = np.zeros(self.n)
        for i in options_byt:
            if tfill[i] <= tmax:
                # Make sure there is space available
                if r[i, 0] > 1:
                    tk = rcl_t.find_t_in_timeline(tfill[i])
                    # Filling early will reduce onrack for all after time[tk]
                    onrack = np.subtract(self.moop.coolrack, rcl_t.space[tk:])
                    maxonrack_fromtk = max(onrack)
                    dparray[i] = weights[0] * r[i, 0] + weights[1] * maxonrack_fromtk
        # Max fill
        if len(np.nonzero(dparray)[0]) > self.beta:
            options = list(np.argsort(-dparray)[:self.beta])
            return options
        else:
            options = list(np.nonzero(dparray)[0])
            return options

    def ls2_loading(self, k, searchfrom):
        # This function finds the restricted candidate list and tries to move
        # cookies toward more favorable configurations to minimize the weighted avg
        u = searchfrom.getfits()
        r, rcl_t = self.getresiduals(searchfrom.getvlrep(), searchfrom.gettfill())
        copy = deepcopy(searchfrom)
        hotbins = np.argsort(searchfrom.getq0bins())
        for s in range(searchfrom.openbins):
            i = hotbins[-s - 1]
            vlrep = copy.getvlrep()
            # If there is only one item in the box, no point in moving
            if len(vlrep[i]) < 2:
                return k, None, rcl_t
            rcl_j = self.ls2_makercl(i, vlrep)
            k, newsol, rcl_t = self.search_rclj(k, i, copy, u, r, rcl_j, rcl_t)
            if newsol:
                return k, newsol, rcl_t
        # If a nondominated solution wasn't found, return nothing
        return k, None, rcl_t

    def ls2_makercl(self, i, vlrep):
        # This function returns the restricted candidate list for local search 2
        # Restricted candidate list
        binkeys = list(vlrep[i])
        avglen = averageLen(vlrep)
        nrcl_min = min(len(binkeys) - 1, self.beta)
        nrcl = max(len(binkeys) - avglen, nrcl_min)
        rcl_j = random.sample(binkeys, nrcl)
        return rcl_j

    def ls3_loading(self, k, searchfrom):
        # This function finds the restricted candidate list for bin i and tries to
        # move cookies to find a new nondominated solution. If unsuccessful, moves
        # to a new bin
        u = searchfrom.getfits()
        r, rcl_t = self.getresiduals(searchfrom.getvlrep(), searchfrom.gettfill())
        copy = deepcopy(searchfrom)
        latebins = np.argsort(searchfrom.gettavail(), axis=0)
        for s in range(searchfrom.openbins):
            i = latebins[-s - 1]
            vlrep = copy.getvlrep()
            # If there is only one item in the box, no point in moving
            if len(vlrep[i]) < 2:
                return k, None, rcl_t
            # Restricted candidate list
            rcl_j = self.ls3_makercl(i, vlrep)
            k, newsol, rcl_t = self.search_rclj(k, i, copy, u, r, rcl_j, rcl_t)
            if newsol:
                return k, newsol, rcl_t
        # If a nondominated solution wasn't found, return nothing
        return k, None, rcl_t

    def ls3_makercl(self, i, vlrep):
        # This function returns the restricted candidate list for local search 3
        # Restricted candidate list
        binkeys = list(vlrep[i])
        n_rclj = int(0.5 * len(binkeys))
        rcl_j = binkeys[-n_rclj - 1: -1]
        return rcl_j

    def search_rclj(self, k, i, solution, u, r, rcl_j, rcl_t):
        # This function moves cookies into new boxes until either it finds a new
        # nondominated solution or it runs out of candidates from this solution
        for m in range(len(rcl_j)):
            k += 1
            j = random.choice(rcl_j)
            rcl_j.remove(j)
            r, rcl_t, solution = self.lsmove(i, j, r, rcl_t, solution)
            # Check if modified solution is nondominated
            solution = self.checkandfit(solution)
            v = solution.getfits()
            if not dom2(u, v):
                return k, solution, rcl_t
        return k, None, rcl_t

    def lsmove(self, i, j, r, rcl_t, solution):
        # This function determines where cookie j should move to
        m = solution.getopenbins()
        tfill = solution.gettfill()
        # Gather bin options and pick new bin for the move
        ilist = self.move_options(j, m, r, rcl_t, tfill)
        inew = random.choice(ilist)
        # Open a new bin or move cookie to a new bin
        if inew == m:
            tmin = self.get_box_tmin([j])
            kwargs = {'mode': 'hload'}
            t, rcl_t = self.get_feasible_tfilli(rcl_t, tmin, **kwargs)
            if t:
                solution.opennewbin(i, j, round(t, 1))
                r[inew, 0] = self.moop.boxcap
                r[inew, 1] = rcl_t.adapt_greedy_function_newbin(t)
            else:
                return r, rcl_t, solution
        else:
            solution.moveitem(i, j, inew)
            r[i, 1], r[inew, 1] = rcl_t.adapt_movebins(tfill[i], tfill[inew])
        r = self.update_spaceresiduals(r, i, inew)
        return r, rcl_t, solution

    def move_options(self, j, m, r, rcl_t, tfill):
        # This function retrieves a candidate list for moving a cookie.
        bcookiej = self.cookies.get(j).getbatch()   # cookie batch number
        tmax = rcl_t.get_tmax(bcookiej * 600, 1)
        i_rlowtohigh = np.argsort(r[:m, 0], axis=0)
        # This module performs the sorting for module ll.
        for i in range(m):
            # Find open bin with max. residual value, moving backward thru i_rlowtohigh
            lsi = i_rlowtohigh[-1 - i]
            if tfill[lsi] <= tmax:
                pack = packable(r[lsi, :], bcookiej, tfill[lsi])
                if pack:
                    return [m, lsi]
        # If least loaded bin won't fit item, need to open new bin.
        return [m]

    def bin_mutation(self, p, numls, solution):
        # Heuristic to locate a better solution in terms of the first objective:
        # minimizing the number of bins.
        k = 0
        neighbors = []
        searchfrom = solution
        while k < numls:
            k, coolneighbor, rcl_t = self.select_mutation_operation(k, searchfrom)
            if coolneighbor:
                coolneighbor.updateid(p)
                coolneighbor = self.ls_time(coolneighbor, rcl_t)
                p += 1
                neighbors.append(coolneighbor)
                searchfrom = coolneighbor
            else:
                k = numls
        return p, neighbors

    def select_mutation_operation(self, k, searchfrom):
        # This function selects the mutation operator
        vlrep = searchfrom.getvlrep()
        avg_bin_size = averageLen(vlrep)
        too_small_lengths = [i for i in vlrep if 2 * len(i) <= avg_bin_size]
        if too_small_lengths:
            k, coolneighbor, rcl_t = self.move_cookies(k, searchfrom)
        else:
            rannum = random.random()
            if rannum < 0.50:
                k, coolneighbor, rcl_t = self.part_swap(k, searchfrom)
            else:
                k, coolneighbor, rcl_t = self.cookie_swap(k, searchfrom)
        return k, coolneighbor, rcl_t

    def time_mutation_by_heat(self, solution, rcl_t):
        # This function tries a new time value for the initial hottest bin to
        # see if that helps
        tfill = solution.gettfill()
        q0_bybin = solution.getq0bins()[:solution.getopenbins()]
        i_hot_list = np.argsort(q0_bybin)
        i_hot = i_hot_list[-1]
        told = tfill[i_hot]
        kwargs = {'mode': 'hload', 'nmove': len(solution.vlrep[i_hot])}
        t_new, rcl_t = self.get_feasible_tfilli(rcl_t, told - 5.0, **kwargs)
        if t_new:
            neighbor = deepcopy(solution)
            neighbor.edit_tfilli(i_hot, t_new)
            # Adapt Greedy Function
            rcl_t.adapt_changetime(told, t_new, len(neighbor.vlrep[i_hot]))
            # Check if modified solution is nondominated
            neighbor = self.checkandfit(neighbor)
            solution = self.test_domination(solution, neighbor)
        return solution

    def split_bin(self, solution, rcl_t):
        # This function splits the highest capacity bin into two boxes.
        vlrep = solution.getvlrep()
        i = self.getmaxbin(vlrep)
        # Get random place to split bin
        jsplit = random.randrange(1, len(vlrep[i]))
        newbin = list(vlrep[i][jsplit:])
        # Open new bin with feasible time value
        tmin = self.get_box_tmin(newbin)
        kwargs = {'mode': 'hload', 'nmove': len(newbin)}
        t_new, rcl_t = self.get_feasible_tfilli(rcl_t, tmin, **kwargs)
        if t_new:
            tfill = solution.gettfill()
            solution.opennewbin(i, newbin[0], round(t_new, 1))
            inew = solution.getopenbins() - 1
            rcl_t.adapt_greedy_function_newbin(t_new, add=0)
            rcl_t.adapt_movebins(tfill[i], t_new)
            if len(newbin) > 1:
                for j in newbin[1:]:
                    solution.moveitem(i, j, inew)
                    rcl_t.adapt_movebins(tfill[i], tfill[inew])
        return solution, rcl_t

    def cookie_swap(self, k, searchfrom):
        # This function selects two random bins and tries to swap cookies between
        # them. If unsuccessful, it splits the highest capacity bin.
        u = searchfrom.getfits()
        r, rcl_t = self.getresiduals(searchfrom.getvlrep(), searchfrom.gettfill())
        copy = deepcopy(searchfrom)
        for s in range(searchfrom.openbins):
            mode = random.choice(['random', 'moveheat', 'movelate'])
            i1, i2 = self.select_two_bins(copy, mode)
            if not i2:
                newsol, rcl_t = self.split_bin(copy, rcl_t)
            else:
                kwargs = {'i1': i1, 'i2': i2, 'mode': mode}
                newsol, rcl_t = self.perform_cookie_swap(copy, rcl_t, **kwargs)
            # Will return None if it's dominated by vector u
            nondominated = self.check4nondomination(u, newsol)
            k += 1
            if nondominated:
                return k, newsol, rcl_t
        # If a nondominated solution wasn't found, return nothing
        return k, None, rcl_t

    def perform_cookie_swap(self, solution, rcl_t, i1, i2, mode):
        # This function performs the part swap between box i1 and i2
        tfill = solution.gettfill()
        vlrep = solution.getvlrep()
        # Get cookies to swap
        bini1_options = [j for j in vlrep[i1] if self.cookies.get(j).getbatch()
                         * self.moop.tbatch < tfill[i2]]
        bini2_options = [j for j in vlrep[i2] if self.cookies.get(j).getbatch()
                         * self.moop.tbatch < tfill[i1]]
        if mode == 'moveheat':
            j1 = bini1_options[-1]
            j2 = bini2_options[0]
        else:
            j1 = random.choice(bini1_options)
            j2 = random.choice(bini2_options)
        solution.moveitem(i1, j1, i2)
        solution.moveitem(i2, j2, i1)
        return solution, rcl_t

    def part_swap(self, k, searchfrom):
        # This function selects two random bins and tries to swap cookies between
        # them. If unsuccessful, it splits the highest capacity bin.
        u = searchfrom.getfits()
        r, rcl_t = self.getresiduals(searchfrom.getvlrep(), searchfrom.gettfill())
        copy = deepcopy(searchfrom)
        for s in range(searchfrom.openbins):
            mode = random.choice(['random', 'moveheat', 'movelate'])
            i1, i2 = self.select_two_bins(copy, mode)
            if not i2:
                newsol, rcl_t = self.split_bin(copy, rcl_t)
            else:
                kwargs = {'i1': i1, 'i2': i2, 'mode': mode}
                newsol, rcl_t = self.perform_part_swap(copy, rcl_t, **kwargs)
            # Will return None if it's dominated by vector u
            nondominated = self.check4nondomination(u, newsol)
            k += 1
            if nondominated:
                return k, newsol, rcl_t
        # If a nondominated solution wasn't found, return nothing
        return k, None, rcl_t

    def perform_part_swap(self, solution, rcl_t, i1, i2, mode):
        # This function performs the part swap between box i1 and i2
        # Get swap points
        if mode == 'moveheat':
            movetobin2, movetobin1 = self.get_heat_swap_sets(solution, i1, i2)
        else:
            movetobin2, movetobin1 = self.get_random_swap_sets(solution, i1, i2)
        if movetobin2:
            kwargs = {'i1': i1, 'movetobin2': movetobin2,
                      'i2': i2, 'movetobin1': movetobin1}
            solution, rcl_t = \
                self.make_swap_happen(solution, rcl_t, **kwargs)
        else:
            solution, rcl_t = self.split_bin(solution, rcl_t)
        return solution, rcl_t

    def make_swap_happen(self, solution, rcl_t, i1, movetobin2, i2, movetobin1):
        # This function swaps a portion of box i1 with box i2
        # potentially fix this: adapt rcl_t all at once instead of cookie by cookie
        tfill = solution.gettfill()
        for j in movetobin2:
            solution.moveitem(i1, j, i2)
            rcl_t.adapt_movebins(tfill[i1], tfill[i2])
        for j in movetobin1:
            solution.moveitem(i2, j, i1)
            rcl_t.adapt_movebins(tfill[i2], tfill[i1])
        return solution, rcl_t

    def get_heat_swap_sets(self, solution, i1, i2):
        # This function returns sets of cookies meant to reduce overall heat
        # between boxes
        vlrep = solution.getvlrep()
        tfill = solution.gettfill()
        # Determine eligible cookies
        bini1_options = [j for j in vlrep[i1] if self.cookies.get(j).getbatch()
                         * self.moop.tbatch < tfill[i2]]
        bini2_options = [j for j in vlrep[i2] if self.cookies.get(j).getbatch()
                         * self.moop.tbatch < tfill[i1]]
        # Pick random swap sets
        min_box_fill = min(len(vlrep[i1]), len(vlrep[i2]))
        max_swap = min(len(bini1_options), len(bini2_options), min_box_fill - 1)
        swap_number = random.randint(1, max_swap)
        movetobin2 = bini1_options[-swap_number:]
        movetobin1 = bini2_options[:swap_number]
        return movetobin2, movetobin1

    def get_random_swap_sets(self, solution, i1, i2):
        # This function returns a random set of cookies to swap between boxes.
        vlrep = solution.getvlrep()
        tfill = solution.gettfill()
        # Determine eligible cookies
        bini1_options = [j for j in vlrep[i1] if self.cookies.get(j).getbatch()
                         * self.moop.tbatch < tfill[i2]]
        bini2_options = [j for j in vlrep[i2] if self.cookies.get(j).getbatch()
                         * self.moop.tbatch < tfill[i1]]
        # Pick random swap sets
        min_box_fill = min(len(vlrep[i1]), len(vlrep[i2]))
        max_swap = min(len(bini1_options), len(bini2_options), min_box_fill - 1)
        swap_number = random.randint(1, max_swap)
        movetobin2 = random.sample(bini1_options, swap_number)
        movetobin1 = random.sample(bini2_options, swap_number)
        return movetobin2, movetobin1

    def getpoints_4swap(self, binitems1, t1, binitems2, t2):
        # This function returns two points to perform the swap on
        # Retrieve boolean lists
        bool1 = self.moop.packatt(binitems1, t2)
        bool2 = self.moop.packatt(binitems2, t1)
        p1 = self.get_swap_point(bool1)
        p2 = self.get_swap_point(bool2)
        # If no swap point, return false
        if not p1 or not p2:
            return None, None
        # Check for capacity violations
        newbin1 = binitems1[:p1] + binitems2[p2:]
        if len(newbin1) > self.moop.boxcap:
            p2 = self.get_new_swap_point(binitems1, p1, binitems2, bool2)
        newbin2 = binitems2[:p2] + binitems1[p1:]
        if len(newbin2) > self.moop.boxcap:
            p1 = self.get_new_swap_point(binitems2, p2, binitems1, bool1)
        # Return the lists of cookies to be swapped
        movetobin2 = list(binitems1[p1:])
        movetobin1 = list(binitems2[p2:])
        return movetobin2, movetobin1

    def get_swap_point(self, booli):
        # This function finds a feasible point to swap with another box
        # Find starting point for bin i
        starti = self.findstartforswap(booli)
        if starti == len(booli):
            return False
        else:
            pi = random.randrange(starti, len(booli))
            return pi

    def get_new_swap_point(self, bin_into, p1, bin_outta, bool_outta):
        # This function finds a swap point that won't violate bin_into's capacity
        can_accept = self.moop.boxcap - len(bin_into[:p1])
        p2 = self.get_swap_point(bool_outta)
        kappa = 10
        while len(bin_outta[p2:]) > can_accept:
            # If can't find point, only swap one item
            if kappa == 10:
                return len(bin_outta) - 1
            p2 = self.get_swap_point(bool_outta)
        return p2

    def findstartforswap(self, boollist):
        # This function returns the index after which all values are True
        start = 1
        for k in range(len(boollist) - 1, 0, -1):
            if boollist[k] is False:
                start = k + 1
                return start
        return start

    def move_cookies(self, k, searchfrom):
        # This function selects two random bins and tries to move cookies between
        # them. If unsuccessful, it splits the highest capacity bin.
        u = searchfrom.getfits()
        r, rcl_t = self.getresiduals(searchfrom.getvlrep(), searchfrom.gettfill())
        copy = deepcopy(searchfrom)
        for s in range(searchfrom.openbins):
            mode = random.choice(['moveheat', 'movelate'])
            i1, i2 = self.get_hot_empty_bins(copy, mode)
            if i2 == None or len(copy.vlrep[i2]) == self.moop.boxcap:
                newsol, rcl_t = self.split_bin(copy, rcl_t)
            else:
                kwargs = {'i1': i1, 'i2': i2, 'mode': mode}
                newsol, rcl_t = self.perform_cookie_move(copy, rcl_t, **kwargs)
            # Will return None if it's dominated by vector u
            nondominated = self.check4nondomination(u, newsol)
            k += 1
            if nondominated:
                return k, newsol, rcl_t
        # If a nondominated solution wasn't found, return nothing
        return k, None, rcl_t

    def perform_cookie_move(self, solution, rcl_t, i1, i2, mode):
        # This function performs the move of one cookie from box i1 to i2
        tfill = solution.gettfill()
        vlrep = solution.getvlrep()
        # Get cookies to swap
        bini1_options = [j for j in vlrep[i1] if self.cookies.get(j).getbatch()
                         * self.moop.tbatch < tfill[i2]]
        empty_space = self.moop.boxcap - len(vlrep[i2])
        max_move = min(empty_space, empty_space // 2 + 1, len(bini1_options))
        nmove = random.randint(1, max_move)
        for k in range(nmove):
            j1 = bini1_options[-1 - k]
            solution.moveitem(i1, j1, i2)
        return solution, rcl_t

    def select_two_bins(self, solution, mode):
        # This module selects two bins for swap using specified function
        vlrep = solution.getvlrep()
        tfill = solution.gettfill()
        if mode == 'moveheat':
            i1, i2 = self.get_hot_cold_bins(vlrep, tfill, solution.getq0bins())
        elif mode == 'movelate':
            i1, i2 = self.get_hot_cold_bins(vlrep, tfill, solution.gettavail())
        else:
            # Pick random bins
            i1, i2 = self.get_two_random_bins(vlrep, tfill)
        return i1, i2

    def get_hot_cold_bins(self, vlrep, tfill, characteristic):
        # This function returns the indices of the hottest bin and the coldest
        # bin that are compatible
        m = len(vlrep)              # number of open bins
        ilist_hot = np.argsort(characteristic[:m])
        for kh in range(m):
            i_hot = ilist_hot[-1 - kh]
            for kc in range(m - kh):
                i_cold = ilist_hot[kc]
                if i_hot != i_cold:
                    compatible = self.good_match(vlrep, tfill, i_hot, i_cold)
                    if compatible:
                        return i_hot, i_cold
        return None, None

    def get_hot_empty_bins(self, solution, mode):
        # This function returns the indices of the hottest bin compatible with
        # the emptiest bin
        m = solution.getopenbins()
        vlrep = solution.getvlrep()
        tfill = solution.gettfill()
        i2 = self.getminbin(vlrep)
        if mode == 'moveheat':
            ilist_hot = np.argsort(solution.getq0bins()[:m])
        else:
            ilist_hot = np.argsort(solution.gettavail()[:m])
        for k in range(m):
            i_hot = ilist_hot[-1 - k]
            compatible = self.good_match(vlrep, tfill, i_hot, i2,
                                         ignore_length=True)
            if compatible:
                return i_hot, i2
        return None, None

    def get_two_random_bins(self, vlrep, tfill):
        # This function returns two individual random bins that can swap cookies
        bin_pairs = list(combinations(range(len(vlrep)), 2))
        for bp in range(len(bin_pairs)):
            i1, i2 = random.choice(bin_pairs)
            can_swap = self.good_match(vlrep, tfill, i1, i2)
            if can_swap:
                return i1, i2
        return None, None

    def good_match(self, vlrep, tfill, i1, i2, ignore_length=False):
        # This function returns True if i1 and i2 are a good match for swapping
        # and False if they are a bad match
        if i1 == i2:
            return False
        if not ignore_length:
            if len(vlrep[i1]) <= 1 or len(vlrep[i2]) <= 1:
                return False
        list1 = [j for j in vlrep[i1] if self.cookies.get(j).getbatch()
                 * self.moop.tbatch < tfill[i2]]
        if not list1:
            return False
        list2 = [j for j in vlrep[i2] if self.cookies.get(j).getbatch()
                 * self.moop.tbatch < tfill[i1]]
        if not list2:
            return False
        # If made it past conditions, return True
        return True

    def getrandombin(self, vlrep):
        # This function returns a random bin with more than one item in it
        bins = range(len(vlrep))
        bini = random.choice(bins)
        while len(vlrep[bini]) <= 1:
            bini = random.choice(bins)
        return bini

    def getrandsecondbin(self, i1, vlrep, tfill):
        # This function returns a second random bin that is not
        # bin i1 and that items in bin i1 can be moved to
        i2 = random.choice(range(len(vlrep)))
        kappa = 1
        while not self.good_match(vlrep, tfill, i1, i2):
            if kappa == len(vlrep):
                return None
            i2 = random.choice(range(len(vlrep)))
            kappa += 1
        return i2

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
        minbin = np.argmin(bincapacity)
        return minbin

    def getresiduals(self, vlrep, tfill):
        # This function calculates the residual matrix associated with a given
        # dynamic bin packing loading. The first column represents the open box
        # capacities, and the second column represents the maximum number of
        # cookies that can be added to the cooling rack right before tfill_i
        coolrack = self.moop.coolrack
        r = np.zeros((self.n, 2), dtype=np.int)
        # Set box capacity residuals
        for i in range(len(vlrep)):
            r[i, 0] = self.moop.boxcap - len(vlrep[i])
            r[i, 1] = coolrack
        # Set cooling rack capacity residuals
        n_b = self.n // self.moop.nbatches
        rcl_t = RCLtime(coolrack, self.moop.fillcap, n_b,
                        self.moop.tbatch, self.moop.nbatches)
        r[:len(vlrep), 1] = rcl_t.initialize_withtfill(len(vlrep), vlrep, tfill)
        return r, rcl_t

    def update_spaceresiduals(self, r, i, inew):
        # This function updates the space residual r after a cookie moves
        # from box i to box inew
        # Update r: box capacity
        r[i, 0] += 1
        r[inew, 0] -= 1
        return r

    def check4nondomination(self, u, solution):
        # Check if modified solution is nondominated
        solution = self.checkandfit(solution)
        v = solution.getfits()
        if not dom2(u, v):
            return True
        else:
            return False

    def countonrack(self, t, solution):
        # Cookies from boxes filled after t might be on rack
        vlrep = solution.getvlrep()
        tfill = solution.gettfill()
        timecheckindices = np.where(tfill > t)
        nrackitems = 0
        for i in timecheckindices[0]:
            for j in vlrep[i]:
                onrack = self.moop.rackij(t, tfill[i], self.cookies.get(j))
                nrackitems += onrack
        return nrackitems

    def calclowerbound(self):
        # This function calculates theoretical lower bound for the number of
        # bins. It assumes this is the total number of cookies divided by
        # the box capacity.
        minbins = ceil(float(self.n) / self.moop.boxcap)
        self.lb = int(minbins)

    def getub(self):
        # Returns the upper bound (bin capacity)
        return self.moop.boxcap

    def getcookies(self):
        # Returns the list of items to pack
        return self.cookies

    def getlb(self):
        # Returns the theoretical lower bound
        return self.lb


class NewSolution:
    # This class performs the GRASP creation of a new solution.
    def __init__(self, beta, n, cookies, moop):
        self.beta = beta            # Cardinality restriction
        self.n = int(n)             # Number of cookies to sort
        self.cookies = cookies      # dictionary of item objects
        self.moop = moop            # Multiobjective problem class
        self.m = 0                  # initialize open bins count
        self.r = np.zeros((n, 2))   # Residual capacity matrix
        self.x = np.zeros((n, n), dtype=np.int)
        self.y = np.zeros(n, dtype=np.int)
        self.vlrep = []
        self.tfill = np.zeros(n, dtype=np.float)
        # Initialize restricted candidate list
        n_b = self.n // self.moop.nbatches
        self.rcl_t = RCLtime(moop.coolrack, moop.fillcap, n_b,
                             moop.tbatch, moop.nbatches)

    def make_newsol(self, index, *args):
        # This function takes the solution from generate_newsol and creates
        # a CookieSol instance.
        # Possible args: a newgenes list containing a chromosome representation
        # and a suggested tfill.
        if args:
            self.generate_newsol_from_chromosome(args[0], args[1])
        else:
            self.generate_newsol()
        newsol = solmaker.CookieSol(index, self.x, self.y, self.vlrep, self.tfill)
        return newsol

    def generate_newsol(self):
        # This function generates a new solution from scratch using GRASP
        modes = ['ss', 'hload']     # Modes for retrieving new tfill time
        self.initialize_greedy_tfill()
        self.open_new_bin(0, 0)
        # Set strategy for the loading
        theta_i = random.random()
        for j in range(1, self.n):
            rcl_i = self.get_rcl_bins(theta_i, j)
            i = random.choice(rcl_i)
            if self.y[i] == 0:
                self.tfill[i] = self.get_feasible_tfilli(j, modes)
                self.open_new_bin(i, j)
            else:
                self.vlrep[i].append(j)
                self.r[i, 0] -= 1
                self.rcl_t.adapt_greedy_function_addtobin(self.tfill[i])
                self.r[:self.m, 1] = \
                    self.rcl_t.retrieve_space_by_tfill(self.m, self.tfill)
        self.constructx()

    def generate_newsol_from_chromosome(self, chrom, tfill_suggested):
        # This function generates a new solution based on a given chromosome
        modes = ['ss', 'hload']  # Modes for retrieving new tfill time
        self.initialize_greedy_tfill(*tfill_suggested)
        chrom = self.initialize_first_bin(chrom)
        # Set strategy for the loading
        theta_i = random.random()
        for j in chrom:
            rcl_i = self.get_rcl_bins(theta_i, j)
            i = random.choice(rcl_i)
            if self.y[i] == 0:
                self.tfill[i] = self.pick_tfilli(j, modes, tfill_suggested)
                self.open_new_bin(i, j)
            else:
                self.vlrep[i].append(j)
                self.r[i, 0] -= 1
                self.rcl_t.adapt_greedy_function_addtobin(self.tfill[i])
                self.r[:self.m, 1] = \
                    self.rcl_t.retrieve_space_by_tfill(self.m, self.tfill)
        self.constructx()

    def initialize_greedy_tfill(self, *args):
        # This function initializes t_fill
        # Calculate tfill_0 using inverse cdf and set residual capacity
        if args:
            # args = tfill_suggested
            self.tfill[0] = self.rcl_t.pick_suggested_t(args, self.moop.tbatch)
        else:
            self.tfill[0] = self.rcl_t.get_new_t(self.moop.tbatch)

    def initialize_first_bin(self, chrom):
        # This function finds the first cookie in list chrom that can be packed
        # at tfill[0] and opens the first bin with that cookie
        for j in chrom:
            if self.moop.cookiedonebaking(j, self.tfill[0]):
                self.open_new_bin(0, j)
                chrom.remove(j)
                return chrom
        print('Error: NewSolution picked a time that cannot be filled.')

    def pick_tfilli(self, j, modes, tfill_maybe):
        # This module tries to use one of the time values from tfill
        tmin = self.cookies.get(j).getbatch() * self.moop.tbatch
        # If tmin when coolrack is overfull, find least worst solution
        tk = self.find_t_in_trange(tmin)
        if self.rcl_t.space[tk] <= 0:
            t_new = self.rcl_t.find_least_worst_newt(tmin)
            return t_new
        t_possible = self.get_t_from_oldtfill(tmin, tfill_maybe)
        if t_possible:
            return t_possible
        else:
            # If nothing in tfill_maybe worked, return new value:
            t_new = self.get_feasible_tfilli(j, modes)
            return t_new

    def get_t_from_oldtfill(self, tmin, tfill_maybe):
        # This function returns a feasible time from tfill_maybe
        # First establish tmax based on moving 1 cookie from the rack
        tmax = self.rcl_t.get_tmax(tmin, 1)
        t_options = np.unique(tfill_maybe)
        for i in range(len(t_options)):
            if t_options[i] < tmax:
                # Avoid reusing a value from tfill_maybe
                if t_options[i] not in self.tfill:
                    if self.rcl_t.time_feasible(t_options[i], tmin):
                        return t_options[i]
        return None

    def get_feasible_tfilli(self, j, modes):
        # This function locates a new value for tfill[i] that doesn't violate
        # rack or fill limits
        theta_t = random.randint(0, 1)
        tmin = self.cookies.get(j).getbatch() * self.moop.tbatch
        # Find fill time for box i
        t_new, p_t = self.find_new_time_value(tmin, modes[theta_t])
        kappa = 0                               # Counter to exit loop
        # Check if possible to fill in period
        while self.rcl_t.res_fill[p_t] < 1:
            if kappa == 10:
                return None
            # If not possible, find new time value
            t_new, p_t = self.find_new_time_value(tmin, modes[theta_t])
            kappa += 1
        return t_new

    def find_new_time_value(self, tmin, mode):
        # This module retrieves a new time value and also returns which period
        # it belongs to
        t_new = self.rcl_t.get_new_t(tmin, mode=mode)
        t_t = self.find_t_in_fill_periods(t_new)
        return t_new, t_t

    def find_t_in_fill_periods(self, t):
        # If the new time value is beyond the current fill periods, extend
        while t > self.rcl_t.t_t[-1]:
            self.rcl_t.extend_fill_periods()
        # Find the period containing t_new
        tlist = np.where(t >= np.array(self.rcl_t.t_t))[0]
        return tlist[-1]

    def find_t_in_trange(self, t):
        # If the new time value is beyond the current timeline, extend
        while t > self.rcl_t.trange[-1]:
            self.rcl_t.extend_timeline()
        tklist = np.where(np.array(self.rcl_t.trange) <= t)[0]
        return tklist[-1]

    def get_rcl_bins(self, theta_i, j):
        # This module selects the strategy based on theta_i and returns
        # the corresponding restricted candidate list.
        if theta_i < 0.33:
            # Least loaded strategy
            rcl_i = self.llmove(j)
        elif theta_i < 0.66:
            # Weighted max strategy
            rcl_i = self.wmaxmove(j)
        else:
            # Combo-t strategy
            rcl_i = self.combot_move(j)
        # Return either a new bin or the list found above
        if not rcl_i:
            rcl_i = self.find_alternative_bin(j)
            return rcl_i
        else:
            return rcl_i

    def llmove(self, j):
        # This module performs the sorting for module ll.
        # The goal of this strategy is to balance the loading of the boxes.
        rcl_i = []
        i_rlowtohigh = np.argsort(self.r[:self.m, 0], axis=0)
        # Add new bin as an option if others are starting to get full
        if self.r[i_rlowtohigh[-1], 0] <= 0.5 * self.moop.boxcap:
            rcl_i.append(self.m)
        for k in range(self.m):
            # Find open bin with max. residual value, moving backward thru i_rlowtohigh
            lli = i_rlowtohigh[- 1 - k]
            bcookiej = self.cookies.get(j).getbatch()
            pack = packable(self.r[lli, :], bcookiej, self.tfill[lli])
            if pack:
                rcl_i.append(lli)
            if len(rcl_i) == self.beta:
                return rcl_i
        return rcl_i

    def wmaxmove(self, j):
        # This module determines the restricted candidate list by the weighted
        # max strategy. The goal is to keep the number of boxes to a minimum.
        rcl_i = []
        # Gather weights: space on rack / maximum space over time
        maxval = np.max(self.r[:self.m, 1])
        weights = np.zeros(self.m)
        for k in range(self.m):
            weights[k] = self.r[k, 1] / maxval
        # Calculate weighted residuals
        wresidual = np.multiply(self.r[:self.m, 0], weights)
        i_rlowtohigh = np.argsort(wresidual, axis=0)
        for k in range(self.m):
            # Find open bin with min. weighted residual value
            i = i_rlowtohigh[k]
            bcookiej = self.cookies.get(j).getbatch()
            pack = packable(self.r[i, :], bcookiej, self.tfill[i])
            if pack:
                rcl_i.append(i)
            if len(rcl_i) == self.beta // 2:
                return rcl_i
        return rcl_i

    def combot_move(self, j):
        # This module determines the restricted candidate list by the combo-t
        # strategy. The goal is to reduce the maximum time until the boxes
        # can be moved to the store front.
        n_b = self.n // self.moop.nbatches  # Number of cookies per batch
        jmax = j - (j % n_b)                # Max. cookie no. for heat restriction
        rcl_i = []
        i_rlowtohigh = np.argsort(self.r[:self.m, 0], axis=0)
        # Add new bin as an option after all bins meet a minimum level
        if self.r[i_rlowtohigh[-1], 0] <= 0.7 * self.moop.boxcap:
            rcl_i.append(self.m)
        for k in range(self.m):
            # Find open bin with max. residual value
            lli = i_rlowtohigh[- 1 - k]
            otherbatch = [jo for jo in self.vlrep[lli] if jo < jmax]
            # Heat restriction
            if (self.r[lli, 0] <= 0.5 * self.moop.boxcap) & \
                    (len(otherbatch) == 0):
                pass
            else:
                bcookiej = self.cookies.get(j).getbatch()
                pack = packable(self.r[lli, :], bcookiej, self.tfill[lli])
                if pack:
                    rcl_i.append(lli)
                if len(rcl_i) == self.beta:
                    return rcl_i
        return rcl_i

    def open_new_bin(self, i, j):
        # This module opens a new bin i with cookie j
        self.m += 1
        self.y[i] = 1
        self.vlrep.insert(i, [j])
        self.r[i, 0] = self.moop.boxcap - 1
        # Adapt Greedy Function (time)
        self.rcl_t.adapt_greedy_function_newbin(self.tfill[i])
        t_t = self.find_t_in_fill_periods(self.tfill[i])
        self.rcl_t.res_fill[t_t] -= 1
        self.r[:self.m, 1] = self.rcl_t.retrieve_space_by_tfill(self.m, self.tfill)

    def find_alternative_bin(self, j):
        # If tmin when coolrack is overfull, find least worst solution
        tmin = self.cookies.get(j).getbatch() * self.moop.tbatch
        tk = self.find_t_in_trange(tmin)
        if self.rcl_t.space[tk] <= 0:
            # Find least-worst alternative
            options = [i for i in range(self.m)
                       if tmin < self.tfill[i] and self.r[i, 0] > 0]
            if options:
                return options
            else:
                return [self.m]
        else:
            return [self.m]

    def constructx(self):
        # This function transforms the variable length representation into
        # the x-matrix
        for i in range(self.m):
            for j in self.vlrep[i]:
                self.x[i, j] = 1
        checkformismatch(self.x, self.vlrep)


class RCLtime:
    # This class maintains and updates the restricted candidate list for a
    # unique t_fill
    def __init__(self, coolrack, fillcap, n_b, tbatch, nbatches):
        self.coolrack = coolrack                 # Cooling rack capacity
        self.fillcap = fillcap                   # Fill period limit
        self.n_b = n_b                           # Number of cookies in one batch
        self.tbatch = tbatch                     # Time to cook one batch
        self.nbatches = nbatches                 # Number of batches cooked
        # Set the time range, extend one cycle past last pull
        self.trange = [(b + 1) * self.tbatch for b in range(self.nbatches + 1)]
        # Space on the cooling rack as a function of time
        self.space = [self.coolrack - (b + 1) * self.n_b
                      for b in range(self.nbatches)]
        self.space.append(self.space[-1])
        # Include restrictions for period fill limits
        n_period = 2 * (nbatches - 1) + 2
        self.t_t = [self.tbatch * (1.0 + t / 2.0) for t in range(n_period)]
        self.res_fill = [fillcap for _ in range(n_period)]

    def initialize_withtfill(self, m, vlrep, tfill):
        # This function adds the information from vlrep and tfill
        # into the trange and space lists
        # First fix the cooling rack related items
        r2 = np.zeros(m, dtype=np.int)      # Collect residual values
        i_lowtohigh = list(np.argsort(tfill[:m], axis=0))
        for i in i_lowtohigh:
            r2[i] = self.adapt_greedy_function_newbin(tfill[i],
                                                      add=len(vlrep[i]))
        # Then fix the fill period related items
        t_latest = np.amax(tfill)
        while t_latest > self.t_t[-1]:
            self.extend_fill_periods()
        for t in range(len(self.t_t) - 1):
            p_t = [i for i in range(m)
                   if self.t_t[t] <= tfill[i] < self.t_t[t + 1]]
            self.res_fill[t] -= len(p_t)
        return r2

    def pick_suggested_t(self, t_maybe, tmin):
        # This function returns a possible starting t-value, first by trying
        # the suggested t values in t_maybe, and then by finding a feasible one
        for i in range(len(t_maybe)):
            if t_maybe[i] < self.trange[-1]:
                if self.time_feasible(t_maybe[i], tmin):
                    return t_maybe[i]
        t_new = self.get_new_t(tmin)
        return t_new

    def time_feasible(self, t, tmin):
        # This function checks if time t is feasible to open a new bin
        if t < tmin:
            return False
        while self.trange[-1] < t:
            self.extend_timeline()
        tk = self.find_t_in_timeline(t)
        # To be feasible, the cooling rack cannot be overcrowded
        if self.space[tk] > 0:
            return self.time_period_feasible(t)
        # If overcrowded, return False
        return False

    def time_period_feasible(self, t):
        # This module determines if time value t is valid within period fill
        # limit constraints.
        if t < self.t_t[0]:
            return False
        ttlist = np.where(np.array(self.t_t) <= t)[0]
        # The number of boxes filled during the period < limit
        if self.res_fill[ttlist[-1]] > 0:
            return True
        else:
            return False

    def get_new_t(self, tmin, mode='ss', nmove=1, told=None):
        # This function returns a random time on the cumulative
        # distribution function of space(trange) greater than tmin
        t = 0
        tmax = self.get_tmax(tmin, nmove)
        dist = self.retrieve_pdensityfunction(mode)
        c_min = dist.cdf(tmin)
        c_max = dist.cdf(tmax)
        if c_min == c_max:
            return None
        k = 0
        while round(t) <= tmin or round(t) >= tmax:
            rannum = random.uniform(c_min, c_max)
            t = dist.ppf(rannum)
            k += 1
            if k == 10:
                return None
        return round(t)

    def retrieve_pdensityfunction(self, mode):
        # This function returns the needed pdf
        if mode == 'hload':
            dist = PiecewiseLinearPDF(self.trange, self.space)
        else:
            dist = PiecewisePDF(self.trange, self.space)
        return dist

    def find_least_worst_newt(self, tmin):
        # This function returns the least worst time for a box to be opened
        # based on tmin.
        tklist = np.where(np.array(self.trange) >= tmin)[0]
        max_space = self.space[tklist[0]]
        tmax = self.get_tmax(tmin, max_space)
        t_new = random.uniform(tmin + 1, tmax)
        kappa = 0
        while not self.time_period_feasible(t_new):
            if kappa == 10:
                return tmin + 1.0
            t_new = random.uniform(tmin + 1, tmax)
            kappa += 1
        return round(t_new)

    def get_tmax(self, tmin, nmove):
        # This function determines if the get_new_t function needs to limit its
        # search to a max. value. If not, it returns the last trange value.
        tklist = np.where(np.array(self.trange) > tmin)[0]
        for tk in tklist:
            if self.space[tk] - nmove <= 0:
                return self.trange[tk]
        # If did not find t_max, and enough space at end of timeline, extend
        if self.space[-1] >= nmove:
            self.extend_timeline()
        return self.trange[-1]

    def adapt_greedy_function_newbin(self, t, add=1):
        # This function updates the space and trange lists after a new bin is
        # opened, add is the space being opened by # of cookies being removed
        # If t is larger than the range, add it on to the end
        if t > self.trange[-1]:
            self.trange.append(t)
            self.space.append(self.space[-1])
            self.update_space(-1, add=add)
            return self.space[-1]
        # If the new t is the same as the last t in trange, extend it by some
        elif t == self.trange[-1]:
            self.update_space(-1, add=add)
            self.extend_timeline()
            return self.space[-2]
        else:
            ilist = np.where(np.array(self.trange) >= t)[0]
            if t == self.trange[ilist[0]]:
                start = ilist[0]
            else:
                self.trange.insert(ilist[0], t)
                self.space.insert(ilist[0], self.space[ilist[0] - 1] + add)
                start = ilist[0] + 1
            for tk in range(start, len(self.space)):
                self.update_space(tk, add=add)
            return self.space[ilist[0]]

    def adapt_greedy_function_addtobin(self, t):
        # This function updates the space and trange lists after a cookie is
        # added to a box and removed from the cooling rack at time t
        tklist = np.where(np.array(self.trange) >= t)[0]
        for tk in tklist:
            self.update_space(tk)
        return self.space[tklist[0]]

    def adapt_movebins(self, t1, t2):
        # This function updates the space list after a cookie is moved from
        # the box filled at t1 to the one filled at t2
        tklist1 = np.where(np.array(self.trange) >= t1)[0]
        tklist2 = np.where(np.array(self.trange) >= t2)[0]
        tklist = np.setxor1d(tklist1, tklist2)
        if t1 == t2:
            return self.space[tklist1[0]], self.space[tklist1[0]]
        elif t1 < t2:
            for tk in tklist:
                self.update_space(tk, add=-1)
        else:
            for tk in tklist:
                self.update_space(tk)
        return self.space[tklist1[0]], self.space[tklist2[0]]

    def adapt_changetime(self, told, tnew, nmove):
        # This function updates the trange and space lists to account for a bin
        # being filled at tnew instead of told.
        # nmove is the size of the box being changed
        while tnew > self.trange[-1]:
            self.extend_timeline()
        tklist1 = np.where(np.array(self.trange) >= told)[0]
        tklist2 = np.where(np.array(self.trange) >= tnew)[0]
        tklist = np.setxor1d(tklist1, tklist2)
        if told < tnew:
            for tk in tklist:
                self.update_space(tk, add=-nmove)
        else:
            for tk in tklist:
                self.update_space(tk, add=nmove)
        self.trange.insert(tklist2[0], tnew)
        self.space.insert(tklist2[0], self.space[tklist2[0] - 1] + nmove)
        return self.space

    def update_space(self, tk, add=1):
        # This function updates the space list at time tk, assuming one cookie
        # was removed from the cooling rack
        self.space[tk] += add
        if self.space[tk] > self.coolrack:
            self.space[tk] = self.coolrack

    def retrieve_space_by_tfill(self, m, tfill):
        # This function returns the space residuals matching tfill
        r2 = np.zeros(m, dtype=np.int)  # Collect residual values
        for i in range(m):
            ilist = np.where(np.array(self.trange) == tfill[i])[0]
            r2[i] = self.space[ilist[0]]
        return r2

    def find_t_in_timeline(self, t):
        tklist = np.where(np.array(self.trange) > t)[0]
        tk = tklist[0] - 1
        return tk

    def extend_timeline(self):
        # This function extends trange by one batch time period.
        new_tlast = self.trange[-1] + 0.5 * self.tbatch
        self.trange.append(new_tlast)
        self.space.append(self.space[-1])

    def extend_fill_periods(self):
        # This function extends t_t by one period
        self.t_t.append(self.t_t[-1] + 0.5 * self.tbatch)
        self.res_fill.append(self.fillcap)


class PiecewisePDF:
    # This class defines a piecewise function along with its pdf and cdf
    def __init__(self, trange, space):
        self.tchunk = np.ediff1d(trange)
        space_array = np.array(space)
        for tk in range(len(space_array)):
            if space_array[tk] < 0.0:
                space_array[tk] = 0.0
        area_chunks = np.multiply(self.tchunk, space_array[:-1])
        area_total = np.sum(area_chunks)
        self.tk = np.array(trange)                                   # time range for distribution
        self.pk = space_array / float(area_total)                    # probability at tk
        self.ck = np.cumsum(np.multiply(self.pk[:-1], self.tchunk))  # cumulative probability
        self.ck = np.insert(self.ck, 0, 0.0)

    def pdf(self, t):
        # This function returns the probability at time t
        if t < self.tk[0]:
            return 0.0
        listi = np.where(t < self.tk)
        probt = self.pk[listi[0][0] - 1]
        return probt

    def cdf(self, t):
        # This function returns the cumulative probability of quantile t
        if t < self.tk[0]:
            return 0.0
        i = np.where(t == self.tk)[0]
        if any(i):
            return self.ck[i[0]]
        else:
            ilist = np.where(t < self.tk)[0]
            i1 = ilist[0] - 1
            i2 = ilist[0]
            slope = (self.ck[i2] - self.ck[i1]) / (self.tk[i2] - self.tk[i1])
            p_c = slope * (t - self.tk[i1]) + self.ck[i1]
            return p_c

    def ppf(self, p):
        # This function returns the time associated with percentile p
        # This is the inverse cumulative distribution function.
        i = np.where(p == self.ck)[0]
        if any(i):
            return self.tk[i[0]]
        else:
            ilist = np.where(p < self.ck)[0]
            # Linear function: t = (t_high - t_low)/(c_high - c_low)* (p - c_low) + t_low
            i1 = ilist[0] - 1
            i2 = ilist[0]
            slope = (self.tk[i2] - self.tk[i1]) / (self.ck[i2] - self.ck[i1])
            return slope * (p - self.ck[i1]) + self.tk[i1]


class PiecewiseLinearPDF:
    # This class defines a piecewise function along with its pdf and cdf, with a
    # linear increase in probability over each given time range
    def __init__(self, trange, space):
        self.tk = np.array(trange)              # time range for distribution
        self.space_array = np.array(space)      # space available in each time range
        for tk in range(len(self.space_array)):
            if self.space_array[tk] < 0.0:
                self.space_array[tk] = 0.0
        self.tchunk = np.ediff1d(trange)        # differences between time values
        area_chunks = np.multiply(self.tchunk, self.space_array[:-1])
        self.area_total = float(np.sum(area_chunks))   # total area under the space(t) curve
        self.ck = np.cumsum(np.divide(area_chunks, self.area_total))  # cumulative probability
        self.ck = np.insert(self.ck, 0, 0.0)

    def pdf(self, t):
        # This function returns the probability at time t
        if t < self.tk[0]:
            return 0.0
        listi = np.where(t < self.tk)[0]
        k = listi[0] - 1
        # Linear function: probt = [(2 * space(tk) - 0) / (tk+1 - tk) * (t - tk)] / totalarea
        slope = 2 * (self.space_array[k]/self.area_total)/self.tchunk[k]
        probt = slope * (t - self.tk[k])
        return probt

    def cdf(self, t):
        # This function returns the cumulative probability of quantile t
        if t < self.tk[0]:
            return 0.0
        i = np.where(t == self.tk)[0]
        if any(i):
            return self.ck[i[0]]
        else:
            ilist = np.where(t < self.tk)[0]
            k = ilist[0] - 1        # index for lower boundary of chunk
            slope = 2 * (self.space_array[k] / self.area_total) / self.tchunk[k]
            p_c = slope * (t - self.tk[k]) ** 2 / 2 + self.ck[k]
            return p_c

    def ppf(self, p):
        # This function returns the time associated with percentile p
        # This is the inverse cumulative distribution function.
        i = np.where(p == self.ck)[0]
        if any(i):
            return self.tk[i[0]]
        else:
            ilist = np.where(p < self.ck)[0]
            # Quad function: t = sqrt(2*(p-c_low)/slope) + t_low
            k = ilist[0] - 1
            slope = 2 * (self.space_array[k]/self.area_total)/self.tchunk[k]
            x = sqrt(2 * (p - self.ck[k]) / slope)
            return x + self.tk[k]


def dom2(u, v):
    # Determines if fitness vector u dominates fitness vector v
    # This function assumes a minimization problem.
    # For u to dominate v, every fitness value must be either
    # equal to or less than the value in v AND one fitness value
    # must be less than the one in v
    equaltest = np.allclose(u, v)
    if equaltest is True:
        # If u == v then nondominated
        return False
    # less_equal returns boolean for each element u[i] <= v[i]
    domtest = np.less_equal(u, v)
    return np.all(domtest)


def packable(ri, batch, tfilli):
    # This module checks to see if cookie j can fit inside bin i at time tfilli
    # Capacity constraints
    r1 = ri[0] - 1
    r2 = ri[1] - 1
    # Time constraint: tbatch = 10 min = 600 s
    t_cook = batch * 600
    return r1 >= 0 and r2 >= 0 and t_cook < tfilli


def checkformismatch(x, vlrep, out=sys.stdout):
    # This function identifies if the given solution does not have an x-matrix
    # and a variable length representation that match.
    for i in range(len(vlrep)):
        for j in vlrep[i]:
            if x[i, j] != 1:
                out.write('Error: NewSolution is not coordinated on item', j)


def averageLen(lst):
    # Calculates the average length of lists inside a list, returns integer value
    lengths = [len(i) for i in lst]
    return 0 if len(lengths) == 0 else (int(sum(lengths) / len(lengths)))


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # This function determines if value a and value b are about equal
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


if __name__ == '__main__':
    print('grasp.py needs to be combined with coolcookies.py')