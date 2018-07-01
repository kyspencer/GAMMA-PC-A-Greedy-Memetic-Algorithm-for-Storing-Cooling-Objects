# grasp.py
#   This script implements the GRASP heuristic for the dynamic bin packing
#   problem.
#   Author: Kristina Yancey Spencer

from __future__ import print_function
import numpy as np
import random
import solutions as solmaker
import sys
from copy import deepcopy
from itertools import combinations
from math import ceil, sqrt
from operator import attrgetter
import pickle


class BPP:
    # This class groups the bin packing problem information and performs
    # the GRASP operations.
    def __init__(self, n, wbin, hbin, items, moop):
        self.beta = 5               # Cardinality restriction
        self.n = int(n)             # Number of items to sort
        self.wbin = wbin            # Max. bin weight
        self.ub = hbin              # Max. bin height
        self.items = items          # dictionary of item objects
        self.moop = moop            # Multiobjective problem class
        self.lb = 0                 # initialize lower bound
        self.calclowerbound()
        self.scale = np.zeros(2)    # Intialize scaling weights for dp
        self.calcweights()

    def generate_newsol(self, index, t, p_ls1, p_ls2, chrom=None):
        # This module creates an instance of a NewSolution class and
        # performs the generate_newsol procedure
        newbie = NewSolution(self.beta, self.n, self.wbin, self.ub,
                             self.items, self.moop)
        newsol = newbie.make_newsol(t, index, chrom=chrom)
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
        checkformismatch(solution.getx(), solution.getvlrep())
        fits, his, wis = self.moop.calcfits(solution.x, solution.y)
        solution.set_heights(his)
        solution.set_weights(wis)
        solution.updatefitvals(fits)
        self.calcfeasibility(solution)
        return solution

    def calcfeasibility(self, solution):
        # All solutions should be feasible
        m = solution.getindex()
        # Constraint 1
        itemspresent = np.sum(solution.x, axis=0)
        for j in range(self.n):
            if itemspresent[j] != 1:
                raise RuntimeError('Error!  Solution {0} has a physicality '
                                   'error: item {1}'.format(m, j))
        # Constraint: max bin weight
        binweights = solution.get_weights()
        for i in range(self.n):
            if binweights[i] > self.wbin:
                raise RuntimeError('Error!  Solution {0}, bin {1} is over'
                                   'weight: {2}'.format(m, i, binweights[i]))
        # Constraint: max bin height
        binheights = solution.get_heights()
        for i in range(self.n):
            if binheights[i] > self.ub:
                raise RuntimeError('Error!  Solution {0}, bin {1} is over'
                                   'height: {2}'.format(m, i, binheights[i]))

    def test_domination(self, solution, neighbor):
        # This function determines if neighbor dominates solution.
        u = solution.getfits()
        v = neighbor.getfits()
        if dom2(v, u):
            return neighbor
        else:
            return solution

    def ls1(self, p, numls, solution):
        # Heuristic to locate a better solution in terms of the first objective:
        # minimizing the number of bins in use
        k = 0
        neighbors = []
        searchfrom = solution
        while k < numls:
            coolneighbor = self.ls1_loading(searchfrom)
            if coolneighbor:
                k += 1
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
            k, coolneighbor = self.ls2_loading(k, searchfrom)
            if coolneighbor:
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
            k, coolneighbor = self.ls3_loading(k, searchfrom)
            if coolneighbor:
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
        r = self.getresiduals(searchfrom)
        copy = deepcopy(searchfrom)
        half = len(vlrep) // 2
        for iloop in range(half):
            # Find the emptiest bin's index number
            lengths = [len(i) for i in copy.getvlrep()]
            i = np.argmin(np.array(lengths))
            copy, r = self.empty_bin(i, copy, r)
            # If a nondominated solution wasn't found, return nothing
            copy = self.checkandfit(copy)
            v = copy.getfits()
            if not dom2(u, v):
                return copy
        return None

    def empty_bin(self, i, copy, r):
        # This function moves items in box i to other boxes
        for j in list(copy.getvlrep()[i]):
            # Find rcl_bins
            rcl_bins = self.ls1_makercl(copy.getopenbins(), i, j, r)
            if len(rcl_bins) == 0:
                return copy, r
            # Pick random bin
            inew = random.choice(rcl_bins)
            # Move cookie to new bin
            copy.moveitem(i, j, inew)
            r = self.update_spaceresiduals(r, j, i, inew)
        return copy, r

    def ls1_makercl(self, m, i, j, r):
        # This function returns the restricted candidate list for item
        # j to move into. It uses the dot product method
        w = self.items[j].getweight()
        h = self.items[j].getheight()
        # options = [bi for bi in range(m) if r[bi, 0] > w and r[bi, 1] > h]
        # if i in options:
        #     options.remove(i)
        # Form the dot product array
        dparray = np.zeros(m)
        for bi in range(m):
            pack = packable(r[bi, :], self.items[j])
            if pack and bi != i:
                dparray[bi] = self.scale[0] * w * r[bi, 0] + \
                              self.scale[1] * h * r[bi, 1]
        # Max fill
        if len(np.nonzero(dparray)[0]) > self.beta:
            options = list(np.argsort(-dparray)[:self.beta])
            return options
        else:
            options = list(np.nonzero(dparray)[0])
            return options

    def ls2_loading(self, k, searchfrom):
        # This function finds the restricted candidate list and tries to move
        # items toward more favorable configurations to minimize the max height
        u = searchfrom.getfits()
        r = self.getresiduals(searchfrom)
        copy = deepcopy(searchfrom)
        tallbins = np.argsort(searchfrom.get_heights())
        for s in range(searchfrom.openbins):
            i = tallbins[-s - 1]
            vlrep = copy.getvlrep()
            # If there is only one item in the box, no point in moving
            if len(vlrep[i]) < 2:
                return k, None
            rcl_j = self.ls2_makercl(vlrep[i])
            k, newsol = self.search_rclj(k, i, copy, u, r, rcl_j)
            if newsol:
                return k, newsol
        # If a nondominated solution wasn't found, return nothing
        return k, None

    def ls2_makercl(self, vlrepi, byheight=True):
        # This function returns the restricted candidate list for local search 2
        # Goal: reduce bin heights, so return tallest items
        # Restricted candidate list
        binkeys = list(vlrepi)
        # If bin is smaller than cardinality restriction, return all item indices
        if len(vlrepi) <= self.beta:
            return binkeys
        if byheight:
            item_heights = [self.items[j].getheight() for j in vlrepi]
        else:
            item_heights = [self.items[j].getweight() for j in vlrepi]
        binkeys_byheight = [x for (h, x) in sorted(zip(item_heights, binkeys))]
        rcl_j = binkeys_byheight[-self.beta:]
        return rcl_j

    def ls3_loading(self, k, searchfrom):
        # This function finds the restricted candidate list for bin i and tries to
        # move cookies to find a new nondominated solution. If unsuccessful, moves
        # to a new bin
        u = searchfrom.getfits()
        r = self.getresiduals(searchfrom)
        copy = deepcopy(searchfrom)
        heavybins = np.argsort(searchfrom.get_weights(), axis=0)
        for s in range(searchfrom.openbins):
            i = heavybins[-s - 1]
            vlrep = copy.getvlrep()
            # If there is only one item in the box, no point in moving
            if len(vlrep[i]) < 2:
                return k, None
            # Restricted candidate list
            rcl_j = self.ls3_makercl(i, vlrep)
            k, newsol = self.search_rclj(k, i, copy, u, r, rcl_j)
            if newsol:
                return k, newsol
        # If a nondominated solution wasn't found, return nothing
        return k, None

    def ls3_makercl(self, i, vlrep):
        # This function returns the restricted candidate list for local search 3
        # Goal: reduce average bin weights, so return random items
        # Restricted candidate list
        binkeys = list(vlrep[i])
        # If bin is smaller than cardinality restriction, return all item indices
        if len(vlrep[i]) <= self.beta:
            return binkeys
        avglen = averageLen(vlrep)
        nrcl_min = min(len(binkeys) - 1, self.beta)
        nrcl = max(len(binkeys) - avglen, nrcl_min)
        rcl_j = random.sample(binkeys, nrcl)
        return rcl_j

    def search_rclj(self, k, i, solution, u, r, rcl_j):
        # This function moves items into new boxes until either it finds a new
        # nondominated solution or it runs out of candidates from this solution
        for m in range(len(rcl_j)):
            k += 1
            j = random.choice(rcl_j)
            rcl_j.remove(j)
            r, solution = self.lsmove(i, j, r, solution)
            # Check if modified solution is nondominated
            solution = self.checkandfit(solution)
            v = solution.getfits()
            if not dom2(u, v):
                return k, solution
        return k, None

    def lsmove(self, i, j, r, solution):
        # This function determines where item j should move to
        m = solution.getopenbins()
        # Gather bin options and pick new bin for the move
        ilist = self.move_options(j, m, r)
        inew = random.choice(ilist)
        # Open a new bin or move cookie to a new bin
        if inew == m:
            solution.opennewbin(i, j)
            r[inew, 0] = self.wbin - self.items[j].getweight()
            r[inew, 1] = self.ub - self.items[j].getheight()
        else:
            solution.moveitem(i, j, inew)
            r = self.update_spaceresiduals(r, j, i, inew)
        return r, solution

    def move_options(self, j, m, r):
        # This function retrieves a candidate list for moving an item.
        ranres = random.choice([0, 1])
        i_rlowtohigh = np.argsort(r[:m, ranres], axis=0)
        # This module performs the sorting for module ll.
        for i in range(m):
            # Find open bin with max. residual value, moving backward thru i_rlowtohigh
            lsi = i_rlowtohigh[-1 - i]
            pack = packable(r[lsi, :], self.items[j])
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
            k, coolneighbor, = self.select_mutation_operation(k, searchfrom)
            if coolneighbor:
                coolneighbor.updateid(p)
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
            k, coolneighbor = self.move_items(k, searchfrom)
        else:
            rannum = random.random()
            if rannum < 0.50:
                k, coolneighbor = self.part_swap(k, searchfrom)
            else:
                k, coolneighbor = self.item_swap(k, searchfrom)
        return k, coolneighbor

    def split_bin(self, solution):
        # This function splits the highest capacity bin into two boxes.
        vlrep = solution.getvlrep()
        i = self.getmaxbin(vlrep)
        # Get random place to split bin
        jsplit = random.randrange(1, len(vlrep[i]))
        newbin = list(vlrep[i][jsplit:])
        # Open new bin
        solution.opennewbin(i, newbin[0])
        inew = solution.getopenbins() - 1
        if len(newbin) > 1:
            for j in newbin[1:]:
                solution.moveitem(i, j, inew)
        return solution

    def item_swap(self, k, searchfrom):
        # This function selects two random bins and tries to swap items between
        # them. If unsuccessful, it splits the highest capacity bin.
        u = searchfrom.getfits()
        r = self.getresiduals(searchfrom)
        copy = deepcopy(searchfrom)
        for s in range(searchfrom.openbins):
            mode = random.choice(['random', 'movetall', 'moveheavy'])
            i1, i2 = self.select_two_bins(r, copy, mode)
            if not i2:
                newsol = self.split_bin(copy)
            else:
                kwargs = {'i1': i1, 'i2': i2, 'mode': mode}
                newsol, r = self.perform_item_swap(copy, r, **kwargs)
            # Will return None if it's dominated by vector u
            nondominated = self.check4nondomination(u, newsol)
            k += 1
            if nondominated:
                return k, newsol
        # If a nondominated solution wasn't found, return nothing
        return k, None

    def perform_item_swap(self, solution, r, i1, i2, mode):
        # This function performs the part swap between box i1 and i2
        vlrep = solution.getvlrep()
        # Get cookies to swap
        r1 = r[i1, :]
        r2 = r[i2, :]
        bini1_options = [j for j in vlrep[i1] if self.items[j].getweight() <= r2[0]
                         and self.items[j].getheight() <= r2[1]]
        bini2_options = [j for j in vlrep[i2] if self.items[j].getweight() <= r1[0]
                         and self.items[j].getheight() <= r1[1]]
        if mode == 'movetall':
            bini1_options = self.ls2_makercl(bini1_options)
        j1 = random.choice(bini1_options)
        j2 = random.choice(bini2_options)
        solution.moveitem(i1, j1, i2)
        r = self.update_spaceresiduals(r, j1, i1, i2)
        solution.moveitem(i2, j2, i1)
        r = self.update_spaceresiduals(r, j2, i2, i1)
        return solution, r

    def part_swap(self, k, searchfrom):
        # This function selects two random bins and tries to swap cookies between
        # them. If unsuccessful, it splits the highest capacity bin.
        u = searchfrom.getfits()
        r = self.getresiduals(searchfrom)
        copy = deepcopy(searchfrom)
        for s in range(searchfrom.openbins):
            mode = random.choice(['random', 'movetall', 'moveheavy'])
            i1, i2 = self.select_two_bins(r, copy, mode)
            if not i2:
                newsol = self.split_bin(copy)
            else:
                kwargs = {'i1': i1, 'i2': i2, 'mode': mode}
                newsol, r = self.perform_part_swap(copy, r, **kwargs)
            # Will return None if it's dominated by vector u
            nondominated = self.check4nondomination(u, newsol)
            k += 1
            if nondominated:
                return k, newsol
        # If a nondominated solution wasn't found, return nothing
        return k, None

    def perform_part_swap(self, solution, r, i1, i2, mode):
        # This function performs the part swap between box i1 and i2
        # Get swap points
        if mode == 'random':
            swap2, swap1 = self.get_random_swap_sets(solution, r, i1, i2)
        else:
            swap2, swap1 = self.get_wxh_swap_sets(solution, r, i1, i2, mode)
        if swap2:
            kwargs = {'i1': i1, 'movetobin2': swap2,
                      'i2': i2, 'movetobin1': swap1}
            solution, r = self.make_swap_happen(solution, r, **kwargs)
        else:
            solution = self.split_bin(solution)
        return solution, r

    def make_swap_happen(self, solution, r, i1, movetobin2, i2, movetobin1):
        # This function swaps a portion of box i1 with box i2
        nswap = max(len(movetobin2), len(movetobin1))
        for s in range(nswap):
            # Move item from bin i1 to i2
            if s < len(movetobin2):
                j1 = movetobin2[s]
                # Check to make sure its packable
                pack = packable(r[i2], self.items[j1])
                if pack:
                    solution.moveitem(i1, j1, i2)
                    r = self.update_spaceresiduals(r, j1, i1, i2)
            # Move item from bin i2 to i1
            if s < len(movetobin1):
                j2 = movetobin1[s]
                # Check to make sure its packable
                pack = packable(r[i1], self.items[j2])
                if pack:
                    solution.moveitem(i2, j2, i1)
                    r = self.update_spaceresiduals(r, j2, i2, i1)
        return solution, r

    def get_wxh_swap_sets(self, solution, r, i1, i2, mode):
        # This function returns sets of items meant to reduce overall weight
        # between boxes
        vlrep = solution.getvlrep()
        # Determine eligible cookies
        r1 = r[i1, :]
        r2 = r[i2, :]
        bini1_options = [j for j in vlrep[i1] if self.items[j].getweight() <= r2[0]
                         and self.items[j].getheight() <= r2[1]]
        bini2_options = [j for j in vlrep[i2] if self.items[j].getweight() <= r1[0]
                         and self.items[j].getheight() <= r1[1]]
        # Restrict bini1_options
        if mode == 'moveheavy':
            bini1_options = self.ls2_makercl(bini1_options, byheight=False)
        else:
            bini1_options = self.ls2_makercl(bini1_options)
        # Pick random swap sets
        min_box_fill = min(len(vlrep[i1]), len(vlrep[i2]))
        max_swap = min(len(bini1_options), len(bini2_options), min_box_fill - 1)
        movetobin2, movetobin1 = \
            self.choose_swap_sets(max_swap, bini1_options, bini2_options)
        return movetobin2, movetobin1

    def get_random_swap_sets(self, solution, r, i1, i2):
        # This function returns a random set of cookies to swap between boxes.
        vlrep = solution.getvlrep()
        # Determine eligible cookies
        r1 = r[i1, :]
        r2 = r[i2, :]
        bini1_options = [j for j in vlrep[i1] if self.items[j].getweight() <= r2[0]
                         and self.items[j].getheight() <= r2[1]]
        bini2_options = [j for j in vlrep[i2] if self.items[j].getweight() <= r1[0]
                         and self.items[j].getheight() <= r1[1]]
        # Pick random swap sets
        min_box_fill = min(len(vlrep[i1]), len(vlrep[i2]))
        max_swap = min(len(bini1_options), len(bini2_options), min_box_fill - 1)
        movetobin2, movetobin1 = \
            self.choose_swap_sets(max_swap, bini1_options, bini2_options)
        return movetobin2, movetobin1

    def choose_swap_sets(self, max_swap, bin1_options, bin2_options):
        # This function returns the portion of bin 1 and bin 2 to be swapped.
        swap_number = random.randint(1, max_swap)
        movetobin2 = random.sample(bin1_options, swap_number)
        movetobin1 = random.sample(bin2_options, swap_number)
        return movetobin2, movetobin1

    def reduce_swap_set(self, r, move_in, move_out, col=0):
        # This function returns a smaller target list that fits inside
        # constraints
        move_in.remove(random.choice(move_in))
        while self.net_burden(move_in, move_out, col) > r[col]:
            move_in.remove(random.choice(move_in))
        return move_in

    def net_burden(self, move_in, move_out, col):
        # This module calculates the net burden of the swap set move_in
        if col == 0:  # Weight
            positive_burden = sum([self.items[j].getweight() for j in move_in])
            negative_burden = sum([self.items[j].getweight() for j in move_out])
        else:  # Height
            positive_burden = sum([self.items[j].getheight() for j in move_in])
            negative_burden = sum([self.items[j].getheight() for j in move_out])
        return positive_burden - negative_burden

    def move_items(self, k, searchfrom):
        # This function selects two random bins and tries to move cookies between
        # them. If unsuccessful, it splits the highest capacity bin.
        u = searchfrom.getfits()
        r = self.getresiduals(searchfrom)
        copy = deepcopy(searchfrom)
        for s in range(searchfrom.openbins):
            mode = random.choice(['movetall', 'moveheavy'])
            i1, i2 = self.get_big_empty_bins(r, copy, mode)
            if i2 == None or r[i2, 0] == 0:
                newsol = self.split_bin(copy)
            else:
                kwargs = {'i1': i1, 'i2': i2, 'mode': mode}
                newsol, r = self.perform_item_move(copy, r, **kwargs)
            # Will return None if it's dominated by vector u
            nondominated = self.check4nondomination(u, newsol)
            k += 1
            if nondominated:
                return k, newsol
        # If a nondominated solution wasn't found, return nothing
        return k, None

    def perform_item_move(self, solution, r, i1, i2, mode):
        # This function performs the move of one cookie from box i1 to i2
        vlrep = solution.getvlrep()
        # Get cookies to swap
        r2 = r[i2, :]
        bini1_options = [j for j in vlrep[i1] if self.items[j].getweight() <= r2[0]
                         and self.items[j].getheight() <= r2[1]]
        if mode != 'random':
            if mode == 'movetall':
                bini1_options = self.ls2_makercl(bini1_options)
            else:
                bini1_options = self.ls2_makercl(bini1_options, byheight=False)
        max_move = min(self.beta, len(bini1_options))
        nmove = random.randint(1, max_move)
        for k in range(nmove):
            j1 = bini1_options[-1 - k]
            pack = packable(r[i2, :], self.items[j1])
            if pack:
                solution.moveitem(i1, j1, i2)
                r = self.update_spaceresiduals(r, j1, i1, i2)
            else:
                # If pack is false, bin i2 is almost full.
                return solution, r
        return solution, r

    def select_two_bins(self, r, solution, mode):
        # This module selects two bins for swap using specified function
        vlrep = solution.getvlrep()
        if mode == 'movetall':
            i1, i2 = self.get_big_lil_bins(r, vlrep, solution.get_heights())
        elif mode == 'moveheavy':
            i1, i2 = self.get_big_lil_bins(r, vlrep, solution.get_weights())
        else:
            # Pick random bins
            i1, i2 = self.get_two_random_bins(r, vlrep)
        return i1, i2

    def get_big_lil_bins(self, r, vlrep, characteristic):
        # This function returns the indices of the fullest bin and a little
        # bin that are compatible
        m = len(vlrep)              # number of open bins
        ilist_hot = np.argsort(characteristic[:m])
        for kh in range(m):
            i_big = ilist_hot[-1 - kh]
            for kc in range(m - kh):
                i_lil = ilist_hot[kc]
                if i_big != i_lil:
                    compatible = self.good_match(r, vlrep, i_big, i_lil)
                    if compatible:
                        return i_big, i_lil
        return None, None

    def get_big_empty_bins(self, r, solution, mode):
        # This function returns the indices of the fullest bin compatible with
        # the emptiest bin
        m = solution.getopenbins()
        vlrep = solution.getvlrep()
        i2 = self.getminbin(vlrep)
        if mode == 'movetall':
            ilist_hot = np.argsort(solution.get_heights()[:m])
        else:
            ilist_hot = np.argsort(solution.get_weights()[:m])
        for k in range(m):
            i_hot = ilist_hot[-1 - k]
            compatible = self.good_match(r, vlrep, i_hot, i2,
                                         ignore_length=True)
            if compatible:
                return i_hot, i2
        return None, None

    def get_two_random_bins(self, r, vlrep):
        # This function returns two individual random bins that can swap cookies
        bin_pairs = list(combinations(range(len(vlrep)), 2))
        for bp in range(len(bin_pairs)):
            i1, i2 = random.choice(bin_pairs)
            can_swap = self.good_match(r, vlrep, i1, i2)
            if can_swap:
                return i1, i2
        return None, None

    def good_match(self, r, vlrep, i1, i2, ignore_length=False):
        # This function returns True if i1 and i2 are a good match for swapping
        # and False if they are a bad match
        if i1 == i2:
            return False
        if not ignore_length:
            if len(vlrep[i1]) <= 1 or len(vlrep[i2]) <= 1:
                return False
        r1 = r[i1, :]
        r2 = r[i2, :]
        list1 = [j for j in vlrep[i1] if self.items[j].getweight() <= r2[0]
                         and self.items[j].getheight() <= r2[1]]
        if not list1:
            return False
        list2 = [j for j in vlrep[i2] if self.items[j].getweight() <= r1[0]
                         and self.items[j].getheight() <= r1[1]]
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

    def getrandsecondbin(self, r, i1, vlrep):
        # This function returns a second random bin that is not
        # bin i1 and that items in bin i1 can be moved to
        i2 = random.choice(range(len(vlrep)))
        kappa = 1
        while not self.good_match(r, vlrep, i1, i2):
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

    def getresiduals(self, solution):
        # This function calculates the residual matrix associated with a given
        # static bin packing loading. The first column represents the open box
        # weight capacities, and the second column represents the height
        # capacities.
        vlrep = solution.getvlrep()
        r = np.zeros((self.n, 2), dtype=np.int)
        # Set box capacity residuals
        for i in range(solution.getopenbins()):
            bin_weight = sum([self.items[j].getweight() for j in vlrep[i]])
            bin_height = sum([self.items[j].getheight() for j in vlrep[i]])
            r[i, 0] = self.wbin - bin_weight
            r[i, 1] = self.ub - bin_height
        return r

    def update_spaceresiduals(self, r, j, i, inew):
        # This function updates the space residual r after item j moves
        # from box i to box inew
        # Update r: box capacity
        r[i, 0] += self.items[j].getweight()
        r[inew, 0] -= self.items[j].getweight()
        r[i, 1] += self.items[j].getheight()
        r[inew, 1] -= self.items[j].getheight()
        return r

    def check4nondomination(self, u, solution):
        # Check if modified solution is nondominated
        solution = self.checkandfit(solution)
        v = solution.getfits()
        if not dom2(u, v):
            return True
        else:
            return False

    def calclowerbound(self):
        # This function calculates theoretical lower bound for the number of
        # bins. It assumes this is the total number of cookies divided by
        # the box capacity.
        totalc = 0
        for j in range(self.n):
            totalc += self.items[j].getweight()
        minbins = ceil(totalc / self.wbin)
        self.lb = int(minbins)

    def calcweights(self):
        # This function calculates the scaling weights using with the dot
        # product packing function
        # calculate totals
        totalw = sum([self.items[j].getweight() for j in range(self.n)])
        totalh = sum([self.items[j].getheight() for j in range(self.n)])
        # find averages
        avgw = totalw / self.n
        avgh = totalh / self.n
        # set scaling weights
        self.scale[0] = 1.0 / avgw
        self.scale[1] = 1.0 / avgh

    def getwbin(self):
        # Returns the bin weight limit
        return self.wbin

    def getub(self):
        # Returns the bin height limit
        return self.ub

    def getitems(self):
        # Returns the list of items to pack
        return self.items

    def getlb(self):
        # Returns the theoretical lower bound
        return self.lb


class NewSolution:
    # This class performs the GRASP creation of a new solution.
    def __init__(self, beta, n, wbin, ub, items, moop):
        self.beta = beta            # Cardinality restriction
        self.n = int(n)             # Number of cookies to sort
        self.wbin = wbin            # Max. bin weight
        self.ub = ub                # Max. bin height
        self.items = items          # dictionary of item objects
        self.moop = moop            # Multiobjective problem class
        self.m = 0                  # initialize open bins count
        self.r = np.zeros((n, 2))   # Residual capacity matrix
        self.x = np.zeros((n, n), dtype=np.int)
        self.y = np.zeros(n, dtype=np.int)
        self.vlrep = []

    def make_newsol(self, t, index, chrom=None):
        # This function takes the solution from generate_newsol and creates
        # a GAMMASol instance.
        # Possible args: a newgenes list containing a chromosome representation.
        if chrom:
            self.generate_newsol_from_chromosome(list(chrom))
            newsol = solmaker.GAMMASol(index, self.x, self.y, self.vlrep, t,
                                       chrom=chrom)
        else:
            self.generate_newsol()
            newsol = solmaker.GAMMASol(index, self.x, self.y, self.vlrep, t)
        return newsol

    def generate_newsol(self):
        # This function generates a new solution from scratch using GRASP
        self.open_new_bin(0, 0)
        # Set strategy for the loading
        theta_i = random.random()
        for j in range(1, self.n):
            rcl_i = self.get_rcl_bins(theta_i, j)
            i = random.choice(rcl_i)
            if self.y[i] == 0:
                self.open_new_bin(i, j)
            else:
                self.vlrep[i].append(j)
                self.r[i, 0] -= self.items[j].getweight()
                self.r[i, 1] -= self.items[j].getheight()
        self.constructx()

    def generate_newsol_from_chromosome(self, chrom):
        # This function generates a new solution based on a given chromosome
        chrom = self.initialize_first_bin(chrom)
        # Set strategy for the loading
        theta_i = random.random()
        for j in chrom:
            rcl_i = self.get_rcl_bins(theta_i, j)
            i = random.choice(rcl_i)
            if self.y[i] == 0:
                self.open_new_bin(i, j)
            else:
                self.vlrep[i].append(j)
                self.r[i, 0] -= self.items[j].getweight()
                self.r[i, 1] -= self.items[j].getheight()
        self.constructx()

    def initialize_first_bin(self, chrom):
        # This function finds the first item in list chrom and opens the
        # first bin with it
        self.open_new_bin(0, chrom[0])
        chrom.remove(chrom[0])
        return chrom

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
            return [self.m]
        else:
            return rcl_i

    def llmove(self, j):
        # This module performs the sorting for module ll.
        # The goal of this strategy is to balance the loading of the boxes.
        rcl_i = []
        i_rlowtohigh = np.argsort(self.r[:self.m, 0], axis=0)
        # Add new bin as an option if others are starting to get full
        if self.r[i_rlowtohigh[-1], 0] <= 0.5 * self.wbin:
            rcl_i.append(self.m)
        for k in range(self.m):
            # Find open bin with max. residual value, moving backward thru i_rlowtohigh
            lli = i_rlowtohigh[- 1 - k]
            pack = packable(self.r[lli, :], self.items[j])
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
            pack = packable(self.r[i, :], self.items[j])
            if pack:
                rcl_i.append(i)
            if len(rcl_i) == self.beta // 2:
                return rcl_i
        return rcl_i

    def combot_move(self, j):
        # This module determines the restricted candidate list by the combo-t
        # strategy. The goal is to reduce the maximum time until the boxes
        # can be moved to the store front.
        rcl_i = []
        i_rlowtohigh = np.argsort(self.r[:self.m, 0], axis=0)
        # Add new bin as an option after all bins meet a minimum level
        if self.r[i_rlowtohigh[-1], 0] <= 0.7 * self.ub:
            rcl_i.append(self.m)
        for k in range(self.m):
            # Find open bin with max. residual value
            lli = i_rlowtohigh[- 1 - k]
            # Height restriction
            if self.r[lli, 0] <= 0.5 * self.ub:
                pass
            else:
                pack = packable(self.r[lli, :], self.items[j])
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
        self.r[i, 0] = self.wbin - self.items[j].getweight()
        self.r[i, 1] = self.ub - self.items[j].getheight()

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


def packable(ri, item):
    # This module checks to see if cookie j can fit inside bin i at time tfilli
    # Capacity constraints
    r1 = ri[0] - item.getweight()
    r2 = ri[1] - item.getheight()
    return r1 >= 0 and r2 >= 0


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