# tests_grasp_static.py
#    This file tests grasp_static.py for errors.
#    Author: Kristina Yancey Spencer

from __future__ import print_function
import unittest
import grasp_static as grasp
import mop
import numpy as np
import pickle
from copy import deepcopy
from random import randint, sample
from items import makeitems


class GRASPTests(unittest.TestCase):

    def setUp(self):
        data = 'example/example500.txt'
        self.n = 500
        pop = 50  # members per gen.
        end = 750  # function evaluations
        binc, binh, items = makeitems(data)
        moop = mop.MOproblem(pop, items)
        self.bpp = grasp.BPP(self.n, binc, binh, items, moop)

    def test_ls1_makercl(self):
        with open('example/sol_ls1.pkl', 'rb') as input:
            sol = pickle.load(input)
        r = self.bpp.getresiduals(sol)
        # Find the emptiest bin's index number
        lengths = [len(i) for i in sol.getvlrep()]
        i = np.argmin(np.array(lengths))
        j = sol.getvlrep()[i][0]
        rcl_bins = self.bpp.ls1_makercl(sol.getopenbins(), i, j, r)
        for i in rcl_bins:
            self.assertTrue(grasp.packable(r[i, :], self.bpp.items[j]))

    def test_ls2_makercl(self):
        vlrep = [[] for _ in range(3)]
        vlrep[1] = sample(range(self.n), 15)
        rclj = self.bpp.ls2_makercl(vlrep[1])
        self.assertLessEqual(len(rclj), self.bpp.beta)
        for jr in rclj:
            h = self.bpp.items[jr].getheight()
            for j in vlrep[1]:
                if j not in rclj:
                    self.assertGreaterEqual(h, self.bpp.items[j].getheight())

    def test_reduce_swap_sets(self):
        # Construct bins
        vlrep1 = sample(range(self.n // 2), 4)
        r = np.zeros((3, 2))
        r[1, 0] = self.bpp.wbin - sum([self.bpp.items[j].getweight() for j in vlrep1])
        r[1, 1] = self.bpp.ub - sum([self.bpp.items[j].getheight() for j in vlrep1])
        # Make vlrep1 longer without violating constraints
        for jadd in range(randint(2, 6)):
            j = randint(0, self.n // 2)
            w = self.bpp.items[j].getweight()
            h = self.bpp.items[j].getheight()
            if j not in vlrep1 and r[1, 0] - w > 10 and r[1, 1] - h > 10:
                vlrep1.append(j)
                r[1, 0] -= w
                r[1, 1] -= h
        move_out = sample(vlrep1, 4)
        vlrep2 = sample(range(self.n // 2, self.n), 4)
        r[2, 0] = self.bpp.wbin - sum([self.bpp.items[j].getweight() for j in vlrep2])
        r[2, 1] = self.bpp.ub - sum([self.bpp.items[j].getheight() for j in vlrep2])
        # Test reduce_swap_sets
        move_in = self.bpp.reduce_swap_set(r[1, :], list(vlrep2), move_out, col=1)
        move_height = sum([self.bpp.items[j].getheight() for j in move_in]) - \
                      sum([self.bpp.items[j].getheight() for j in vlrep2])
        self.assertLessEqual(move_height, r[2, 1])


class PartSwapFailTest(unittest.TestCase):

    def setUp(self):
        data = 'GAMMA-PC/SBSBPP500/Experiment01/SBSBPP500_run1.txt'
        self.n = 500
        pop = 50  # members per gen.
        end = 750  # function evaluations
        binc, binh, items = makeitems(data)
        moop = mop.MOproblem(pop, items)
        self.bpp = grasp.BPP(self.n, binc, binh, items, moop)

    def test_part_swap(self):
        with open('example/solbefore.pkl', 'rb') as input:
            solb = pickle.load(input)
        with open('example/sol13283.pkl', 'rb') as input:
            sol3 = pickle.load(input)
        self.bpp.calcfeasibility(solb)

        # Make one change to get to state before error using part
        # swap options
        copy = deepcopy(solb)
        r = self.bpp.getresiduals(solb)
        bin1 = 26
        bin2 = 39
        bini1options = [334, 348, 354, 342, 360]
        bini2options = [478, 479, 495, 497]
        movetobin2, movetobin1 = \
            self.bpp.choose_swap_sets(4, bini1options, bini2options)
        copy, r = self.bpp.make_swap_happen(copy, r, bin1,
                                            movetobin2, bin2, movetobin1)
        # Check to make sure no new error
        copy = self.bpp.checkandfit(copy)
        self.assertLessEqual(sum([self.bpp.items[j].getheight()
                                  for j in copy.getvlrep()[26]]), self.bpp.ub)

        # # Perform part swap that caused error
        # bin1 = 34
        # bin2 = 38
        # bini1options = [459, 476, 456, 464, 436]
        # bini2options = [477, 480, 488, 490, 493, 496, 498, 499]
        # movetobin2, movetobin1 = \
        #     self.bpp.choose_swap_sets(5, r[26], r[39],
        #                               bini1options, bini2options)
        # copy, r = self.bpp.make_swap_happen(copy, r, bin1, movetobin2, bin2, movetobin1)
        # self.bpp.calcfeasibility(copy)

    def test_ls1_loading(self):
        with open('example/sol_ls1.pkl', 'rb') as input:
            sol = pickle.load(input)
        self.bpp.checkandfit(sol)
        r = self.bpp.getresiduals(sol)
        cool1 = self.bpp.ls1_loading(sol)
        cool1 = self.bpp.checkandfit(cool1)
        r = self.bpp.getresiduals(cool1)
        coolneighbor = self.bpp.ls1_loading(cool1)
        coolneighbor = self.bpp.checkandfit(coolneighbor)
        r = self.bpp.getresiduals(coolneighbor)
        self.assertEqual(len(coolneighbor.getvlrep()), coolneighbor.getopenbins())


if __name__ == '__main__':
    unittest.main()
