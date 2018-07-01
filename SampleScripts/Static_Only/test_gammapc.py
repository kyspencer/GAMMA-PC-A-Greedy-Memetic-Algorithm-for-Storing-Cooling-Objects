# tests_grasp_static.py
#    This file tests grasp_static.py for errors.
#    Author: Kristina Yancey Spencer

from __future__ import print_function
import unittest
import gammapc
import grasp_static as grasp
import mop
import numpy as np
import pickle
import solutions as sols
from copy import deepcopy
from random import randint, sample
from items import makeitems


class GAMMAPCTests(unittest.TestCase):

    def setUp(self):
        data = 'GAMMA-PC/SBSBPP500/Experiment01/SBSBPP500_run1.txt'
        self.n = 500
        pop = 50  # members per gen.
        end = 750  # function evaluations
        binc, binh, items = makeitems(data)
        moop = mop.MOproblem(pop, items)
        bpp = grasp.BPP(self.n, binc, binh, items, moop)
        self.gen = gammapc.Generation(self.n, pop, end, items, bpp, moop)

    def test_add_to_archive(self):
        # Set up problem
        self.gen.initialq()
        self.gen.rungen()
        # Make dummy stub solution with known dominant fitvals
        x = np.zeros((self.n, self.n))
        y = np.zeros(self.n)
        vlrep = []
        a = 0
        for i in range(50):
            vlrep.append([i for i in range(a, a + 10)])
            a += 10
        newsol = sols.GAMMASol(self.gen.idnum, x, y, vlrep, self.gen.t)
        fitvals = np.array([100, 481, 283.76])
        newsol.updatefitvals(fitvals)
        # Test if adding to archive will delete anything
        deleted = self.gen.archive[66]
        self.gen.add_to_archive(newsol)
        self.assertFalse(deleted in self.gen.archive)

if __name__ == '__main__':
    unittest.main()
