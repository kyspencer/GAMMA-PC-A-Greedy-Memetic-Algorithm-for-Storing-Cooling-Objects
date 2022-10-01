# test_nsgaii.py
#    This file tests nsgaii_dynamic.py for errors.
#    Author: Kristina Yancey Spencer

import unittest
from gammapc import coolcookies, binpacking_dynamic as bp, mooproblem
from gammapc import solutions_dynamic as sols
import nsgaii_dynamic as nsgaii
import numpy as np
from random import choice, sample

class GenerationTests(unittest.TestCase):

    def setUp(self):
        cookies = coolcookies.makeobjects(10, 2, 'tests/Cookies10.txt')
        moop = mooproblem.MOCookieProblem(10, 4, 8, 2, cookies)
        self.bpp = bp.BPP(10, moop.boxcap, 8, cookies)
        self.gen = nsgaii.Generation(10, 5, 10, cookies, self.bpp, moop)

    def test_initialp(self):
        self.gen.initialp('tests/seed.txt')
        self.assertEqual(len(self.gen.newgenes), 5)
        self.assertEqual(len(self.gen.newgenes[0][0]), 10)

    def test_getseedvalue(self):
        self.assertTrue(self.gen.getseedvalue('tests/seed.txt') == 3572)

    def test_initialtfill(self):
        tfill = self.gen.initialtfill()
        self.assertEqual(len(tfill), 10)

    def test_fittruncation(self):
        # Create 20 solutions with cd-values
        self.gen.archive = {}
        for m in range(20):
            newsol = sols.MultiSol(m, sample(range(10), 10), np.zeros((10, 1), dtype=int),
                                   self.bpp)
            newsol.updatefitvals([10, 50, 900])
            newsol.updaterank(choice([1, 2]))
            newsol.updatecd(choice(range(10)) / 10)
            self.gen.archive[m] = newsol
        approxset = [m for k, m in self.gen.archive.items() if m.getrank() == 1]
        keys = [k for k, m in self.gen.archive.items() if m.getrank() == 1]
        self.gen.fittruncation(keys, approxset, 5)
        newapproxset = [m for k, m in self.gen.archive.items() if m.getrank() == 1]
        self.assertEqual(len(newapproxset), 5)

    def test_reduce(self):
        # Create 10 solutions
        for m in range(10):
            newsol = sols.MultiSol(m, sample(range(10), 10), np.zeros((10, 1), dtype=int), self.bpp)
            newsol.updaterank(round(m / 2))
            self.gen.archive[m] = newsol
        # Test for normal reduction
        self.gen.reduce(6)
        self.assertEqual(len(self.gen.archive), 6)
        # Recreate 10 solutions with different crowding distance values
        for m in range(6):
            self.gen.archive[m].updaterank(round(m / 5))
            self.gen.archive[m].updatecd(m)
        for m in range(4):
            newsol = sols.MultiSol(m, sample(range(10), 10), np.zeros((10, 1), dtype=int), self.bpp)
            newsol.updaterank(round((m + 6) / 5))
            newsol.updatecd(m + 6)
            self.gen.archive[m + 6] = newsol
        # Test for cd reduction
        self.gen.reduce(6)
        self.assertEqual(len(self.gen.archive), 6)

    def test_approx(self):
        # Create 100 solutions
        self.gen.archive = {}
        ids = sample(range(1000), 100)
        for m in range(100):
            newsol = sols.MultiSol(ids[m], sample(range(10), 10), np.zeros((10, 1), dtype=int), self.bpp)
            newsol = self.gen.moop.calcfeasibility(newsol)
            newsol.x, newsol.y, newsol.tfill = bp.coordarrays(newsol.x, newsol.y, newsol.tfill)
            fits = self.gen.moop.calcfits(newsol)
            newsol.updatefitvals(fits)
            self.gen.archive[newsol.index] = newsol
        ndset = self.gen.finalapproxset()
        self.assertNotEqual(len(ndset), 0)


# class OutputTests(unittest.TestCase):
#
#     def test_savexys(self):
#         ndset = []
#         mock = Mock()
#         mock.getindex.return_value = 1
#         mock.getx.return_value = np.matrix([[1, 0, 0, 0, 0],
#                                             [0, 1, 0, 0, 0],
#                                             [0, 0, 1, 0, 0],
#                                             [0, 0, 0, 1, 0],
#                                             [0, 0, 0, 0, 1]])
#         mock.gety.return_value = np.ones(5)
#         for m in range(10):
#             ndset.append(mock)
#         nsgaii.savexys(ndset, 'tests/')
#         h5f = h5py.File('tests/xymatrices.h5', 'r')
#         gname = h5f.keys()
#         self.assertEqual(gname[0], u'xmatrices')
#         xitems = h5f[gname[0]].items()
#         yitems = h5f[gname[1]].items()
#         self.assertEqual(len(xitems), len(yitems))


if __name__ == '__main__':
    unittest.main()
