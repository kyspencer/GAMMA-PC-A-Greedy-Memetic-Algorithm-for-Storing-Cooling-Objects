# test_moma.py
#    This file tests moma_dynamic.py for errors.
#    Author: Kristina Yancey Spencer

import unittest
from gammapc import coolcookies, binpacking_dynamic as bp, mooproblem
from gammapc import solutions_dynamic as sols
import moma_dynamic as moma
import numpy as np
from random import choice, sample


@unittest.skip('too much output')
class LocalSearchTests(unittest.TestCase):

    def setUp(self):
        storedchrom = np.load('tests/chrom0.npz')
        chrom = storedchrom[storedchrom.files[0]]
        self.n = len(chrom)
        storedtfill = np.load('tests/tfill0.npz')
        tfill = storedtfill[storedtfill.files[0]]
        cookies = coolcookies.makeobjects(self.n, 100, 'tests/Cookies1000.txt')
        bpp = bp.BPP(self.n, 24, 300, cookies)
        self.moop = mooproblem.MOCookieProblem(self.n, 24, 300, 8, cookies)
        self.gen = moma.Generation(self.n, 100, 200, cookies, bpp, self.moop)
        self.m = sols.MultiSol(0, chrom, tfill, bpp)
        self.m = self.gen.moop.calcfeasibility(self.m)

    def test_paretols(self):
        neighbors = self.gen.paretols(self.m, 25, retrieve=True)
        for n in range(len(neighbors)):
            self.assertNotEqual(self.m.getid(), neighbors[n].getid())
            self.assertFalse(np.all(np.equal(self.m.getx(),
                                             neighbors[n].getx())))

    def test_itemswap(self):
        solution = self.gen.itemswap(self.m)
        self.moop.calcfeasibility(solution)

    def test_partswap(self):
        solution = self.gen.partswap(self.m)
        x = solution.getx()
        boxitems = np.sum(x, axis=1)
        for i in range(self.n):
            self.assertLessEqual(boxitems[i], self.moop.boxcap,
                                 msg='Over capacity error')

    def test_splitbin(self):
        solution = self.gen.splitbin(self.m)
        self.moop.calcfeasibility(solution)
        x = solution.getx()
        itemspresent = np.sum(x, axis=0)
        for j in range(self.n):
            self.assertEqual(itemspresent[j], 1, msg='No replacement error')

class GenerationTests(unittest.TestCase):

    def setUp(self):
        n = 24
        cookies = coolcookies.makeobjects(n, 6, 'tests/Cookies24.txt')
        moop = mooproblem.MOCookieProblem(n, 8, 15, 2, cookies)
        self.bpp = bp.BPP(n, 8, 15, cookies)
        self.gen = moma.Generation(n, 5, 10, cookies, self.bpp, moop)

    def test_initialp(self):
        self.gen.initialp('tests/seed.txt')
        self.assertEqual(len(self.gen.newgenes), 5)
        self.assertEqual(len(self.gen.newgenes[0][0]), 24)

    def test_initialtfill(self):
        tfill = self.gen.initialtfill()
        self.assertEqual(len(tfill), 24)

    def test_getpointtoswap(self):
        # Test for success
        vl1 = [7, 8, 10, 11]
        vl2 = [2, 5, 6, 9]
        p1, p2 = self.gen.getpointtoswap(vl1, 1263.0, vl2, 1437.0)
        self.assertTrue(p1 < 4 and p2 < 4)
        # Test for failure
        vl0 = [0, 1, 3, 4]
        fail = self.gen.getpointtoswap(vl0, 1090.0, vl1, 1437.0)
        self.assertFalse(fail)

    def test_findstartforswap(self):
        bool1 = [True, False, True, True, True]
        bool2 = [False, False, True, False]
        bool3 = [True, True, True]
        start1 = self.gen.findstartforswap(bool1)
        start2 = self.gen.findstartforswap(bool2)
        start3 = self.gen.findstartforswap(bool3)
        self.assertEqual(start1, 2)
        self.assertEqual(start2, len(bool2))
        self.assertEqual(start3, 1)

    def test_getrandsecondbin(self):
        vlrep = [[0, 1, 3, 4], [7, 8, 10, 11], [2, 5, 6, 9],
                 [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]]
        tfill = np.zeros(24, dtype=np.float)
        tfill[:5] = [1090.3, 1263.9, 1437.5, 1950.0,  2550.0]
        i2 = self.gen.getrandsecondbin(0, vlrep, tfill, range(5))
        self.assertTrue(i2 in [1, 2, 3, 4])

    def test_getseedvalue(self):
        self.assertEqual(self.gen.getseedvalue('tests/seed.txt'), 3572)

    def test_fittruncation(self):
        # Create 20 solutions with cd-values
        self.gen.archive = {}
        for m in range(20):
            tfill = self.gen.initialtfill()
            newsol = sols.MultiSol(m, sample(range(24), 24), tfill, self.bpp)
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
            tfill = self.gen.initialtfill()
            newsol = sols.MultiSol(m, sample(range(24), 24), tfill, self.bpp)
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
            tfill = self.gen.initialtfill()
            newsol = sols.MultiSol(m, sample(range(24), 24), tfill, self.bpp)
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
            tfill = self.gen.initialtfill()
            newsol = sols.MultiSol(m, sample(range(24), 24), tfill, self.bpp)
            newsol = self.gen.moop.calcfeasibility(newsol)
            newsol = bp.coordarrays(newsol)
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
