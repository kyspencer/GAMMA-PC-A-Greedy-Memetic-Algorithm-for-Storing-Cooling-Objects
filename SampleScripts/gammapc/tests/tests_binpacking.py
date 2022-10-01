# tests_binpacking.py
#   This file tests binpacking_dynamic.py for errors.
#   Author: Kristina Yancey Spencer

import numpy as np
import unittest
from copy import copy
from mock import Mock
from random import randint, sample, uniform
from io import StringIO

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from .. import coolcookies, binpacking_dynamic as bp
from ..mooproblem import checkformismatch, MOCookieProblem
from ..solutions_dynamic import MultiSol
from . import stubs  # relative-import the *package* containing the stubs


class BinpackingTests(unittest.TestCase):

    def setUp(self):
        # Mock cookie object
        self.mock = Mock()
        self.mock.getbatch.return_value = 3
        # Class BPP stub
        cookies = coolcookies.makeobjects(24, 6, 'tests/Cookies24.txt')
        self.bpp = bp.BPP(24, 8, 15, cookies)
        # tfill stubs
        self.tfill1 = np.zeros(self.bpp.n, dtype=np.float)
        self.tfill2 = np.zeros(self.bpp.n, dtype=np.float)
        tfill_unique = [800, 900, 950, 1000, 1500, 1800, 2400, 2700]
        tfill_unique2 = [800, 900, 950, 1000, 1050, 1100, 1200, 1300]
        for i in range(len(tfill_unique)):
            self.tfill1[i] = tfill_unique[i]
            self.tfill2[i] = tfill_unique2[i]

    def test_llmove(self):
        # Test for an open bin
        r1 = np.array([0, 4, 7, 14, 3, 17, 3, 5, 0, 0])
        i1 = bp.llmove(10, 8, r1, self.mock, self.tfill1)
        self.assertEqual(i1, 7)
        # Test for all bins full
        r2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        i2 = bp.llmove(10, 8, r2, self.mock, self.tfill1)
        self.assertEqual(i2, 8)
        # Test for all bins too early
        i3 = bp.llmove(10, 8, r1, self.mock, self.tfill2)
        self.assertEqual(i3, 8)

    def test_dpmove(self):
        # Set up
        weights = [1.0 / self.bpp.getub(), 1.0 / self.bpp.rack]  # 1/boxcap, 1/coolrackcap
        r1 = np.array([8, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        t_range, onrack = bp.get_coolrack_variation(self.tfill1, self.bpp)
        # Test for an open bin - with tfill true
        cookie = self.bpp.items.get(0)
        i1a = bp.dpmove(1, r1, cookie, self.tfill1, weights, t_range, onrack)
        self.assertEqual(i1a, 0)
        # Test for an open bin - with tfill false
        i1b = bp.dpmove(1, r1, self.mock, self.tfill1, weights, t_range, onrack)
        self.assertEqual(i1b, 1)
        # Test for all bins full
        r2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        i2 = bp.dpmove(8, r2, self.mock, self.tfill1, weights, t_range, onrack)
        self.assertEqual(i2, 8)
        # Test for all bins too early
        i3 = bp.dpmove(8, r1, self.mock, self.tfill2, weights, t_range, onrack)
        self.assertEqual(i3, 8)

    def test_combo(self):
        chrom = sample(range(self.bpp.n), self.bpp.n)
        x, y, tfill = bp.combo(chrom, self.tfill1, self.bpp)
        x_collapse = np.sum(x, axis=0)
        for j in range(len(x_collapse)):
            self.assertEqual(x_collapse[j], 1)
        self.assertEqual(np.sum(y),
                         np.count_nonzero(np.sum(x, axis=1)))

    def test_get_coolrack_variation(self):
        # This function generates a random tfill matrix.
        tbox0 = uniform(700, 1200)
        fillbins = randint(self.bpp.lb, self.bpp.n)
        # Find tintervals
        tintervals = (600 * (self.bpp.nbatches + 1) - tbox0) / (max(fillbins - 1, 1))
        tfill = np.zeros(self.bpp.n, dtype=np.float)
        for i in range(fillbins):
            tfill[i] = tintervals * i + tbox0
        trange, onrack = bp.get_coolrack_variation(tfill, self.bpp)
        self.assertEqual(len(trange), len(onrack))
        self.assertFalse(0.0 in trange)
        self.assertEqual(onrack[-1], self.bpp.n)

    def test_update_coolrack_variation(self):
        trange, onrack = bp.get_coolrack_variation(self.tfill1, self.bpp)
        # Test moving cookie to established box:
        trange, onrack = bp.update_coolrack_variation(0, self.tfill1, trange, onrack)
        self.assertEqual(onrack[-1], self.bpp.n - 1)
        # Test moving cookie to new box, late tfill
        tfill = copy(self.tfill1)
        tfill[8] = 3300
        trange, onrack = bp.update_coolrack_variation(8, tfill, trange, onrack)
        self.assertEqual(trange[-1], 3300)
        self.assertEqual(onrack[-1], self.bpp.n - 2)
        self.assertEqual(len(trange), len(onrack))
        # Test moving cookie to new box, middle tfill
        tfill[9] = 1350.0
        trange, onrack = bp.update_coolrack_variation(9, tfill, trange, onrack)
        self.assertTrue(1350.0 in trange)
        self.assertEqual(len(trange), len(onrack))

    def test_addtobin(self):
        x = np.zeros((10, 10), dtype=np.int)  # initialize x
        y = np.zeros(10, dtype=np.int)        # initialize y
        # Test for an open bin
        r1 = np.array([0, 4, 7, 14, 3, 17, 3, 5, 0, 0])
        m, x, y, r, tfill1 = bp.addtobin(5, 1, 8, x, y, r1, 24, self.mock, self.tfill1)
        self.assertEqual(x[5, 1], 1)
        self.assertEqual(r[5], 16)
        self.assertEqual(m, 8)
        # Test for all bins full
        r2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        m, x, y, r, tfill1 = bp.addtobin(8, 1, 8, x, y, r2, 24, self.mock, tfill1)
        self.assertEqual(x[8, 1], 1)
        self.assertEqual(r[8], 23)
        self.assertEqual(m, 9)

    def test_initial(self):
        # Test to ensure works for ll
        m1 = int(self.bpp.getlb())
        n, c, x, y, r = bp.initial(m1, self.bpp)
        self.assertEqual(y[m1-1], 1)
        # Test to ensure work for dp
        m2 = 1
        n, c, x, y, r = bp.initial(m2, self.bpp)
        self.assertEqual(y[1], 0)

    def test_packable(self):
        # Test tfill False
        self.assertFalse(bp.packable(5, 1200, self.mock)[1])
        # Test capacity False
        self.assertFalse(bp.packable(0, 2000, self.mock)[0])
        # Test both True
        self.assertTrue(all(bp.packable(5, 2000, self.mock)))

    def test_coordarrays1(self):
        # Set up
        chromstored = np.load('tests/chrom0.npz')
        chromname = chromstored.files
        chrom0 = chromstored[chromname[0]]
        tfillstored = np.load('tests/tfill0.npz')
        tfillname = tfillstored.files
        tfill0 = tfillstored[tfillname[0]]
        cookies = coolcookies.makeobjects(1000, 100, 'tests/Cookies1000.txt')
        bpp = bp.BPP(1000, 24, 300, cookies)
        moop = MOCookieProblem(1000, 24, 300, 8, cookies)
        solution = MultiSol(0, chrom0, tfill0, bpp)
        checkformismatch(solution)
        testviolations = moop.rackcapacity(solution.getx(), solution.gettfill())
        moop.fixcoolingrack(testviolations, solution)
        checkformismatch(solution)
        # Test
        solution = bp.coordarrays(solution)
        out = StringIO()
        checkformismatch(solution, out=out)
        output = out.getvalue().strip()
        self.assertEqual(output, '')

    def test_findrowspace(self):
        y = np.zeros(10, dtype=int)
        for i in range(5):
            y[i] = 1
        # Test return None
        s = bp.findrowspace(10, 5, y)
        self.assertEqual(s, None)
        # Test return a number
        y[8] = 1
        s = bp.findrowspace(10, 5, y)
        self.assertEqual(s, 3)
    #
    # def test_repackitems_newbin(self):
    #     x = np.zeros((5, 5), dtype=np.int)
    #     y = np.zeros(5, dtype=np.int)
    #     # Test open new bin
    #     y[0] = 1
    #     for j in range(5):
    #         x[0, j] = 1
    #     tfill = np.zeros(5, 1)
    #     tfill[0] = 400
    #     binitems1 = [(0, 3), (0, 4)]
    #     x, y, tfill = bp.repackitems(1, x, y, tfill, self.bpp, 750, binitems1)
    #     for i, j in binitems1:
    #         self.assertEqual(x[i, j], 0)
    #     self.assertNotEqual(tfill[1], 0)
    #     # Test move to old bin
    #     y[1] = 1
    #     for j in range(3, 5):
    #         x[0, j] = 0
    #         x[1, j] = 1
    #     tfill[0] = 750
    #     tfill[1] = 850
    #     binitems2 = [(1, 3), (1, 4)]
    #     x, y, tfill = bp.repackitems(2, x, y, tfill, self.bpp, 750, binitems2)
    #     for i, j in binitems2:
    #         self.assertEqual(x[0, j], 1)
    #         self.assertEqual(x[i, j], 0)
    #     self.assertEqual(tfill[2], 0)
    #
    # def test_initializerepack(self):
    #     x = np.zeros((5, 5), dtype=np.int)
    #     y = np.zeros(5, dtype=np.int)
    #     # Test open new bin
    #     y[0] = 1
    #     for j in range(5):
    #         x[0, j] = 1
    #     tfill = np.zeros(5)
    #     tfill[0] = 800
    #     m, c, y1, tfill1, r = bp.initializerepack(1, x, y, tfill, self.bpp, 750)
    #     self.assertEqual(m, 2)
    #     self.assertEqual(tfill1[1], 750)
    #     # Test open old bins
    #     y[1] = 1
    #     for j in range(3, 5):
    #         x[0, j] = 0
    #         x[1, j] = 1
    #     tfill[0] = 750
    #     tfill[1] = 850
    #     m, c, y2, tfill2, r = bp.initializerepack(2, x, y, tfill, self.bpp, 750)
    #     self.assertEqual(m, 2)
    #     self.assertEqual(r[0], 21)

    def test_openonebin(self):
        x = np.zeros((5, 5), dtype=np.int)
        y = np.zeros(5, dtype=np.int)
        # Test open new bin
        y[0] = 1
        for j in range(5):
            x[0, j] = 1
        tfill = np.zeros(5, dtype=np.float)
        tfill[0] = 400
        r = np.zeros(self.bpp.n, dtype=np.int)
        y, tfill, r, m = bp.openonebin(1, y, tfill, r, 24, 750)
        self.assertEqual(tfill[1], 750)
        self.assertEqual(m, 2)


class CoordArraysTests(unittest.TestCase):

    def setUp(self):
        chromstored = np.load('tests/chrom0.npz')
        chromname = chromstored.files
        self.chrom0 = chromstored[chromname[0]]
        tfillstored = np.load('tests/tfill0.npz')
        tfillname = tfillstored.files
        self.tfill0 = tfillstored[tfillname[0]]
        newgenesstored = np.load('tests/newgenes129.npz')
        newgenesfiles = newgenesstored.files
        self.chrom = newgenesstored[newgenesfiles[0]]
        self.tfill = newgenesstored[newgenesfiles[1]]
        cookies = coolcookies.makeobjects(1000, 100, 'tests/Cookies1000.txt')
        self.bpp = bp.BPP(1000, 24, 300, cookies)
        self.moop = MOCookieProblem(1000, 24, 300, 8, cookies)

    def test_missing_bin_in_bpp_make(self):
        x, y, tfill = bp.ed(self.chrom, self.tfill, self.bpp)
        self.assertEqual(np.sum(y),
                         np.count_nonzero(np.sum(x, axis=1)))

    def test_missing_bin_in_y_and_tfill(self):
        solution = MultiSol(130, self.chrom, self.tfill, self.bpp)
        testviolations = self.moop.rackcapacity(solution.getx(), solution.gettfill())
        self.moop.fixcoolingrack(testviolations, solution)
        solution = bp.coordarrays(solution)
        x = solution.getx()
        y = solution.gety()
        vlrep = solution.getvlrep()
        tfill = solution.gettfill()
        self.assertEqual(np.sum(y),
                         np.count_nonzero(np.sum(x, axis=1)))
        self.assertEqual(np.sum(y), solution.getopenbins())
        self.assertEqual(np.sum(y), len(vlrep))
        self.assertEqual(np.sum(x), solution.n)
        self.assertEqual(tfill[len(vlrep)], 0.0)


if __name__ == '__main__':
    unittest.main()
