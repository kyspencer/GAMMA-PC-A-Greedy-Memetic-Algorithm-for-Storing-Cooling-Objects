# test_solutions.py
#    This file tests solutions_dynamic.py for errors.
#    Author: Kristina Yancey Spencer

import unittest
import numpy as np
import solutions_dynamic as sols
from binpacking_dynamic import BPP
from coolcookies import makeobjects
from random import sample, seed

class MultiSolTests(unittest.TestCase):

    def setUp(self):
        storedchrom = np.load('tests/chrom8.npz')
        self.chrom = storedchrom[storedchrom.files[0]]
        storedtfill = np.load('tests/tfill8.npz')
        self.tfill = storedtfill[storedtfill.files[0]]
        cookies = makeobjects(1000, 100, 'tests/Cookies1000.txt')
        self.bpp = BPP(1000, 24, 300, cookies)

    def test_init(self):
        solution = sols.MultiSol(8, self.chrom, self.tfill, self.bpp)
        open = solution.getopenbins()
        self.assertEqual(len(solution.vlrep), open)
        self.assertEqual(np.sum(solution.getx()[open,:]), 0)

    def test_getvlrep(self):
        solution = sols.MultiSol(8, self.chrom, self.tfill, self.bpp)
        vlrep = solution.getvlrep()
        self.assertTrue(vlrep)
        vlrepi = solution.getvlrep(i=0)
        self.assertEqual(vlrep[0], vlrepi)


class CookieSolTests(unittest.TestCase):

    def setUp(self):
        self.n = 24
        # variable length representation
        vlrep = [[0, 1, 2, 4], [3, 5], [6, 7, 8, 9], [10, 11], [12, 13, 14, 15],
                 [16, 17], [18, 19, 20], [21, 22, 23]]
        # t_fill
        tfill = np.zeros(self.n, dtype=np.float)
        tfill_unique = [885.0, 722.0, 1507.0, 1428.0, 2210.0,
                        1958.0, 2764.0, 2509.0]
        for i in range(len(vlrep)):
            tfill[i] = tfill_unique[i]
        # x and y matrices
        y = np.zeros(self.n)
        y[:len(vlrep)] = 1
        x = np.zeros((self.n, self.n))
        for i in range(len(vlrep)):
            for j in vlrep[i]:
                x[i, j] = 1
        self.newsol = sols.CookieSol(0, x, y, vlrep, tfill)

    def test_moveitem(self):
        self.newsol.moveitem(1, 3, 2)
        self.assertEqual(np.sum(self.newsol.x[1, :]), 1)

    def test_closebin(self):
        self.newsol.moveitem(1, 3, 2)
        self.newsol.moveitem(1, 5, 3)
        self.assertEqual(self.newsol.y[7], 0)
        self.assertEqual(self.newsol.tfill[7], 0)
        self.assertEqual(len(self.newsol.getvlrep()), 7)
        self.assertEqual(np.sum(self.newsol.x[7, :]), 0)


class SolutionFunctionTests(unittest.TestCase):

    def test_oldnew(self):
        # Create 5 stub solutions
        archive = {0: sols.Sol(0, range(10), np.zeros((10, 1), dtype=int))}
        for m in range(1, 5):
            newsol = sols.Sol(m, sample(range(10), 10), np.zeros((10, 1), dtype=int))
            archive[m] = newsol
        q = []
        genes = [(range(10), np.zeros((10, 1), dtype=int)),
                 (sample(range(10), 10), np.zeros((10, 1), dtype=int))]
        genes, q = sols.oldnew(archive, q, genes)
        self.assertTrue(q[0] is archive.get(0))


if __name__ == '__main__':
    unittest.main()