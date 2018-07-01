# test_algorithm.py
#    This file tests algorithm.py for errors.
#    Author: Kristina Yancey Spencer

import unittest
from mock import Mock
import coolcookies
import grasp
import h5py
import mooproblem as mop
import algorithm
import numpy as np
import solutions_dynamic as sols
from ga import binsel
from random import choice, sample


class GeneralTests(unittest.TestCase):

    def setUp(self):
        self.n = 24
        self.folder = 'tests/'
        file = 'Cookies24.txt'
        cookies = coolcookies.makeobjects(self.n, 6, self.folder+file)
        moop = mop.MOCookieProblem(self.n, 8, 15, 2, cookies)
        bpp = grasp.BPP(self.n, cookies, moop)
        self.gen = algorithm.Generation(self.n, 50, 750, cookies, bpp, moop)
        # Make sure calling this doesn't create problems, no specific test
        self.gen.initialq(self.folder + 'seed.txt')

    def test_run(self):
        # Algorithm
        while not self.gen.endgen():
            self.gen.rungen()
        keys0 = [k for k, sol in self.gen.archive.items()]
        self.gen.update_cdvalues()
        keys1 = [k for k, sol in self.gen.archive.items()]
        self.assertListEqual(keys0, keys1)
        self.assertGreaterEqual(self.gen.funkeval, 750)

    def test_adaptive_crossover(self):
        # Set up for test
        self.gen.makep()
        if self.gen.g == 1:
            self.gen.q = binsel(self.gen.p, self.gen.pop, 'elitism')
        else:
            self.q = binsel(self.gen.p, self.gen.pop, 'cco')
        # Run adaptive crossover
        self.gen.adaptive_crossover()
        self.assertEqual(len(self.gen.q) + len(self.gen.newgenes), self.gen.pop)

    def test_add_to_archive(self):
        # Initialize archive
        self.gen.makep()
        length = len(self.gen.archive)
        # Test fitness values: no adding
        u1 = [8, 25.0, 6000]
        mocksol = Mock()
        mocksol.getfits.return_value = u1
        mocksol.getid.return_value = 1001
        self.gen.add_to_archive(mocksol)
        self.assertEqual(self.gen.archive.get(1001), None)
        # Test fitness values: adding but not removing
        u2 = [5, 11.0, 6500]
        mocksol.getfits.return_value = u2
        mocksol.getid.return_value = 1002
        self.gen.add_to_archive(mocksol)
        self.assertEqual(len(self.gen.archive), length + 1)
        # Test fitness values: adding and removing
        u3 = [4, 18.0, 5800]
        mocksol.getfits.return_value = u3
        mocksol.getid.return_value = 1003
        self.gen.add_to_archive(mocksol)
        self.assertEqual(len(self.gen.archive), length + 1)

    @unittest.skip('too much output')
    def test_fittruncation(self):
        self.gen.rungen()
        self.gen.rungen()
        self.gen.fittruncation(5)
        self.assertLessEqual(len(self.gen.archive), 5)

    @unittest.skip('too much output')
    def test_cluster_solutions_bybin(self):
        self.gen.rungen()
        self.gen.rungen()
        clusters = self.gen.cluster_solutions_bybin()
        # Assert smallest cluster is greater than 1
        self.assertGreater(len(clusters[-1]), 1)
        # Assert all solutions in a cluster have the same number of bins
        for c in range(len(clusters)):
            c1 = clusters[c][0]
            c2 = clusters[c][-1]
            binsize = self.gen.archive.get(c1).getbins()
            for key in clusters[c]:
                self.assertEqual(self.gen.archive.get(key).getbins(), binsize)
            self.assertLess(self.gen.archive.get(c1).getcd(),
                            self.gen.archive.get(c2).getcd())

    @unittest.skip('too much output')
    def test_sort_cluster_bycd(self):
        self.gen.rungen()
        self.gen.rungen()
        clusters = self.gen.cluster_solutions_bybin()
        c0 = self.gen.sort_cluster_bycd(clusters[0])
        self.assertLess(self.gen.archive.get(c0[0]).getcd(),
                        self.gen.archive.get(c0[-1]).getcd())


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
