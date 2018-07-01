# tests_grasp.py
#    This file tests grasp.py for errors.
#    Author: Kristina Yancey Spencer

from __future__ import print_function
import unittest
import coolcookies
import grasp
import mooproblem as mop
import numpy as np
import solutions_dynamic as sols
from copy import deepcopy
from mock import Mock
from random import randint, uniform
from StringIO import StringIO


class GRASPTests(unittest.TestCase):

    def setUp(self):
        self.n = 24
        cookies = coolcookies.\
            makeobjects(self.n, 6, '/Users/gelliebeenz/Documents/Python/ObjectiveMethod/'
                                   'TimeDependent/ToyProblem/Cookies24.txt')
        self.moop = mop.MOCookieProblem(self.n, 8, 15, 2, cookies)
        self.bpp = grasp.BPP(self.n, cookies, self.moop)
        # Make stub solution ------------------------------------------------------
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
        y = np.zeros(self.n, dtype=np.int)
        y[:len(vlrep)] = 1
        x = np.zeros((self.n, self.n))
        for i in range(len(vlrep)):
            for j in vlrep[i]:
                x[i, j] = 1
        self.newsol = sols.CookieSol(0, x, y, vlrep, tfill)
        self.newsol = self.bpp.checkandfit(self.newsol)

    def test_generate_newsol(self):
        p, newsol = self.bpp.generate_newsol(0, 0.33, 0.66)

    def test_find_new_tfilli_biggertnew(self):
        mocksol = Mock()
        mocksol.getvlrep.return_value = self.newsol.getvlrep()
        mocksol.gettfill.return_value = self.newsol.gettfill()
        r, rcl_t = self.bpp.getresiduals(self.newsol.getvlrep(),
                                         self.newsol.gettfill())
        mocksol, rcl_t = self.bpp.find_new_tfilli(1, mocksol, rcl_t)
        # Inspect the arguments that solution.edittfilli was called with
        inspect = mocksol.mock_calls[-1]
        name, args, kwargs = inspect
        # Make sure the box number did not change
        self.assertEqual(args[0], 1)
        # Make sure the cooling rack capacity isn't negative
        tklist = np.where(np.array(rcl_t.trange) >= args[1])[0]
        self.assertTrue(rcl_t.space[tklist[0]] >= 0)

    def test_find_new_tfilli_smallertnew(self):
        mocksol = Mock()
        mocksol.getvlrep.return_value = self.newsol.getvlrep()
        mocksol.gettfill.return_value = self.newsol.gettfill()
        r, rcl_t = self.bpp.getresiduals(self.newsol.getvlrep(),
                                         self.newsol.gettfill())
        mocksol, rcl_t = self.bpp.find_new_tfilli(5, mocksol, rcl_t)
        # Inspect the arguments that solution.edittfilli was called with
        inspect = mocksol.mock_calls[-1]
        name, args, kwargs = inspect
        # Make sure the box number did not change
        self.assertEqual(args[0], 5)
        # Make sure the cooling rack capacity isn't negative
        tklist = np.where(np.array(rcl_t.trange) >= args[1])[0]
        self.assertTrue(rcl_t.space[tklist[0]] >= 0)

    def test_get_feasible_tfilli(self):
        vlrep = self.newsol.getvlrep()
        tfill = self.newsol.gettfill()
        r, rcl_t = self.bpp.getresiduals(vlrep, tfill)
        # Check that new time value doesn't violate rack or fill limits
        tmin = self.bpp.get_box_tmin(vlrep[4])
        kwargs = {'mode': 'hload', 'nmove': len(vlrep[4]), 'told': tfill[4]}
        tnew, rcl_t = self.bpp.get_feasible_tfilli(rcl_t, tmin, **kwargs)
        t_p = self.bpp.find_t_in_fill_periods(tnew, rcl_t)
        self.assertGreaterEqual(rcl_t.res_fill[t_p], 0)

    def test_ls1_loading(self):
        copy = deepcopy(self.newsol)
        neighbor, rcl_t = self.bpp.ls1_loading(copy)
        self.assertTrue(len(copy.vlrep) > len(neighbor.getvlrep()))

    def test_lsmove(self):
        vlrep = [[0, 1, 2, 4], [3, 5], [6, 7, 8, 9], [10, 11], [12, 13, 14, 15],
                 [16, 17], [18, 19, 20], [21, 22, 23]]
        tfill = np.zeros(self.n, dtype=np.float)
        tfill_unique = [885.0, 722.0, 1507.0, 1428.0, 2210.0,
                        1958.0, 2764.0, 2509.0]
        for i in range(len(vlrep)):
            tfill[i] = tfill_unique[i]
        y = np.zeros(self.n)
        y[:len(vlrep)] = 1
        r, rcl_t = self.bpp.getresiduals(vlrep, tfill)
        newsol = sols.CookieSol(0, np.zeros((self.n, self.n)), y, vlrep, tfill)
        r, rcl_t, neighbor = self.bpp.lsmove(4, 15, r, rcl_t, newsol)
        # Make sure if new bin was opened, everything happened together
        self.assertEqual(neighbor.openbins, len(neighbor.getvlrep()))
        self.assertEqual(neighbor.y[neighbor.openbins - 1], 1)
        self.assertEqual(neighbor.y[neighbor.openbins], 0)

    def test_ls2_makercl(self):
        vlrep = [[1, 2, 3, 4, 6], [0, 5, 7, 8, 9, 10, 11, 13],
                 [12, 16], [14, 15, 17], [18, 19, 20, 21, 22, 23]]
        # Test to return the beta value
        rcl_j1 = self.bpp.ls2_makercl(1, vlrep)
        self.assertEqual(len(rcl_j1), 5)
        # Test to return the length of the bin minus 1
        rcl_j2 = self.bpp.ls2_makercl(2, vlrep)
        self.assertEqual(len(rcl_j2), 1)

    def test_move_options(self):
        vlrep = [[1, 2, 3, 4, 6], [0, 5, 7, 8, 9, 10, 11, 13],
                 [12, 16], [14, 15, 17], [18, 19, 20], [21, 22, 23]]
        tfill = np.zeros(self.n, dtype=np.float)
        tfill_unique = [1441.0, 1651.0, 1799.0, 2233.0, 2543.0, 2833.0]
        for i in range(6):
            tfill[i] = tfill_unique[i]
        # Test to return two indices (i.e. available space in bins)
        r, rcl_t = self.bpp.getresiduals(vlrep, tfill)
        ilist = self.bpp.move_options(20, 6, r, rcl_t, tfill)
        self.assertEqual(len(ilist), 2)
        # Test to return one index (i.e. no space in bins at right time)
        r[4:, :] = 0
        ilist = self.bpp.move_options(20, 6, r, rcl_t, tfill)

    def test_getresiduals(self):
        vlrep = [[0, 1, 2, 4], [3, 5], [6, 7, 8, 9], [10, 11], [12, 13, 14, 15],
                 [16, 17], [18, 19, 20], [21, 22, 23]]
        tfill = np.zeros(self.n, dtype=np.float)
        tfill_unique = [885.0, 722.0, 1507.0, 1428.0, 2210.0,
                        1958.0, 2764.0, 2509.0]
        for i in range(len(vlrep)):
            tfill[i] = tfill_unique[i]
        r, rcl_t = self.bpp.getresiduals(vlrep, tfill)
        self.assertEqual(r[8, 0], 0)
        self.assertEqual(r[3, 0], 6)
        self.assertEqual(len(rcl_t.trange), 13)
        self.assertEqual(r[3, 1], 11)
        self.assertEqual(r[7, 1], 12)

    def test_update_spaceresiduals(self):
        r = np.zeros((self.n, 2))
        r[0, :] = [4, 15]
        r[1, :] = [6, 11]
        r[2, :] = [4, 15]
        r[3, :] = [6, 11]
        r[4, :] = [4, 15]
        r[5, :] = [6, 11]
        r[6, :] = [4, 15]
        r[7, :] = [6, 11]
        # Test for moving 21 from box 6 to box 7
        r = self.bpp.update_spaceresiduals(r, 6, 7)
        self.assertEqual(r[6, 0], 5)
        self.assertEqual(r[7, 0], 5)

    def test_get_two_random_bins(self):
        tfill = np.zeros(self.n, dtype=np.float)
        tfill[:3] = [1765.,  2381.,  2532.]
        vlrep = [[0, 1, 2, 3, 8, 11, 4, 5], [9, 6, 7, 10, 12, 13, 14, 16],
                 [15, 21, 23, 22, 18, 19, 17, 20]]
        i1, i2 = self.bpp.get_two_random_bins(vlrep, tfill)
        self.assertTrue(i1 in range(3))
        self.assertTrue(i2 in range(3))

    def test_get_hot_cold_bins(self):
        tfill = np.zeros(self.n, dtype=np.float)
        tfill[:3] = [1570.3, 2290.1, 3586]
        q0bins = np.zeros(self.n, dtype=np.float)
        q0bins[:3] = [23.03395164, 23.89092102, 14.58089498]
        vlrep = [[0, 1, 2, 3, 4, 5, 6, 8], [7, 9, 10, 11, 13, 14, 15, 17],
                 [12, 16, 18, 19, 20, 21, 22, 23]]
        i1, i2 = self.bpp.get_hot_cold_bins(vlrep, tfill, q0bins)
        self.assertGreater(q0bins[i1], q0bins[i2])

    def test_count_on_rack(self):
        vlrep = [[1, 2, 3, 4, 6], [0, 5, 7, 8, 9, 10, 11],
                 [12], [13, 14, 15, 17], [16, 22], [18, 19, 20, 21, 23]]
        tfill = np.zeros(self.n, dtype=np.float)
        tfill_unique = [1441.0, 1651.0, 1799.0, 2233.0, 2543.0, 2833.0]
        for i in range(6):
            tfill[i] = tfill_unique[i]
        mocksol = Mock()
        mocksol.gettfill.return_value = tfill
        mocksol.getvlrep.return_value = vlrep
        # Test for rack capacity before more cookies added at 1800
        nrack = self.bpp.countonrack(1799.0, mocksol)
        self.assertEqual(nrack, 0)
        # Test for rack capacity after last batch on rack
        nrack = self.bpp.countonrack(2543.0, mocksol)
        self.assertEqual(nrack, 5)


class NewSolutionTests(unittest.TestCase):

    def setUp(self):
        beta = 5
        n = 24
        cookies = coolcookies.\
            makeobjects(n, 6, '/Users/gelliebeenz/Documents/Python/ObjectiveMethod/'
                              'TimeDependent/ToyProblem/Cookies24.txt')
        self.moop = mop.MOCookieProblem(n, 8, 15, 2, cookies)
        self.newbie1 = grasp.NewSolution(beta, n, cookies, self.moop)
        self.newbie2 = grasp.NewSolution(beta, n, cookies, self.moop)
        self.newbie3 = grasp.NewSolution(beta, n, cookies, self.moop)
        self.newbie1.initialize_greedy_tfill()
        self.newbie1.open_new_bin(0, 0)

    def test_make_newsol(self):
        newsol = self.newbie3.make_newsol(3)
        # Constraint 3: Time Viability
        timeviolations = self.moop.timeconstraint(newsol)
        self.assertEqual(timeviolations, [])
        # Constraint 4: Cooling Rack Capacity
        rackviolations = self.moop.rackcapacity(newsol.getx(), newsol.gettfill())
        self.assertEqual(rackviolations, [])
        # Constraint 1: No replacement
        self.moop.noreplacement(newsol.getid(), newsol.getx())
        # Constraint 2: Box Capacity
        self.moop.boxcapacity(newsol.getid(), newsol.getx())
        # x and y in agreement?
        mop.xycheck(newsol.getid(), newsol.getx(), newsol.gety())
        out = StringIO()
        mop.checkformismatch(newsol, out=out)
        output = out.getvalue().strip()
        self.assertEqual(output, '')

    def test_generate_newsol(self):
        self.newbie2.generate_newsol()
        # Check no replacement constraint
        self.moop.noreplacement(0, self.newbie2.x)
        # Check capacity constraints
        self.moop.boxcapacity(0, self.newbie2.x)
        violations = self.moop.rackcapacity(self.newbie2.x, self.newbie2.tfill)
        self.assertEqual(violations, [])
        # Check done baking constraint constraint
        for i in range(self.newbie2.m):
            for j in self.newbie2.vlrep[i]:
                baked = self.moop.cookiedonebaking(j, self.newbie2.tfill[i])
                self.assertTrue(baked)

    def test_initialize_greedy_tfill(self):
        # No more cool rack capacity after batch 3: 1800
        self.assertLessEqual(self.newbie1.tfill[0], 1800)
        self.assertEqual(self.newbie1.r[0, 0], self.newbie1.moop.boxcap - 1)
        self.assertGreater(self.newbie1.r[0, 1], 0)

    def test_get_feasible_tfilli(self):
        modes = ['ss', 'hload']  # Modes for retrieving new tfill time
        t_new = self.newbie1.get_feasible_tfilli(5, modes)
        tk = self.newbie1.find_t_in_fill_periods(t_new)
        self.assertTrue(t_new)
        self.assertLess(self.newbie1.rcl_t.t_t[tk], t_new)
        self.assertLessEqual(t_new, self.newbie1.rcl_t.t_t[tk + 1])

    def test_find_new_time_value(self):
        modes = ['ss', 'hload']  # Modes for retrieving new tfill time
        theta_t = randint(0, 1)
        tmin = self.newbie1.cookies.get(5).getbatch() * self.moop.tbatch
        args = [tmin, modes[theta_t]]
        t_new, p_t = self.newbie1.find_new_time_value(*args)
        self.assertTrue(t_new)
        self.assertLess(self.newbie1.rcl_t.t_t[p_t], t_new)
        self.assertLess(t_new, self.newbie1.rcl_t.t_t[p_t + 1])

    def test_llmove(self):
        rcl_i = self.newbie1.llmove(4)
        self.assertEqual(rcl_i[0], 0)

    def test_wmaxmove(self):
        rcl_i = self.newbie1.wmaxmove(4)
        self.assertEqual(rcl_i[0], 0)

    def test_packable(self):
        bcookie1 = self.moop.cookies.get(1).getbatch()
        bcookie2 = self.moop.cookies.get(20).getbatch()
        self.assertTrue(grasp.packable(self.newbie1.r[0, :], bcookie1, 900))
        self.assertFalse(grasp.packable(self.newbie1.r[0, :], bcookie2, 1200))

    def test_open_new_bin(self):
        self.assertEqual(self.newbie1.m, 1)
        self.assertEqual(len(self.newbie1.vlrep), 1)
        self.assertEqual(self.newbie1.vlrep[0], [0])
        self.assertTrue(self.newbie1.r[0, 1] == 10 or self.newbie1.r[0, 1] == 4)


class NewSolutionfromChromTests(unittest.TestCase):

    def setUp(self):
        beta = 5
        n = 24
        cookies = coolcookies. \
            makeobjects(n, 6, '/Users/gelliebeenz/Documents/Python/ObjectiveMethod/'
                              'TimeDependent/ToyProblem/Cookies24.txt')
        self.moop = mop.MOCookieProblem(n, 8, 15, 2, cookies)
        self.newbie1 = grasp.NewSolution(beta, n, cookies, self.moop)
        self.chrom = [0, 1, 2, 8, 3, 4, 5, 7, 9, 10, 6, 11, 12, 13,
                      15, 16, 14, 17, 18, 19, 21, 20, 22, 23]
        self.tfill_suggested = np.array([1439, 1784, 3818, 3759, 2524, 4277, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0])
        self.tfill_later = np.array([1790, 2371, 3288, 2470, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_generate_newsol_from_chromosome(self):
        self.newbie1.generate_newsol_from_chromosome(self.chrom,
                                                     self.tfill_suggested)
        # Check no replacement constraint
        self.moop.noreplacement(0, self.newbie1.x)
        # Check capacity constraints
        self.moop.boxcapacity(0, self.newbie1.x)
        violations = self.moop.rackcapacity(self.newbie1.x, self.newbie1.tfill)
        self.assertEqual(violations, [])
        # Check done baking constraint constraint
        for i in range(self.newbie1.m):
            for j in self.newbie1.vlrep[i]:
                baked = self.moop.cookiedonebaking(j, self.newbie1.tfill[i])
                self.assertTrue(baked)

    def test_generate_newsol_late_cookies_early_in_chrom(self):
        chrom = [0, 20, 22, 1, 2, 8, 3, 4, 5, 7, 9, 10, 6, 11,
                 12, 13, 15, 16, 14, 17, 18, 19, 21, 23]
        self.newbie1.generate_newsol_from_chromosome(chrom, self.tfill_suggested)
        # Check no replacement constraint
        self.moop.noreplacement(0, self.newbie1.x)
        # Check capacity constraints
        self.moop.boxcapacity(0, self.newbie1.x)
        violations = self.moop.rackcapacity(self.newbie1.x, self.newbie1.tfill)
        self.assertEqual(violations, [])
        # Check done baking constraint constraint
        for i in range(self.newbie1.m):
            for j in self.newbie1.vlrep[i]:
                baked = self.moop.cookiedonebaking(j, self.newbie1.tfill[i])
                self.assertTrue(baked)
        # Check fill period constraint
        mocksol = Mock()
        mocksol.gettfill.return_value = self.newbie1.tfill
        mocksol.gety.return_value = self.newbie1.y
        fviolations = self.moop.period_fill_limit(mocksol)
        self.assertEqual(fviolations, [])

    def test_init_greedy_tfill_with_tmaybe(self):
        # Test for t value from tfill
        args = self.tfill_suggested
        self.newbie1.initialize_greedy_tfill(*args)
        self.assertEqual(self.newbie1.tfill[0], 1439)
        # Test for new value
        args_late = self.tfill_later
        args_late[0] = 1890
        self.newbie1.initialize_greedy_tfill(*args_late)
        self.assertFalse(self.newbie1.tfill[0] in self.tfill_later)

    def test_init_first_bin(self):
        self.newbie1.initialize_greedy_tfill(*self.tfill_suggested)
        chrom = self.newbie1.initialize_first_bin(self.chrom)
        self.assertFalse(0 in chrom)


class RCLtimeTests(unittest.TestCase):

    def setUp(self):
        self.rcl_t = grasp.RCLtime(15, 2, 6, 600, 4)
        vlrep = [[0, 1, 2, 4], [3, 5], [6, 7, 8, 9], [10, 11], [12, 13, 14, 15],
                 [16, 17], [18, 19, 20], [21, 22, 23]]
        tfill = np.zeros(24, dtype=np.float)
        tfill_unique = [885.0, 722.0, 1507.0, 1428.0, 2210.0,
                        1958.0, 2764.0, 2509.0]
        for i in range(8):
            tfill[i] = tfill_unique[i]
        self.r2 = self.rcl_t.initialize_withtfill(8, vlrep, tfill)

    def test_init_space(self):
        rcl_t = grasp.RCLtime(15, 2, 6, 600, 4)
        self.assertEqual(rcl_t.space[0], 9)
        self.assertEqual(rcl_t.space[1], 3)
        self.assertEqual(len(rcl_t.trange), len(rcl_t.space))

    def test_init_withtfill(self):
        self.assertEqual(len(self.rcl_t.trange), 13)
        self.assertEqual(self.rcl_t.space[-1], 15)
        self.assertEqual(len(self.r2), 8)

    def test_get_new_t(self):
        trange = [t for t in range(10)]
        rcl_t = grasp.RCLtime(1, 2, 0, 1, 9)
        t = rcl_t.get_new_t(trange[0])
        self.assertLessEqual(t, 10.0)

    @unittest.skip('Unskip if need to consider precision again.')
    def test_get_new_t_precision(self):
        # This test helps determine what the tolerance should be
        dist = self.rcl_t.retrieve_pdensityfunction('ss')
        c_min = dist.cdf(2400)
        c_max = dist.cdf(2401)
        print(c_min, c_max)

    def test_adapt_greedy_function_newbin(self):
        rcl_t = grasp.RCLtime(15, 2, 6, 600, 4)
        r2 = rcl_t.adapt_greedy_function_newbin(900)
        self.assertEqual(rcl_t.trange[1], 900)
        self.assertEqual(len(rcl_t.space), 6)
        self.assertEqual(r2, 10)

    def test_adapt_greedy_function_addtobin(self):
        rcl_t = grasp.RCLtime(15, 2, 6, 600, 4)
        r2 = rcl_t.adapt_greedy_function_addtobin(1200)
        # Test normal function
        self.assertEqual(len(rcl_t.space), 5)
        self.assertEqual(r2, 4)
        # Test opening new time range
        for i in range(3):
            r2 = rcl_t.adapt_greedy_function_addtobin(1200)
        self.assertEqual(rcl_t.space[2], 1)

    def test_adapt_movebins(self):
        # Test t1 < t2
        r1, r2 = self.rcl_t.adapt_movebins(1507.0, 2210.0)
        self.assertEqual(r1, 14)    # r1 should change
        self.assertEqual(r2, 15)    # r2 should not change
        # Test t1 > t2
        r1, r2 = self.rcl_t.adapt_movebins(1958.0, 1428.0)
        self.assertEqual(r1, 10)    # r1 should not change
        self.assertEqual(r2, 12)    # r2 should change


class PiecewisePDFTests(unittest.TestCase):

    def setUp(self):
        self.trange = [2 * (t + 1) for t in range(10)]
        self.space = [1 for t in range(8)]
        self.space.extend([0, 0])
        self.piecewise = grasp.PiecewisePDF(self.trange, self.space)

    def test_pdf_sum_to_1(self):
        # The pdf should integrate to 1.0
        pdf_total = np.sum(np.multiply(self.piecewise.pk[:-1], self.piecewise.tchunk))
        self.assertEqual(pdf_total, 1.0)

    def test_pdf_positive(self):
        self.assertEqual(self.piecewise.pdf([6]), 0.0625)
        self.assertEqual(self.piecewise.pdf([4.5]), 0.0625)

    def test_pdf_0(self):
        self.assertEqual(self.piecewise.pdf([0]), 0.0)
        self.assertEqual(self.piecewise.pdf([19]), 0.0)

    def test_cdf_0(self):
        c1 = self.piecewise.cdf(0)
        self.assertEqual(c1, 0.0)

    def test_cdf_exact_match(self):
        c2 = self.piecewise.cdf(10.0)
        self.assertEqual(c2, 0.5)

    def test_cdf_in_between(self):
        c3 = self.piecewise.cdf(6.8)
        self.assertAlmostEqual(c3, 0.30, places=1)

    def test_ppf(self):
        # Test exact match
        self.assertEqual(self.piecewise.ppf(0.5), 10.0)
        # Test in-between match
        self.assertEqual(self.piecewise.ppf(0.30), 6.8)

    def test_ppf_beyond_empty_zone(self):
        trange = [b * 300 + 600 for b in range(10)]
        space = [9, 12, 6, 13, 7, 14, 8, 15, 15, 15]
        piece = grasp.PiecewisePDF(trange, space)
        cmin = piece.cdf(3000)
        cmax = piece.cdf(3300)
        ranc = uniform(cmin, cmax)
        t_new = piece.ppf(ranc)
        self.assertTrue(3000 <= t_new <= 3300)


class Piecewise_LinearPDFTests(unittest.TestCase):

    def setUp(self):
        self.trange = [600 * t for t in range(6)]
        self.space = [max(0, min(20, 20 * t) - 6 * t) for t in range(6)]
        self.linear = grasp.PiecewiseLinearPDF(self.trange, self.space)

    def test_pdf_sum_to_1(self):
        # The pdf should integrate to 1.0
        prob_chunk = np.zeros(len(self.trange))
        for k in range(len(self.trange) - 1):
            p1 = self.linear.pdf(self.trange[k+1] - 1)
            prob_chunk[k] = self.linear.tchunk[k] * p1 / 2
        self.assertAlmostEqual(np.sum(prob_chunk), 1.0, places=2)

    def test_pdf_positive(self):
        p2 = self.linear.pdf(1000)
        self.assertAlmostEqual(p2, 0.00130, places=5)
        p4 = self.linear.pdf(1199)
        self.assertAlmostEqual(p4, 0.0019, places=4)

    def test_pdf_0(self):
        p1 = self.linear.pdf(600)
        self.assertEqual(p1, 0)
        p3 = self.linear.pdf(1200)
        self.assertEqual(p3, 0)
        p5 = self.linear.pdf(400)
        self.assertEqual(p5, 0)

    def test_cdf_0(self):
        c1 = self.linear.cdf(600)
        self.assertEqual(c1, 0.0)

    def test_cdf_exact_match(self):
        c2 = self.linear.cdf(1200)
        self.assertEqual(c2, float(8400)/14400)

    def test_cdf_in_between(self):
        c3 = self.linear.cdf(1155.49)
        self.assertAlmostEqual(c3, 0.5, places=1)
        c4 = self.linear.cdf(1683.735)
        self.assertAlmostEqual(c4, 0.8, places=1)

    def test_ppf(self):
        # Test exact match
        self.assertEqual(self.linear.ppf(float(8400)/14400), 1200)
        # Test in-between match
        self.assertAlmostEqual(self.linear.ppf(0.5), 1155.49, places=2)
        self.assertAlmostEqual(self.linear.ppf(0.8), 1683.735, places=3)


if __name__ == '__main__':
    unittest.main()
