# tests_misc.py
#   This script trys to determine the cause of errors that don't seem to be
#   contained in one module. It works with Python 3

import unittest
import pickle

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from gammapc.binpacking_dynamic import coordarrays
from ..coolcookies import makeobjects
from gammapc import mooproblem
from . import stubs  # relative-import the *package* containing the stubs


@unittest.skip('Focus on another')
class Tests(unittest.TestCase):

    def setUp(self):
        n = 1000
        with pkg_resources.path(stubs, 'Cookies1000.txt') as cookietext:
            cookies = makeobjects(n, 100, cookietext)
            self.moop = mooproblem.MOCookieProblem(n, 24, 300, 8, cookies)

    def test_coordarrays2(self):
        # Set up
        with open('stubs/solution597.pkl', 'rb') as input:
            solution = pickle.load(input)

        for i in range(solution.openbins):
            print(i, solution.vlrep[i])

        coordarrays(solution)
        mooproblem.checkformismatch(solution)
        fitvals = self.moop.calcfits(solution)
        self.assertEqual(len(fitvals), 3)


class FixTfillLoopTests(unittest.TestCase):

    def setUp(self):
        n = 1000
        batchsize = 100
        boxcap = 24
        rackcap = 300
        fillcap = 8

        with pkg_resources.path(stubs, 'Cookies1000.txt') as cookietext:
            cookies = makeobjects(n, batchsize, cookietext)
            self.moop = mooproblem.MOCookieProblem(n, boxcap, rackcap, fillcap, cookies)

        with pkg_resources.path(stubs, 'solution143.pkl') as solution143:
            with open(solution143, 'rb') as sol143:
                self.solution = pickle.load(sol143)

        self.solution = self.check_sol_for_rack_violations(self.solution)

    @unittest.skip('too much output')
    def test_fix_infinite_loop_fix_tfill(self):
        violations = self.moop.period_fill_limit(self.solution)
        sol = self.moop.fix_tfill(violations, self.solution)
        self.assertEqual(sol.getopenbins(), len(sol.getvlrep()))
        rcl_tfill = self.moop.get_move_restrictions(sol)
        for tk in range(len(rcl_tfill.res_fill)):
            self.assertGreaterEqual(rcl_tfill.res_fill[tk], 0)

    @unittest.skip('too much output')
    def test_fix_infinite_loop_all_earlier_boxes_full(self):
        violations = self.moop.period_fill_limit(self.solution)
        rcl_tfill = self.moop.get_move_restrictions(self.solution)
        sol, rcl_tfill = self.moop.select_fix_mode(0, violations,
                                                   rcl_tfill, self.solution)
        violations = self.moop.period_fill_limit(sol)
        sol, rcl_tfill = self.moop.open_colderbox(violations[0], rcl_tfill, sol)
        inew = sol.getopenbins() - 1
        j = sol.vlrep[inew][0]
        tmin = self.moop.cookies.get(j).getbatch() * 600
        self.assertTrue(rcl_tfill.time_feasible(sol.tfill[inew], tmin))

    @unittest.skip('too much output')
    def test_fix_open_colderbox(self):
        with open('stubs/solution836.pkl', 'rb') as input:
            sol836 = pickle.load(input)

        sol836 = self.check_sol_for_rack_violations(sol836)
        violations = self.moop.period_fill_limit(sol836)
        rcl_tfill = self.moop.get_move_restrictions(sol836)
        for loop in range(3):
            sol836, rcl_tfill = self.moop.select_fix_mode(loop, violations,
                                                          rcl_tfill, sol836)
            violations = self.moop.period_fill_limit(sol836)
        if violations:
            sol836, rcl_tfill = \
                self.moop.open_colderbox(violations[0], rcl_tfill, sol836)
        rviolations = self.moop.rackcapacity(sol836.getx(), sol836.gettfill())
        self.assertListEqual(rviolations, [])

    @unittest.skip('too much output')
    def test_fix_infinite_loop_boxes_not_combining(self):
        with open('stubs/solution555.pkl', 'rb') as input:
            sol555 = pickle.load(input)

        sol555 = self.check_sol_for_rack_violations(sol555)
        violations = self.moop.period_fill_limit(sol555)
        ran_correctly = True
        try:
            sol555 = self.moop.fix_tfill(violations, sol555)
        except KeyboardInterrupt:
            ran_correctly = False
        self.assertTrue(ran_correctly)

    @unittest.skip('too much output')
    def test_fix_problem_with_adapt_movebins(self):
        with open('stubs/sol_14349.pkl', 'rb') as input:
            sol14349 = pickle.load(input)

        sol14349 = self.check_sol_for_rack_violations(sol14349)
        violations = self.moop.period_fill_limit(sol14349)
        self.moop.fix_tfill(violations, sol14349)

    @unittest.skip('too much output')
    def test_fix_no_options_at_5400(self):
        with open('stubs/momasol_13263.pkl', 'rb') as input:
            sol13263 = pickle.load(input)

        sol13263 = self.check_sol_for_rack_violations(sol13263)
        violations = self.moop.period_fill_limit(sol13263)
        sol13263 = self.moop.fix_tfill(violations, sol13263)
        violations = self.moop.period_fill_limit(sol13263)
        self.assertFalse(violations)

    def test_fix_remove_hot_cookies(self):
        with pkg_resources.path(stubs, 'momasol_9931.pkl') as momasol_9931:
            with open(momasol_9931, 'rb') as input:
                sol9931 = pickle.load(input)

        sol9931 = self.check_sol_for_rack_violations(sol9931)

    def check_sol_for_rack_violations(self, sol):
        # sol is an instance of a solution class
        rackviolations = self.moop.rackcapacity(sol.getx(), sol.gettfill())

        # We can fix cooling rack violations:
        if rackviolations:
            self.moop.fixcoolingrack(rackviolations, sol)
        return sol


if __name__ == '__main__':
    unittest.main()