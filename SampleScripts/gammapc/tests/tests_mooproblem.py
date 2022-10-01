# test_mop.py
#    This file tests mooproblem.py for errors.
#    Author: Kristina Yancey Spencer

import unittest
from mock import Mock
import numpy as np
from io import StringIO
from random import choice

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from ..binpacking_dynamic import BPP, coordarrays
from .. import coolcookies, mooproblem, solutions_dynamic as sols
from ..grasp import RCLtime
from . import stubs  # relative-import the *package* containing the stubs


class MOCookieProblemTests(unittest.TestCase):

    def setUp(self):
        with pkg_resources.path(stubs, 'Cookies24.txt') as cookietext:
            cookies = coolcookies.makeobjects(24, 6, cookietext)
            self.moop = mooproblem.MOCookieProblem(24, 8, 15, 2, cookies)

        # Make mock solution --------------------------------------------------
        # variable length representation
        vlrep = [[0, 1, 2, 4], [3, 5], [6, 7, 8, 9], [10, 11], [12, 13, 14, 15],
                 [16, 17], [18, 19, 20], [21, 22, 23]]
        # tfill matrix
        self.tfill = np.zeros(24, dtype=np.float64)
        tfill_unique = [885.0, 722.0, 1507.0, 1428.0, 2210.0,
                        1958.0, 2764.0, 2509.0]
        for i in range(len(tfill_unique)):
            self.tfill[i] = tfill_unique[i]

        # x and y matrices
        self.y = np.zeros(24, dtype=int)
        self.y[:len(vlrep)] = 1
        self.x1 = np.zeros((24, 24), dtype=int)
        for i in range(len(vlrep)):
            for j in vlrep[i]:
                self.x1[i, j] = 1

        self.mock = Mock()
        self.mock.getid.return_value = 0
        self.mock.getx.return_value = self.x1
        self.mock.gety.return_value = self.y
        self.mock.gettfill.return_value = self.tfill
        self.mock.getvlrep.return_value = vlrep

    def test_avgheat(self):
        avgheat = self.moop.avgheat(self.mock)
        # Inspect the arguments that solution.setq0bins was called with
        inspect = self.mock.mock_calls[-1]
        name, args, kwargs = inspect
        q0bins = args[0]
        # Make sure the average is less than the maximum
        self.assertLess(avgheat, np.max(q0bins))
        # Make sure only filled bins have a heat
        self.assertEqual(q0bins[8], 0)

    def test_getinitialboxheat(self):
        boxheat = self.moop.getinitialboxheat(0, self.x1, 700)
        self.assertAlmostEqual(boxheat, 21.17, places=2)

    def test_maxreadyt(self):
        maxtime = self.moop.maxreadyt(self.mock)
        self.assertAlmostEqual(maxtime, 5179.9, places=1)

    def test_modregulafalsi(self):
        vlrepi = [0, 1, 2, 4]
        ck = self.moop.modregulafalsi(vlrepi, 885.0, 4800)
        self.assertGreater(ck, 3600)
        self.assertLess(ck, 3700)
        # Check that ready function is close to 0
        f_ck = self.moop.readyfunction(vlrepi, ck)
        self.assertAlmostEqual(f_ck, 0, delta=1.0)

    def test_findnewinput(self):
        vlrepi = [0, 1, 2, 4]
        # Check if return old t0 and t1
        t0, t1 = self.moop.findnewinput(vlrepi, 885.0, 4800)
        self.assertTupleEqual((t0, t1), (885.0, 4800))
        # Check if return new t0 and t1
        t0, t1 = self.moop.findnewinput(vlrepi, 885.0, 900)
        self.assertTupleEqual((t0, t1), (3600, 7200))
        # Check else clause:
        t0, t1 = self.moop.findnewinput(vlrepi, 4000, 7200)
        self.assertTupleEqual((t0, t1), (2000, 4000))

    def test_readyfunction(self):
        vlrepi = [0, 1, 2, 4]
        fofxi = self.moop.readyfunction(vlrepi, 600)
        self.assertAlmostEqual(fofxi, 549.597, places=2)

    def test_newx_regfalsi(self):
        vlrepi = [0, 1, 2, 4]
        # Check for normal operation
        ck = self.moop.newx_regfalsi(vlrepi, 885.0, 4800)
        self.assertAlmostEqual(ck, 4576.96, places=2)

    def test_newx_modregfalsi(self):
        vlrepi = [0, 1, 2, 4]
        ck = self.moop.newx_modregfalsi(vlrepi, 885.0, 4800, 1, 0.5)
        self.assertGreater(ck, 4576.96)

    def test_noreplacement_pass(self):
        # This test will fail if the RuntimeError is raised.
        self.moop.noreplacement(1, self.x1)

    def test_noreplacement_missingcookie(self):
        x2 = self.x1.copy()
        x2[0, 4] = 0
        self.assertRaises(RuntimeError, self.moop.noreplacement, 1, x2)

    def test_noreplacement_toomany(self):
        x2 = self.x1.copy()
        x2[1, 4] = 1
        self.assertRaises(RuntimeError, self.moop.noreplacement, 1, x2)

    def test_boxcapacity_passes(self):
        # This test passes if it does not raise a RuntimeError
        self.moop.boxcapacity(1, self.x1)

    def test_boxcapacity_overfull(self):
        x2 = np.zeros((24, 24), dtype=int)
        x2[0, :] = 1
        self.assertRaises(RuntimeError, self.moop.boxcapacity, 2, x2)

    def test_timeconstraint(self):
        self.assertTrue(self.moop.timeconstraint(self.mock) == [])

    def test_rackcapacity(self):
        violations = self.moop.rackcapacity(self.x1, self.tfill)
        self.assertTrue(violations == [])

    def test_period_fill_limit(self):
        # Test for no violations
        violations = self.moop.period_fill_limit(self.mock)
        self.assertTrue(violations == [])
        # Test for violations
        tfill2 = np.zeros(24, dtype=np.float64)
        tfill2[:8] = 2500
        y2 = np.zeros(24, dtype=int)
        y2[:8] = 1
        self.mock.gettfill.return_value = tfill2
        self.mock.gety.return_value = y2
        violations = self.moop.period_fill_limit(self.mock)
        self.assertEqual(violations[0], 2400)

    def test_timeintervals(self):
        tfill = np.zeros(5, dtype=np.float64)
        tfill_unique = [700, 800, 1450, 900, 1100]
        for i in range(len(tfill_unique)):
            tfill[i] = tfill_unique[i]
        expected = [600, 900, 1200, 1450, 1800, 2400]
        t_intervals = self.moop.gettimeintervals(tfill)
        self.assertListEqual(t_intervals, expected)

    def test_rackij(self):
        cookie = self.moop.cookies.get(0)
        # Test for present
        self.assertEqual(self.moop.rackij(600, 700, cookie), 1)
        # Test for absent
        self.assertEqual(self.moop.rackij(800, 700, cookie), 0)

    def test_edittfill(self):
        mocksol = Mock()
        vlrep = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15],
                 [16, 17, 18, 19, 20, 21, 22, 23]]
        mocksol.getvlrep.return_value = vlrep
        tfill = np.zeros(24, dtype=np.float64)
        tfill[:3] = [1900, 2200, 2700]
        mocksol.gettfill.return_value = tfill
        x = np.zeros((24, 24), dtype=int)
        x[0, :8] = 1
        x[1, 8:16] = 1
        x[2, 16:24] = 1
        mocksol.getx.return_value = x
        mocksol.getopenbins.return_value = 3
        self.moop.edittfill2(1800, mocksol)
        # Inspect the arguments that solution.edit_tfilli was called with
        inspect = mocksol.mock_calls[-1]
        name, args, kwargs = inspect
        self.assertLess(args[1], 1800)

    def test_movetonewbox(self):
        indexoptions = [(0, 0), (0, 1), (0, 2), (1, 3), (0, 4)]
        # Make list of 16 cookies
        clist = [choice(indexoptions) for _ in range(16)]
        self.moop.movecookies(650, clist, self.mock)
        self.mock.repackitemsatt.assert_called_with(650, [(0, 4)])


class CoolingRackTests(unittest.TestCase):

    def setUp(self):
        #seed(56)
        with pkg_resources.path(stubs, 'chrom0.npz') as chrom0:
            chromstored = np.load(chrom0)
            chromname = chromstored.files
            self.chrom0 = chromstored[chromname[0]]

        with pkg_resources.path(stubs, 'tfill0.npz') as tfill0:
            tfillstored = np.load(tfill0)
            tfillname = tfillstored.files
            self.tfill0 = tfillstored[tfillname[0]]

        with pkg_resources.path(stubs, 'newgenes129.npz') as newgenes129:
            newgenesstored = np.load(newgenes129)
            newgenesfiles = newgenesstored.files
            self.chrom = newgenesstored[newgenesfiles[0]]
            self.tfill = newgenesstored[newgenesfiles[1]]
            self.tfill = np.reshape(self.tfill, 1000)

        with pkg_resources.path(stubs, 'Cookies1000.txt') as cookietext:
            cookies = coolcookies.makeobjects(1000, 100, cookietext)
            self.bpp = BPP(1000, 24, 300, cookies)
            self.moop = mooproblem.MOCookieProblem(1000, 24, 300, 8, cookies)

    def test_fixcoolingrack(self):
        solution = sols.MultiSol(129, self.chrom, self.tfill, self.bpp)
        testviolations = self.moop.rackcapacity(solution.getx(), solution.gettfill())
        self.moop.fixcoolingrack(testviolations, solution)
        solution = coordarrays(solution)
        # Make sure fixcoolingrack() fixed all violations
        testviolations = self.moop.rackcapacity(solution.getx(), solution.gettfill())
        self.assertEqual(testviolations, [])
        # Make sure it didn't create other constraint errors
        self.moop.noreplacement(129, solution.getx())
        tviolations = self.moop.timeconstraint(solution)
        self.assertEqual(tviolations, [])
        self.moop.calcfeasibility(solution)

    def test_fix_tfill(self):
        solution = sols.MultiSol(130, self.chrom, self.tfill, self.bpp)
        testviolations = self.moop.rackcapacity(solution.getx(), solution.gettfill())
        self.moop.fixcoolingrack(testviolations, solution)
        solution = coordarrays(solution)
        fillviolations = self.moop.period_fill_limit(solution)
        if fillviolations:
            self.moop.fix_tfill(fillviolations, solution)
        violations_left = self.moop.period_fill_limit(solution)
        self.assertEqual(violations_left, [])

    def test_openonebox(self):
        solution = sols.MultiSol(0, self.chrom0, self.tfill0, self.bpp)
        mooproblem.checkformismatch(solution)
        violations = self.moop.rackcapacity(solution.getx(), solution.gettfill())
        for tv in range(len(violations)):
            t = violations[tv][0]
            self.moop.openonebox(t, solution)
        out = StringIO()
        mooproblem.checkformismatch(solution, out=out)
        output = out.getvalue().strip()
        self.assertEqual(output, '')

#    def test_edittfill(self):
#        solution = sols.MultiSol(0, self.chrom, self.tfill, self.bpp)
#        testviolations = self.moop.rackcapacity(solution.getx(), solution.gettfill())
#        t = testviolations[0][0]
#        clist = testviolations[0][1]
#        solutions = self.moop.edittfill(t, clist, solution)
#        self.assertEqual(len(testviolations[0][1]), 481)
#
#    def test_swapitems(self):
#        solution = sols.MultiSol(0, self.chrom, self.tfill, self.bpp)
#        # Problem: swapitems in solution says cannot remove j1 from vlrep[i1]
#        booleanbins = [[42, [True, True, True, True, True, True, True, True,
#                             True, True, True, True, True, True, True, True,
#                             True, False, False, False, False, False, False,
#                             False]],
#                       [43, [True, True, True, True, True, True, True, False,
#                             False, False, False, False, False, False, False,
#                             False, False, False, False, False, False, False,
#                             False, False]],
#                       [44, [True, True, True, True, True, True, True, True,
#                             False, False, False, False, False, False, False,
#                             False, False, False, False, False, False, False,
#                             False, False]],
#                       [45, [True, True, True, True, True, True, True, True,
#                             True, True, True, True, True, True, True, True,
#                             False, False, False, False, False, False, False,
#                             False]]]
#        clist = [(81, 32), (81, 54), (81, 55), (81, 213), (81, 217), (81, 228),
#                 (81, 236), (81, 268), (81, 298), (81, 328), (81, 340), (81, 388),
#                 (81, 462), (81, 473), (81, 516), (81, 585), (80, 97), (80, 193),
#                 (80, 252), (80, 362), (80, 471), (80, 478), (79, 15), (79, 51),
#                 (79, 67), (79, 98), (79, 186), (79, 232), (79, 245), (79, 355),
#                 (79, 361), (79, 397), (79, 439), (79, 540), (79, 576), (78, 23),
#                 (78, 33), (78, 50), (78, 78), (78, 87), (78, 105), (78, 122),
#                 (78, 246), (78, 262), (78, 263), (78, 307), (78, 366), (78, 380),
#                 (78, 401), (78, 531), (77, 163), (77, 170), (77, 216), (77, 260),
#                 (77, 364), (77, 395), (77, 404), (77, 407), (77, 418), (77, 430),
#                 (77, 481), (77, 509), (77, 518), (77, 524), (77, 548), (76, 1),
#                 (76, 13), (76, 60), (76, 106), (76, 179), (76, 182), (76, 235),
#                 (76, 322), (76, 331), (76, 350), (76, 373), (76, 399), (76, 513),
#                 (76, 569), (76, 586), (75, 114), (75, 128), (75, 138), (75, 183),
#                 (75, 195), (75, 218), (75, 300), (75, 327), (75, 486), (75, 511),
#                 (75, 520), (74, 9), (74, 18), (74, 83), (74, 91), (74, 107),
#                 (74, 201), (74, 275), (74, 377), (74, 414), (74, 433), (74, 444),
#                 (74, 538), (73, 49), (73, 58), (73, 62), (73, 84), (73, 135),
#                 (73, 141), (73, 154), (73, 199), (73, 202), (73, 203), (73, 210),
#                 (73, 342), (73, 365), (73, 381), (73, 436), (73, 469), (73, 495),
#                 (73, 584), (72, 40), (72, 356), (72, 510), (71, 2), (71, 48),
#                 (71, 95), (71, 104), (71, 143), (71, 166), (71, 185), (71, 250),
#                 (71, 285), (71, 291), (71, 313), (71, 344), (71, 372), (71, 451),
#                 (71, 515), (71, 543), (71, 591), (71, 594), (70, 17), (70, 56),
#                 (70, 118), (70, 134), (70, 149), (70, 196), (70, 231), (70, 333),
#                 (70, 354), (70, 580), (70, 595), (70, 597), (69, 93), (69, 113),
#                 (69, 189), (69, 249), (69, 269), (69, 287), (69, 396), (69, 413),
#                 (69, 420), (69, 438), (69, 456), (69, 479), (69, 532), (69, 564),
#                 (68, 47), (68, 75), (68, 121), (68, 178), (68, 226), (68, 237),
#                 (68, 324), (68, 336), (68, 378), (68, 398), (68, 455), (68, 459),
#                 (68, 494), (68, 526), (68, 575), (67, 108), (67, 164), (67, 292),
#                 (67, 351), (67, 389), (67, 429), (67, 450), (67, 498), (67, 523),
#                 (67, 552), (67, 557), (67, 558), (66, 115), (66, 187), (66, 288),
#                 (66, 341), (66, 417), (66, 501), (66, 508), (66, 539), (66, 554),
#                 (66, 566), (65, 220), (65, 238), (65, 304), (65, 387), (65, 402),
#                 (65, 442), (65, 474), (65, 503), (65, 547), (64, 43), (64, 74),
#                 (64, 153), (64, 155), (64, 169), (64, 180), (64, 206), (64, 222),
#                 (64, 248), (64, 254), (64, 338), (64, 435), (64, 493), (64, 496),
#                 (64, 507), (64, 596), (63, 68), (63, 77), (63, 137), (63, 144),
#                 (63, 146), (63, 294), (63, 335), (63, 357), (63, 359), (63, 446),
#                 (63, 519), (63, 534), (62, 5), (62, 36), (62, 125), (62, 198),
#                 (62, 229), (62, 244), (62, 258), (62, 283), (62, 303), (62, 312),
#                 (62, 358), (62, 412), (62, 445), (62, 464), (62, 593), (61, 90),
#                 (61, 92), (61, 140), (61, 173), (61, 224), (61, 286), (61, 309),
#                 (61, 316), (61, 370), (61, 384), (61, 485), (61, 505), (61, 535),
#                 (61, 587), (60, 19), (60, 46), (60, 57), (60, 65), (60, 86),
#                 (60, 116), (60, 123), (60, 158), (60, 171), (60, 299), (60, 317),
#                 (60, 403), (60, 449), (60, 466), (60, 497), (59, 12), (59, 59),
#                 (59, 103), (59, 176), (59, 207), (59, 208), (59, 227), (59, 274),
#                 (59, 277), (59, 310), (59, 349), (59, 393), (59, 437), (59, 589),
#                 (58, 22), (58, 41), (58, 52), (58, 64), (58, 72), (58, 73),
#                 (58, 82), (58, 120), (58, 148), (58, 188), (58, 190), (58, 204),
#                 (58, 271), (58, 305), (58, 321), (58, 330), (58, 460), (58, 489),
#                 (58, 521), (58, 588), (57, 0), (57, 147), (57, 162), (57, 194),
#                 (57, 197), (57, 205), (57, 247), (57, 266), (57, 369), (57, 383),
#                 (57, 391), (57, 522), (57, 562), (57, 563), (56, 63), (56, 102),
#                 (56, 110), (56, 174), (56, 211), (56, 212), (56, 297), (56, 347),
#                 (56, 374), (56, 426), (56, 434), (56, 458), (56, 468), (56, 545),
#                 (56, 550), (56, 571), (56, 574), (55, 156), (55, 200), (55, 223),
#                 (55, 233), (55, 284), (55, 352), (55, 360), (55, 367), (55, 415),
#                 (55, 441), (55, 492), (55, 544), (55, 570), (54, 21), (54, 71),
#                 (54, 133), (54, 139), (54, 145), (54, 165), (54, 168), (54, 243),
#                 (54, 259), (54, 325), (54, 390), (53, 4), (53, 26), (53, 27),
#                 (53, 66), (53, 126), (53, 136), (53, 167), (53, 234), (53, 256),
#                 (53, 270), (53, 528), (53, 577), (53, 599), (52, 31), (52, 129),
#                 (52, 130), (52, 221), (52, 306), (52, 406), (52, 525), (52, 549),
#                 (52, 551), (52, 560), (52, 568), (52, 578), (51, 25), (51, 39),
#                 (51, 79), (51, 81), (51, 219), (51, 318), (51, 320), (51, 323),
#                 (51, 343), (51, 392), (51, 408), (51, 422), (51, 432), (51, 447),
#                 (51, 484), (51, 488), (51, 502), (51, 517), (49, 337), (49, 339),
#                 (49, 363), (49, 385), (49, 419), (49, 453), (49, 454), (49, 491),
#                 (49, 542), (49, 556), (49, 561), (49, 592), (48, 368), (48, 379),
#                 (48, 424), (48, 440), (48, 467), (48, 500), (48, 533), (48, 559),
#                 (48, 573), (48, 579), (47, 334), (47, 346), (47, 410), (47, 472),
#                 (47, 475), (47, 477), (47, 506), (47, 527), (47, 530), (47, 541),
#                 (47, 546), (47, 572), (46, 400), (46, 427), (46, 452), (46, 470),
#                 (46, 555), (45, 302), (45, 326), (45, 332), (45, 353), (45, 376),
#                 (45, 386), (45, 411), (45, 416), (45, 421), (45, 428), (45, 443),
#                 (45, 448), (45, 457), (45, 480), (45, 582), (45, 598), (44, 314),
#                 (44, 348), (44, 382), (44, 423), (44, 482), (44, 483), (44, 487),
#                 (44, 514), (43, 315), (43, 329), (43, 409), (43, 431), (43, 461),
#                 (43, 465), (43, 512), (42, 301), (42, 308), (42, 311), (42, 319),
#                 (42, 345), (42, 394), (42, 405), (42, 463), (42, 476), (42, 499),
#                 (42, 529), (42, 536), (42, 537), (42, 565), (42, 567), (42, 583), (42, 590)]
#        i, cb, solution, booleanbins, clist =\
#            self.moop.swapitems(3750.0, booleanbins, solution, clist)
#        self.assertEqual(i, 42)
#        self.assertEqual(cb, 0)
#        self.assertTrue(all(booleanbins[cb][1]))
#        self.assertEqual(clist.count((42, 315)), 1)
#
#    def test_findcboolbin(self):
#        booleanbins = [(42, [True, True, True, True, True, True, True, True,
#                             True, True, True, True, True, True, True, True,
#                             True, False, False, False, False, False, False,
#                             False]),
#                       (43, [True, True, True, True, True, True, True, False,
#                             False, False, False, False, False, False, False,
#                             False, False, False, False, False, False, False,
#                             False, False]),
#                       (44, [True, True, True, True, True, True, True, True,
#                             False, False, False, False, False, False, False,
#                             False, False, False, False, False, False, False,
#                             False, False]),
#                       (45, [True, True, True, True, True, True, True, True,
#                             True, True, True, True, True, True, True, True,
#                             False, False, False, False, False, False, False,
#                             False])]
#        cb = self.moop.findcboolbin(booleanbins)
#        self.assertEqual(cb, 0)
#
#    def test_findcforswap(self):
#        solution = sols.MultiSol(0, self.chrom, self.tfill, self.bpp)
#        booleanbins = [(42, [True, True, True, True, True, True, True, True,
#                             True, True, True, True, True, True, True, True,
#                             True, False, False, False, False, False, False,
#                             False]),
#                       (43, [True, True, True, True, True, True, True, False,
#                             False, False, False, False, False, False, False,
#                             False, False, False, False, False, False, False,
#                             False, False]),
#                       (44, [True, True, True, True, True, True, True, True,
#                             False, False, False, False, False, False, False,
#                             False, False, False, False, False, False, False,
#                             False, False]),
#                       (45, [True, True, True, True, True, True, True, True,
#                             True, True, True, True, True, True, True, True,
#                             False, False, False, False, False, False, False,
#                             False])]
#        i2, j2, k1, k2 = self.moop.findcforswap(42, 603, 3750.0, booleanbins,
#                                                solution.tfill, solution.vlrep)
#        self.assertEqual(i2, 43)
#        self.assertEqual(j2, 315)


class FixTFillTests(unittest.TestCase):

    def setUp(self):
        with pkg_resources.path(stubs, 'Cookies24.txt') as cookietext:
            cookies = coolcookies.makeobjects(24, 6, cookietext)
            self.moop = mooproblem.MOCookieProblem(24, 8, 15, 2, cookies)

        # Make mock solution --------------------------------------------------
        # variable length representation
        self.vlrep = [[0, 1, 2, 4], [3, 5], [6, 7, 8, 9], [10, 11], [12, 13, 14, 15],
                      [16, 17], [18, 19, 20], [21, 22, 23]]

        # tfill matrix
        self.tfill = np.zeros(24, dtype=np.float64)
        tfill_unique = [885.0, 722.0, 1507.0, 1428.0, 2210.0,
                        2958.0, 2764.0, 2809.0]
        for i in range(8):
            self.tfill[i] = tfill_unique[i]

        # x and y matrices
        self.y = np.zeros(24, dtype=int)
        self.y[:len(self.vlrep)] = 1
        self.x1 = np.zeros((24, 24), dtype=int)
        for i in range(len(self.vlrep)):
            for j in self.vlrep[i]:
                self.x1[i, j] = 1

        self.mock = Mock()
        self.mock.getid.return_value = 0
        self.mock.getx.return_value = self.x1
        self.mock.gety.return_value = self.y
        self.mock.gettfill.return_value = self.tfill
        self.mock.getvlrep.return_value = self.vlrep
        self.mock.getopenbins.return_value = 8

    def test_fix_tfill(self):
        sol_try = sols.CookieSol(100, self.x1, self.y, self.vlrep, self.tfill)
        violations = self.moop.period_fill_limit(sol_try)
        solution = self.moop.fix_tfill(violations, sol_try)
        # Make sure it didn't create other constraint errors
        self.moop.noreplacement(129, solution.getx())
        tviolations = self.moop.timeconstraint(solution)
        self.assertEqual(tviolations, [])

    def test_new_tfill(self):
        violations = self.moop.period_fill_limit(self.mock)

        # Get restricted candidate list for new tfill options
        n_b = self.moop.n // self.moop.nbatches
        rcl_tfill = RCLtime(self.moop.coolrack, self.moop.fillcap, n_b,
                            self.moop.tbatch, self.moop.nbatches)
        rcl_tfill.initialize_withtfill(8, self.vlrep, self.tfill)
        t = violations[0]
        solution, rcl_tfill = self.moop.new_tfill(t, rcl_tfill, self.mock)

        # Inspect the arguments that solution.edit_tfilli was called with
        inspect = self.mock.mock_calls[-1]
        name, args, kwargs = inspect
        i, t_new = args

        # Check that t_new doesn't violate baking, rack, or fill limits
        # Baking limits:
        cookieboolean = self.moop.packatt(self.vlrep[i], t_new)
        self.assertTrue(all(cookieboolean))

        # Rack limits:
        tlist = np.where(t_new > np.array(rcl_tfill.t_t))[0]
        t = rcl_tfill.t_t[tlist[-1]]
        cookiesonrack = self.moop.find_cookies_on_rack(t, self.tfill, self.x1)
        space = self.moop.coolrack - len(cookiesonrack) - len(self.vlrep[i])
        self.assertGreater(space, 0)

        # Fill limits:
        self.assertGreaterEqual(rcl_tfill.res_fill[tlist[-1]], 0)

    def test_find_new_tfilli(self):
        violations = self.moop.period_fill_limit(self.mock)
        # Get restricted candidate list for new tfill options
        n_b = self.moop.n // self.moop.nbatches
        rcl_tfill = RCLtime(self.moop.coolrack, self.moop.fillcap, n_b,
                            self.moop.tbatch, self.moop.nbatches)
        rcl_tfill.initialize_withtfill(8, self.vlrep, self.tfill)
        # Check that t_new doesn't violate baking, rack, or fill limits
        t_new, rcl_tfill = self.moop.find_new_tfilli(self.vlrep[5], rcl_tfill)
        # Baking limits:
        cookieboolean = self.moop.packatt(self.vlrep[5], t_new)
        self.assertTrue(all(cookieboolean))
        # Rack limits:
        tlist = np.where(t_new > np.array(rcl_tfill.t_t))[0]
        t = rcl_tfill.t_t[tlist[-1]]
        cookiesonrack = self.moop.find_cookies_on_rack(t, self.tfill, self.x1)
        self.assertGreater(self.moop.coolrack - len(cookiesonrack) - 2, 0)
        # Fill limits:
        self.assertGreaterEqual(rcl_tfill.res_fill[tlist[-1]], 0)

    def test_empty_one_box(self):
        violations = self.moop.period_fill_limit(self.mock)

        # Get restricted candidate list for new tfill options
        n_b = self.moop.n // self.moop.nbatches
        rcl_tfill = RCLtime(self.moop.coolrack, self.moop.fillcap, n_b,
                            self.moop.tbatch, self.moop.nbatches)
        rcl_tfill.initialize_withtfill(8, self.vlrep, self.tfill)
        mock, rcl_tfill = self.moop.empty_one_box(violations[0], rcl_tfill,
                                                  self.mock)

        # Inspect the arguments that solution.moveitem was called with
        inspect = self.mock.mock_calls
        for k in range(len(inspect)):
            name, args, kwargs = inspect[k]
            if name == 'moveitem':
                self.assertIn(args[0], [5, 6, 7])
                self.assertIn(args[1], [16, 17, 18, 19, 20, 21, 22, 23])
                tmin = self.moop.cookies.get(args[1]).getbatch() * self.moop.tbatch
                self.assertGreaterEqual(self.tfill[args[2]], tmin)

    def test_get_box_tmin(self):
        vlrepi = [3, 5, 6, 7, 8, 9]
        tmin = self.moop.get_box_tmin(vlrepi)
        self.assertEqual(tmin, 1200)


class MOPfunctionTests(unittest.TestCase):

    def setUp(self):
        self.u = np.array([38, 500, 500], dtype=int)

    def test_dom1_u_dominant(self):
        v = np.array([39, 600, 600], dtype=int)
        self.assertTrue(mooproblem.dom1(self.u, v))

    def test_dom1_v_dominant(self):
        v = np.array([38, 500, 450], dtype=int)
        self.assertFalse(mooproblem.dom1(self.u, v))

    def test_dom1_same_fitness_values(self):
        self.assertFalse(mooproblem.dom1(self.u, self.u))

    def test_dom2_u_dominant(self):
        v = np.array([39, 600, 600], dtype=int)
        self.assertTrue(mooproblem.dom2(self.u, v))

    def test_dom2_v_dominant(self):
        v = np.array([38, 500, 450], dtype=int)
        self.assertFalse(mooproblem.dom2(self.u, v))

    def test_dom2_same_fitness_values(self):
        self.assertFalse(mooproblem.dom2(self.u, self.u))


if __name__ == '__main__':
    unittest.main()