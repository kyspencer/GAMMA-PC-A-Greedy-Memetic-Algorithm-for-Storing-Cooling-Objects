''' test_eaf3D.py

    This python scripts tests methods in eaf3D.py.
        - insert()
        - adjustBalances()
        - case1()
        - case2()
        - case3()

'''

import unittest
import avltree_eaf3d as bst
import eaf3D
import numpy as np
from operator import attrgetter
from stack import Stack


class EAF3DTransformTests(unittest.TestCase):

    def setUp(self):
        folder = 'NSGAII/Cookies24/'
        sets = eaf3D.retrieve_input_sequences(folder)
        self.eaf_maker = eaf3D.EAF_3D(sets)

    def test_ensure_setUp_worked(self):
        self.assertEqual(self.eaf_maker.lstar[0].root.point.x, 4)
        self.assertAlmostEqual(self.eaf_maker.lstar[0].root.point.y, 12.67,
                               places=2)

    def test_transform(self):
        self.eaf_maker.transform()
        for t in range(self.eaf_maker.n):
            self.check_node_balances(self.eaf_maker.lstar[t])

    def test_find_attainment_point(self):
        # p will be point at (5, 10.42, 6113.4) from set 0
        p = self.eaf_maker.qstack.pop()
        j = p.input_set()
        # q will be point at (-inf, inf, -inf)
        q = self.eaf_maker.xstar[j].floor_x(p)
        # Here, t=0, so only one while loop encountering r at (4, 12.67, 6113),
        # whose y-value is bigger than p's, so s will be a sentinel
        t, tmin = self.eaf_maker.tmax, 0
        s, tmin = self.eaf_maker.find_attainment_point(p, q, t, tmin)
        self.assertEqual(s[0].x, self.eaf_maker.p1.x)
        self.assertEqual(s[0].y, self.eaf_maker.p1.y)
        self.assertEqual(tmin, 0)

    def test_compare_p_to_surfaces(self):
        # p will be point at (4, 15.42, 6131.4) from set 3
        p = self.eaf_maker.qstack.pop()
        j = p.input_set()
        # q will be point at (-inf, inf, -inf)
        q = self.eaf_maker.xstar[j].floor_x(p)
        # Here, t=0, so only one while loop encountering r at (3, 22.6, 6131),
        # resulting in new point s[0] at (4, 22.6, 6131)
        t, tmin = self.eaf_maker.tmax, 0
        s, tmin = self.eaf_maker.find_attainment_point(p, q, t, tmin)
        s = self.eaf_maker.compare_p_to_surfaces(s, p, q, j, tmin)
        self.assertEqual(len(s), self.eaf_maker.n)

    def test_lstar_higher_x(self):
        # p will be point at (4, 15.42, 6131.4) from set 3
        p = self.eaf_maker.qstack.pop()
        j = p.input_set()
        # q will be point at (-inf, inf, -inf)
        q = self.eaf_maker.xstar[j].floor_x(p)
        newq = self.eaf_maker.xstar[j].higher_x(q)
        self.assertEqual(newq.x, 10E10)

    def test_submit_to_lstar(self):
        st = self.eaf_maker.lstar[0].root.point
        self.eaf_maker.submit_to_lstar(st, 1)
        (pivot, theStack, parent, found) = self.eaf_maker.lstar[1].search(st)
        self.assertTrue(found)

    def test_submit_to_lstar_wdelete(self):
        # This tests is EAF3D correctly deletes nodes from trees
        # Loop 1
        p = self.eaf_maker.qstack.pop()
        j = p.input_set()
        q = self.eaf_maker.xstar[j].floor_x(p)
        if p.y < q.y:
            t, tmin = self.eaf_maker.tmax, 0
            s, tmin = self.eaf_maker.find_attainment_point(p, q, t, tmin)
            s = self.eaf_maker.compare_p_to_surfaces(s, p, q, j, tmin)
            self.eaf_maker.submit_points_lstar(s, p, q, tmin)
            self.eaf_maker.submit_to_xstar(p, j)
        if j not in self.eaf_maker.a_tracker:
            self.eaf_maker.a_tracker.append(j)
            self.eaf_maker.tmax = min(self.eaf_maker.tmax + 1, self.eaf_maker.n - 2)
        # Loop 1.5
        p = self.eaf_maker.qstack.pop()
        j = p.input_set()
        q = self.eaf_maker.xstar[j].floor_x(p)
        if p.y < q.y:
            t, tmin = self.eaf_maker.tmax, 0
            s, tmin = self.eaf_maker.find_attainment_point(p, q, t, tmin)
            s = self.eaf_maker.compare_p_to_surfaces(s, p, q, j, tmin)

    def check_node_balances(self, tree):
        # This module performs a check of all the balances in the tree
        leaves = [tree.root]
        while any(leaves):
            for f in range(len(leaves)):
                if leaves[f]:
                    correct_balance = tree.recalculate_balance(leaves[f])
                    if leaves[f].balance != correct_balance:
                        tree.print_astree()
                    self.assertEqual(leaves[f].balance, correct_balance)
            leaves = tree.next_tree_row(leaves)


class SmallSubmitLstarTests(unittest.TestCase):

    def setUp(self):
        folder = 'MOMA/Cookies24/'
        sets = eaf3D.retrieve_input_sequences(folder)
        self.eaf_maker = eaf3D.EAF_3D(sets)

    def test_missing_sentinel(self):
        # This test recreates an error to figure out the problem.
        # Recreate tree right before error occured
        p1 = eaf3D.ApproxPoint(None, None, np.array([3, 25.58, 0]))
        p2 = eaf3D.ApproxPoint(None, None, np.array([5, 24.93, 0]))
        p3 = eaf3D.ApproxPoint(None, None, np.array([8, 21.23, 0]))
        self.eaf_maker.lstar[3].insert(p1)
        self.eaf_maker.lstar[3].insert(p3)
        self.eaf_maker.lstar[3].insert(p2)
        # Insert problem child
        newp = eaf3D.ApproxPoint(None, None, np.array([5, 21.226, 5987]))
        self.eaf_maker.submit_to_lstar(newp, 3)
        # The issue was that the left only child of the right only child
        # is not connecting
        (pivot, theStack, parent, found) = self.eaf_maker.lstar[3].\
            search(self.eaf_maker.p1)
        self.assertTrue(found)

    def test_incorrect_balance(self):
        # This test recreates an error to figure out the problem.
        # Recreate tree right before error occured
        p1 = eaf3D.ApproxPoint(None, None, np.array([3, 21.28, 0]))
        p2 = eaf3D.ApproxPoint(None, None, np.array([5, 12.93, 0]))
        p3 = eaf3D.ApproxPoint(None, None, np.array([6, 10.42, 0]))
        self.eaf_maker.lstar[1].insert(p1)
        self.eaf_maker.lstar[1].insert(p3)
        self.eaf_maker.lstar[1].insert(p2)
        # Insert problem child (newroot needs to be rebalanced after kids)
        newp = eaf3D.ApproxPoint(None, None, np.array([4, 15.79, 0]))
        self.eaf_maker.lstar[1].insert(newp)
        # Assert that the tree correctly balances
        self.check_node_balances(self.eaf_maker.lstar[1])

    def check_node_balances(self, tree):
        # This module performs a check of all the balances in the tree
        leaves = [tree.root]
        while any(leaves):
            for f in range(len(leaves)):
                if leaves[f]:
                    correct_balance = tree.recalculate_balance(leaves[f])
                    if leaves[f].balance != correct_balance:
                        tree.print_astree()
                    self.assertEqual(leaves[f].balance, correct_balance)
            leaves = tree.next_tree_row(leaves)


class LargeSubmitLstarTests(unittest.TestCase):

    def setUp(self):
        folder = 'GAMMA-PC/Cookies1000/'
        sets = eaf3D.retrieve_input_sequences(folder)
        self.eaf_maker = eaf3D.EAF_3D(sets)

    def test_incorrect_balance(self):
        # This test recreates an error to figure out the problem.
        # Recreate tree right before error occured
        p1 = eaf3D.ApproxPoint(None, None, np.array([58, 32.51, 0]))
        p2 = eaf3D.ApproxPoint(None, None, np.array([43, 35.39, 0]))
        p3 = eaf3D.ApproxPoint(None, None, np.array([44, 33.46, 0]))
        p4 = eaf3D.ApproxPoint(None, None, np.array([103, 32.26, 0]))
        p5 = eaf3D.ApproxPoint(None, None, np.array([42, 37.10, 0]))
        self.eaf_maker.lstar[8].insert(p1)
        self.eaf_maker.lstar[8].insert(p3)
        self.eaf_maker.lstar[8].insert(p2)
        self.eaf_maker.lstar[8].insert(p4)
        self.eaf_maker.lstar[8].insert(p5)
        # Insert problem child and remove dominated points
        newp = eaf3D.ApproxPoint(None, None, np.array([54, 32.26, 0]))
        omegas = self.eaf_maker.lstar[8].list_nodes_domxy(newp)
        while omegas:
            self.eaf_maker.lstar[8].remove_node(omegas[0])
            omegas = self.eaf_maker.lstar[8].list_nodes_domxy(newp)
        self.eaf_maker.lstar[8].insert(newp)
        # Assert that the tree correctly balances
        self.check_node_balances(self.eaf_maker.lstar[8])

    def test_1(self):
        # This test recreates an error to figure out the problem.
        # Recreate tree right before error occured
        t = 7
        p1 = eaf3D.ApproxPoint(None, None, np.array([41.0, 50.56, 0]))
        p2 = eaf3D.ApproxPoint(None, None, np.array([42, 34.94, 9740.30]))
        p3 = eaf3D.ApproxPoint(None, None, np.array([44.0, 33.30406447, 0]))
        p4 = eaf3D.ApproxPoint(None, None, np.array([43.0, 33.75, 0]))
        p5 = eaf3D.ApproxPoint(None, None, np.array([45, 33.08, 0]))
        self.eaf_maker.lstar[t].insert(p2)
        self.eaf_maker.lstar[t].insert(p1)
        self.eaf_maker.lstar[t].insert(p3)
        self.eaf_maker.lstar[t].insert(p4)
        self.eaf_maker.lstar[t].insert(p5)
        newchild = self.eaf_maker.lstar[t].root.left.rotateLeft()
        newchild.balance = -1
        newchild.left.balance = 0
        self.eaf_maker.lstar[t].root.left = newchild
        child2 = self.eaf_maker.lstar[t].root.right.right.rotateRight()
        child2.balance = 1
        child2.right.balance = 0
        self.eaf_maker.lstar[t].root.right.right = child2
        # Add point right before error:
        p6 = eaf3D.ApproxPoint(None, None, np.array([46, 32.99667, 9708.755383]))
        self.eaf_maker.submit_to_lstar(p6, t)
        # Insert problem child and remove dominated points
        newp = eaf3D.ApproxPoint(None, None, np.array([43, 33.30406447, 9700.916965]))
        omegas = self.eaf_maker.lstar[t].list_nodes_domxy(newp)
        while omegas:
            self.eaf_maker.lstar[t].remove_node(omegas[0])
            self.check_node_balances(self.eaf_maker.lstar[t])
            omegas = self.eaf_maker.lstar[t].list_nodes_domxy(newp)
        self.eaf_maker.lstar[t].insert(newp)
        # Assert that the tree correctly balances
        self.check_node_balances(self.eaf_maker.lstar[t])

    def test_2(self):
        # This test recreates an error to figure out the problem.
        # Recreate tree right before error occured
        t = 8
        p1 = eaf3D.ApproxPoint(None, None, np.array([41, 52.61580174, 9787.313057]))
        p2 = eaf3D.ApproxPoint(None, None, np.array([42, 50.55516332, 9792.087863]))
        p3 = eaf3D.ApproxPoint(None, None, np.array([46, 34.52374955, 9776.017285]))
        p4 = eaf3D.ApproxPoint(None, None, np.array([44, 35.27757581, 9773.745261]))
        self.eaf_maker.lstar[t].insert(p2)
        self.eaf_maker.lstar[t].insert(p1)
        self.eaf_maker.lstar[t].insert(p3)
        self.eaf_maker.lstar[t].insert(p4)
        # Insert new child and remove dominated points
        newp = eaf3D.ApproxPoint(None, None, np.array([43, 40.23564548, 9772.417427]))
        self.eaf_maker.lstar[t].insert(newp)
        # Assert that the tree correctly balances
        self.check_node_balances(self.eaf_maker.lstar[t])
        # Remove node (problem occurs here)
        newp2 = eaf3D.ApproxPoint(None, None, np.array([42, 46.04608999, 9763.185098]))
        omegas = self.eaf_maker.lstar[t].list_nodes_domxy(newp2)
        self.eaf_maker.lstar[t].remove_node(omegas[0])
        # Assert that the tree correctly balances
        self.check_node_balances(self.eaf_maker.lstar[t])

    def test_3(self):
        # This test recreates an error to figure out the problem.
        # Recreate tree right before error occured
        t = 9
        p1 = eaf3D.ApproxPoint(None, None, np.array([44, 32.99667322, 9692.300569]))
        p2 = eaf3D.ApproxPoint(None, None, np.array([41, 50.18079041, 9652.966452]))
        p3 = eaf3D.ApproxPoint(None, None, np.array([43, 33.29213592, 9646.838321]))
        p4 = eaf3D.ApproxPoint(None, None, np.array([42, 34.93930038, 9590.186112]))
        p5 = eaf3D.ApproxPoint(None, None, np.array([51, 29.59311919, 9564.932893]))
        p6 = eaf3D.ApproxPoint(None, None, np.array([47, 31.53745389, 9559.198558]))
        p7 = eaf3D.ApproxPoint(None, None, np.array([50, 30.54022012, 9555.970874]))
        p8 = eaf3D.ApproxPoint(None, None, np.array([53, 28.91678773, 9555.401697]))
        p9 = eaf3D.ApproxPoint(None, None, np.array([58, 27.73733822, 9555.401697]))
        p10 = eaf3D.ApproxPoint(None, None, np.array([61, 26.67391754, 9555.401697]))
        p11 = eaf3D.ApproxPoint(None, None, np.array([67, 25.55429681, 9552.098401]))
        p12 = eaf3D.ApproxPoint(None, None, np.array([62, 25.88606603, 9551.119924]))
        p13 = eaf3D.ApproxPoint(None, None, np.array([64, 25.55429681, 9551.119924]))
        self.eaf_maker.lstar[t].insert(p1)
        self.eaf_maker.lstar[t].insert(p2)
        self.eaf_maker.lstar[t].insert(p5)
        self.eaf_maker.lstar[t].insert(p3)
        self.eaf_maker.lstar[t].insert(p6)
        self.eaf_maker.lstar[t].insert(p10)
        self.eaf_maker.lstar[t].insert(p4)
        self.eaf_maker.lstar[t].insert(p7)
        self.eaf_maker.lstar[t].insert(p9)
        self.eaf_maker.lstar[t].insert(p8)
        self.eaf_maker.lstar[t].insert(p11)
        self.eaf_maker.submit_to_lstar(p12, t)
        self.eaf_maker.submit_to_lstar(p13, t)
        self.eaf_maker.lstar[t].print_astree()
        # Assert that the tree correctly balances
        self.check_node_balances(self.eaf_maker.lstar[t])
        # Remove node (problem occurs here)
        newp = eaf3D.ApproxPoint(None, None, np.array([50, 29.59311919, 9547.593165]))
        omegas = self.eaf_maker.lstar[t].list_nodes_domxy(newp)
        while omegas:
            self.eaf_maker.lstar[t].remove_node(omegas[0])
            self.check_node_balances(self.eaf_maker.lstar[t])
            omegas = self.eaf_maker.lstar[t].list_nodes_domxy(newp)
            self.eaf_maker.lstar[t].print_astree()
        # Assert that the tree correctly balances
        self.check_node_balances(self.eaf_maker.lstar[t])

    def check_node_balances(self, tree):
            # This module performs a check of all the balances in the tree
            leaves = [tree.root]
            while any(leaves):
                for f in range(len(leaves)):
                    if leaves[f]:
                        correct_balance = tree.recalculate_balance(leaves[f])
                        if leaves[f].balance != correct_balance:
                            tree.print_astree()
                        self.assertEqual(leaves[f].balance, correct_balance)
                        self.assertLessEqual(abs(leaves[f].balance), 1)
                leaves = tree.next_tree_row(leaves)


if __name__ == '__main__':
    unittest.main()
