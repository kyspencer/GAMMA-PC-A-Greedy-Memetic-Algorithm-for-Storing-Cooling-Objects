''' test_avltree.py

    This python scripts tests methods in avltree_eaf3d.py, including new
    functions requested by the article authors.
        - remove_node()
        - floor_x()
        - higher_x()
        - lower_y()

'''

import unittest
import avltree_eaf3d as bst
import eaf3D
import numpy as np
from operator import attrgetter
from stack import Stack


class AVLTreeTestsFromStarter(unittest.TestCase):

    def setUp(self):
        # Set initial points for sentinels (to simulate infinity)
        big_pos_value = 10E10
        big_neg_value = -1 * big_pos_value
        self.p1 = np.array([big_neg_value, big_pos_value, big_neg_value])
        p2 = np.array([big_pos_value, big_neg_value, big_neg_value])
        # Initialize avltree
        point0 = eaf3D.ApproxPoint(None, 1000, self.p1)
        point1 = eaf3D.ApproxPoint(None, 1001, p2)
        node0 = bst.AVLNode(point0, 1)
        node1 = bst.AVLNode(point1, 0)
        self.t = bst.AVLTree()
        self.t.root = node0
        node0.right = node1
        self.t.count = 2
        # Import more data points
        fname = 'example/run01'
        exset = eaf3D.import_approximate_set(fname)
        x, m = eaf3D.multiset_sum([exset])
        # Q is X sorted in ascending order of the z coordinate
        self.qstack = Stack()
        xintoq = sorted(x.values(), key=attrgetter('z'))
        for i in range(len(xintoq)):
            self.qstack.push(xintoq[i])
        # Add new data points to tree
        for i in range(4):
            p = self.qstack.pop()
            self.t.insert(p)

    def test_if_init_made_correctly(self):
        self.assertEqual(self.t.root.balance, 0)
        self.assertEqual(self.t.root.left.balance, 1)
        self.assertEqual(self.t.root.right.balance, -1)
        self.assertEqual(self.t.root.left.right.balance, 0)
        self.assertEqual(self.t.root.right.left.balance, 0)

    def test_insert_found(self):
        expected = str(self.t)
        self.t.insert(eaf3D.ApproxPoint(None, 1000, self.p1))
        self.assertEqual(str(self.t), expected)

    def test_insert_case2(self):
        # Test insert function for adjust only
        tree = bst.AVLTree()
        tree.set_newroot(self.qstack.pop())
        for i in range(12):
            point = self.qstack.pop()
            tree.insert(point)
        # Insert nonvalid point to prove test
        point1 = eaf3D.ApproxPoint(None, 1002, np.array([2, 0, 0]))
        tree.insert(point1)
        point2 = eaf3D.ApproxPoint(None, 1003, np.array([3.5, 0, 0]))
        tree.insert(point2)
        (pivot, theStack, parent, found) = tree.search(point1)
        theStack.pop()
        while not theStack.isEmpty():
            node = theStack.pop()
            self.assertEqual(node.balance, 0)

    def test_insert_case1(self):
        # Test for inserting into balanced tree
        # Get balanced tree
        tree = bst.AVLTree()
        tree.set_newroot(self.qstack.pop())
        for i in range(12):
            point = self.qstack.pop()
            tree.insert(point)
        # Insert nonvalid point to balance tree
        point1 = eaf3D.ApproxPoint(None, 1002, np.array([2, 0, 0]))
        tree.insert(point1)
        point2 = eaf3D.ApproxPoint(None, 1003, np.array([3.5, 0, 0]))
        tree.insert(point2)
        # Point to test
        point3 = eaf3D.ApproxPoint(None, 1004, np.array([8, 0, 0]))
        tree.insert(point3)
        # Check that all nodes along stack have balance = 1
        self.assertEqual(tree.count, 8)
        (pivot, theStack, parent, found) = tree.search(point3)
        theStack.pop()
        while not theStack.isEmpty():
            node = theStack.pop()
            self.assertEqual(node.balance, 1)

    def test_insert_case3_subcaseA(self):
        point = eaf3D.ApproxPoint(None, 1002, np.array([4.5, 0, 0]))
        (pivot, theStack, parent, found) = self.t.search(point)
        self.t.insert(point)
        self.assertEqual(pivot.balance, 0)
        self.assertEqual(self.t.root.balance, 0)

    def test_insert_case3_subcaseB(self):
        for i in range(4):
            self.qstack.pop()
        point = self.qstack.pop()
        self.t.insert(point)
        self.assertEqual(self.t.root.right.balance, 0)
        self.assertEqual(self.t.root.right.point, point)
        self.assertEqual(self.t.root.right.right.balance, 0)
        self.assertEqual(self.t.root.right.left.balance, 0)

    def test_adjustBalances_negative(self):
        for i in range(4):
            self.qstack.pop()
        point = self.qstack.pop()
        (pivot, theStack, parent, found) = self.t.search(point)
        newNode = bst.AVLNode(point)
        self.t.adjustBalances_add(theStack, pivot, newNode)
        self.assertEqual(pivot.balance, -2)

    def test_floor_x(self):
        p1 = eaf3D.ApproxPoint(1, 1002, np.array([3, 23.4623828, 6059.2348600000005]))
        p2 = eaf3D.ApproxPoint(1, 1003, np.array([4, 14.07345342, 5990.93696]))
        p3 = eaf3D.ApproxPoint(1, 1004, np.array([5, 10.90633272, 5965.522494]))
        p4 = eaf3D.ApproxPoint(1, 1005, np.array([7, 10.73267638, 5868.0173159999995]))
        p5 = eaf3D.ApproxPoint(0, 1006, np.array([1, 28, 7000]))
        q1 = self.t.floor_x(p1)
        q2 = self.t.floor_x(p2)
        q3 = self.t.floor_x(p3)
        q4 = self.t.floor_x(p4)
        q5 = self.t.floor_x(p5)
        self.assertEqual(q1.x, 3)
        self.assertEqual(q2.x, 4)
        self.assertEqual(q3.x, 5)
        self.assertEqual(q4.x, 5)
        self.assertEqual(q5.x, -10E10)

    def test_higher_x(self):
        p1 = eaf3D.ApproxPoint(1, 1002, np.array([3, 23.4623828, 6059.2348600000005]))
        p2 = eaf3D.ApproxPoint(1, 1003, np.array([4, 14.07345342, 5990.93696]))
        p3 = eaf3D.ApproxPoint(1, 1004, np.array([5, 10.90633272, 5965.522494]))
        p4 = eaf3D.ApproxPoint(1, 1005, np.array([7, 10.73267638, 5868.0173159999995]))
        p5 = eaf3D.ApproxPoint(0, 1006, np.array([1, 28, 7000]))
        q1 = self.t.higher_x(p1)
        q2 = self.t.higher_x(p2)
        q3 = self.t.higher_x(p3)
        q4 = self.t.higher_x(p4)
        q5 = self.t.higher_x(p5)
        self.assertEqual(q1.x, 4)
        self.assertEqual(q2.x, 5)
        self.assertEqual(q3.x, 10E10)
        self.assertEqual(q4.x, 10E10)
        self.assertEqual(q5.x, 3)

    def test_lower_y(self):
        p1 = eaf3D.ApproxPoint(1, 1002, np.array([3, 23.4623828, 6059.2348600000005]))
        p2 = eaf3D.ApproxPoint(1, 1003, np.array([4, 14.07345342, 5990.93696]))
        p3 = eaf3D.ApproxPoint(1, 1004, np.array([5, 10.90633272, 5965.522494]))
        p4 = eaf3D.ApproxPoint(1, 1005, np.array([7, 10.73267638, 5868.0173159999995]))
        p5 = eaf3D.ApproxPoint(0, 1006, np.array([9, 6.5, 7000]))
        q1 = self.t.lower_y(p1)
        q2 = self.t.lower_y(p2)
        q3 = self.t.lower_y(p3)
        q4 = self.t.lower_y(p4)
        q5 = self.t.lower_y(p5)
        self.assertAlmostEqual(q1.y, 20.21, places=2)
        self.assertAlmostEqual(q2.y, 12.67, places=2)
        self.assertAlmostEqual(q3.y, 10.42, places=2)
        self.assertAlmostEqual(q4.y, 10.42, places=2)
        self.assertEqual(q5.y, -10E10)

    def test_getRightMost(self):
        rightmost, stack, pivot = self.t.getRightMost(self.t.root)
        self.assertEqual(rightmost.point.x, 10E10)
        rightmost, stack, pivot = self.t.getRightMost(self.t.root.left)
        self.assertEqual(rightmost.point.x, 3)

    def test_list_nodes_domxy(self):
        p1 = eaf3D.ApproxPoint(1, 1002, np.array([3, 19.0, 6059.]))
        list = self.t.list_nodes_domxy(p1)
        self.assertListEqual(list, [self.t.root.left.right])

    def test_height(self):
        height = self.t.height(self.t.root)
        self.assertEqual(height, 3)

    def test_print_astree(self):
        self.t.print_astree()


class RemoveNodeTests(unittest.TestCase):

    def setUp(self):
        self.t = bst.AVLTree()
        # Import more data points
        fname = 'example/run01'
        exset = eaf3D.import_approximate_set(fname)
        x, m = eaf3D.multiset_sum([exset])
        # Q is X sorted in ascending order of the z coordinate
        self.qstack = Stack()
        xintoq = sorted(x.values(), key=attrgetter('z'))
        for i in range(len(xintoq)):
            self.qstack.push(xintoq[i])
        self.t.set_newroot(self.qstack.pop())
        # Add data points to tree
        while not self.qstack.isEmpty():
            p = self.qstack.pop()
            self.t.insert(p)
        p1 = eaf3D.ApproxPoint(None, 1001, np.array([12, 6.6, 0]))
        self.t.insert(p1)
        p2 = eaf3D.ApproxPoint(None, 1002, np.array([8, 9.7, 0]))
        self.t.insert(p2)
        p3 = eaf3D.ApproxPoint(None, 1002, np.array([10, 7.5, 0]))
        self.t.insert(p3)

    def test_remove_node_case1_norotation(self):
        # This tests if it correctly removes a node without children from
        # the tree
        node, theStack, pivot = self.t.getRightMost(self.t.root)
        self.t.remove_node(node.left)
        self.assertEqual(self.t.root.right.balance, 0)
        self.assertEqual(self.t.root.balance, 0)

    def test_remove_node_case1_wrotation(self):
        # This tests if it correctly removes a node without children and
        # performs the necessary rotation.
        self.t.remove_node(self.t.root.left.left)
        self.assertEqual(self.t.count, 7)
        self.assertNotEqual(self.t.root.balance, 2)
        self.check_node_balances(self.t)

    def test_remove_node_case2_norotation(self):
        # This tests if AVLTree correctly removes a node and connects its
        # one child to its parent node.
        rightmost, rightStack, pivot = self.t.getRightMost(self.t.root)
        # Get node for comparison
        child = rightmost.left
        # Remove node
        self.t.remove_node(rightmost)
        self.check_node_balances(self.t)
        newright, rightStack, pivot = self.t.getRightMost(self.t.root)
        self.assertEqual(newright, child)

    def test_remove_node_case3_longer_subtree(self):
        # This tests if AVLTree correctly removes a node, connects its
        # one child to its parent node, and performs the necessary rotation
        rightmost, rightStack, pivot = self.t.getRightMost(self.t.root.right.left)
        self.t.remove_node(self.t.root.right)
        # There would have been a left rotation at root.right
        self.assertEqual(self.t.root.right.left.point, rightmost.point)
        self.check_node_balances(self.t)

    def test_remove_node_case3_smaller_subtree(self):
        # This tests if AVLTree correctly removes a node, connects its
        # one child to its parent node, and performs the necessary rotation
        rightmost, rightStack, pivot = self.t.getRightMost(self.t.root.left.left)
        self.t.remove_node(self.t.root.left)
        self.assertEqual(self.t.root.left.point, rightmost.point)
        self.check_node_balances(self.t)

    def test_remove_node_case3_root(self):
        # This tests if AVLTree correctly removes the root node with
        # children
        # Try with root and long stack
        rightmost, rightStack, pivot = self.t.getRightMost(self.t.root.left)
        self.t.remove_node(self.t.root)
        self.assertEqual(self.t.root.point, rightmost.point)
        self.check_node_balances(self.t)
        # Try with no stack (get down to two nodes)
        for i in range(5):
            self.t.remove_node(self.t.root)
            # Make sure no errors pop up in removal
            self.check_node_balances(self.t)

    def test_remove_node_case3_rootright(self):
        # This tests for a specific problem that popped where the right
        # node was removed and the balance at the root was wrong
        # Get tree to position that caused error, checking for new errors
        self.t.remove_node(self.t.root)
        self.check_node_balances(self.t)
        self.t.remove_node(self.t.root.right.right)
        self.check_node_balances(self.t)
        # The tree should now be balance=0 at root with right with two kids
        # After removing right, the root should still be balanced
        self.t.remove_node(self.t.root.right)
        self.check_node_balances(self.t)

    def test_remove_node_case3_left_is_rightmost(self):
        # Set up tree to test case 3 when the left node is the rightmost
        tree = bst.AVLTree()
        point0 = eaf3D.ApproxPoint(1, 1, np.array([4, 12.67, 0]))
        tree.set_newroot(point0)
        # Set initial points for sentinels (to simulate infinity)
        big_pos_value = 10E10
        big_neg_value = -1 * big_pos_value
        p1 = np.array([big_neg_value, big_pos_value, big_neg_value])
        p2 = np.array([big_pos_value, big_neg_value, big_neg_value])
        # Initialize avltree
        point1 = eaf3D.ApproxPoint(None, 1000, p1)
        av1 = bst.AVLNode(point1, balance=1)
        tree.root.left = av1
        point2 = eaf3D.ApproxPoint(None, 1001, p2)
        av2 = bst.AVLNode(point2, balance=-1)
        tree.root.right = av2
        point3 = eaf3D.ApproxPoint(1, 2, np.array([3, 20.21, 0]))
        av3 = bst.AVLNode(point3)
        tree.root.left.right = av3
        point4 = eaf3D.ApproxPoint(1, 3, np.array([5, 10.42, 0]))
        av4 = bst.AVLNode(point4)
        tree.root.right.left = av4
        tree.count = 5
        tree.remove_node(tree.root)
        tree.remove_node(tree.root)
        self.assertEqual(tree.root, av4)
        self.check_node_balances(tree)

    def check_node_balances(self, tree):
        # This module performs a check of all the balances in the tree
        leaves = [tree.root]
        while any(leaves):
            for f in range(len(leaves)):
                if leaves[f]:
                    correct_balance = tree.recalculate_balance(leaves[f])
                    self.assertEqual(leaves[f].balance, correct_balance)
            leaves = tree.next_tree_row(leaves)


if __name__ == '__main__':
    unittest.main()
