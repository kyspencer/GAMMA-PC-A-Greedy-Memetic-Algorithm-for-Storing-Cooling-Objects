''' test_avltree.py

    This python scripts tests methods in avltree.py that the authors did not
    leave clues for.
        - insert()
        - adjustBalances()
        - case1()
        - case2()
        - case3()

'''

import unittest
import avltree
import stack


class AVLTreeTestsFromStarter(unittest.TestCase):

    def setUp(self):
        a = avltree.AVLNode(8, -1)
        b = avltree.AVLNode(18, -1)
        c = avltree.AVLNode(3)
        d = avltree.AVLNode(20)
        self.t = avltree.AVLTree()
        self.t.root = b
        b.left = a
        a.left = c
        b.right = d
        self.t.count = 4

    def test_insert_found(self):
        expected = str(self.t)
        self.t.insert(18)
        self.assertEqual(str(self.t), expected)

    def test_insert_case2(self):
        # Test insert function for adjust only
        self.t.insert(22)
        self.assertEqual(self.t.root.balance, 0)
        self.assertEqual(self.t.count, 5)

    def test_insert_case1(self):
        # First balance tree:
        self.t.insert(22)
        self.t.insert(17)
        self.t.insert(19)
        # Check to make sure tree is balanced
        (pivot, theStack, parent, found) = self.t.search(14)
        while not theStack.isEmpty():
            node = theStack.pop()
            self.assertEqual(node.balance, 0)
        # Then test for inserting into balanced tree
        self.t.insert(14)
        (pivot, theStack, parent, found) = self.t.search(14)
        theStack.pop()
        while not theStack.isEmpty():
            node = theStack.pop()
            self.assertNotEqual(node.balance, 0)

    def test_insert_case3_subcaseA(self):
        (pivot, theStack, parent, found) = self.t.search(1)
        self.t.insert(1)
        self.assertEqual(pivot.balance, 0)
        self.assertEqual(self.t.root.balance, -1)

    def test_insert_case3_subcaseB(self):
        self.t.insert(10)
        self.t.insert(12)
        self.assertEqual(self.t.root.balance, 0)
        self.assertEqual(self.t.root.right.item, 18)
        self.assertEqual(self.t.root.right.balance, 0)
        self.assertEqual(self.t.root.left.balance, -1)

    def test_adjustBalances_negative(self):
        (pivot, theStack, parent, found) = self.t.search(1)
        newNode = avltree.AVLNode(1)
        self.t.adjustBalances_add(theStack, pivot, newNode)
        self.assertEqual(pivot.balance, -2)

    def test_height(self):
        height = self.t.height(self.t.root)
        self.assertEqual(height, 3)

    def test_print_astree(self):
        self.t.print_astree()


class AVLTreeTestsFromBook(unittest.TestCase):

    def setUp(self):
        # Original tree
        rootitem = avltree.AVLNode(10.0)
        self.t = avltree.AVLTree(root=rootitem)
        items = [3.0, 18.0, 2.0, 4.0, 13.0, 40.0]
        for newItem in items:
            self.t.insert(newItem)

    def test_case1(self):
        self.t.insert(39)
        self.assertEqual(self.t.root.balance, 1)
        node = self.t.root.right
        self.assertEqual(node.balance, 1)
        node = node.right
        self.assertEqual(node.balance, -1)

    def test_case2(self):
        self.t.insert(39)
        self.t.insert(12)
        self.assertEqual(self.t.root.right.balance, 0)
        self.assertEqual(self.t.root.right.left.balance, -1)

    def test_case3_subcaseA(self):
        self.t.insert(39.0)
        self.t.insert(12.0)
        self.t.insert(38)
        self.assertEqual(self.t.root.balance, 1)
        node = self.t.root.right
        self.assertEqual(node.balance, 0)
        node = node.right
        self.assertEqual(node.balance, 0)
        self.assertEqual(node.item, 39)
        self.assertEqual(node.left.item, 38)
        self.assertEqual(node.right.item, 40)

    def test_case3_subcaseB(self):
        self.t.insert(39.0)
        self.t.insert(12.0)
        self.t.insert(38.0)
        self.t.insert(14.0)
        self.t.insert(11)
        self.assertEqual(self.t.root.item, 13.0)
        self.assertEqual(self.t.root.balance, 0)
        self.assertEqual(self.t.root.left.item, 10.0)
        self.assertEqual(self.t.root.left.balance, 0)
        self.assertEqual(self.t.root.right.item, 18.0)
        self.assertEqual(self.t.root.right.balance, 1)


if __name__ == '__main__':
    unittest.main()
