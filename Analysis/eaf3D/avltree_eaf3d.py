""" avltree_eaf3d.py

  The following script was developed for the eaf3D.py script, using
  avltree.py as a model. The algorithm for the 3D empirical attainment
  function requires the use of 2n data structures based on height-balanced
  binary search trees. The incoming data (i.e. item) are ApproxPoint objects
  where the fitvals attribute is used to perform the sorting in the tree.
  The ApproxPoint objects have fitness vectors with x, y, and z direction,
  but only x and y are used for storage.

  The authors did not give much guidance on 2n height-balanced binary search
  trees, so this is my guess at what they meant. Everything is sorted on the
  x-value.

  Search operations defined in article:
    - floor_x(p, X*): the point q belonging to X* with the greatest q_x <= p_x
    - lower_x(p, X*): the point q belonging to X* with the greatest q_x < p_x
    - ceiling_x(p, X*): the point q belonging to X* with the least q_x >= p_x
    - higher_x(p, X*): the point q belonging to X* with the least q_x > p_x
    These and their y-coordinate partners can be performed in logarithmic time
    using 2n data structures on a height-balanced binary search tree.

  Author: Kristina Yancey Spencer

  Refer to license.txt for permissions.

"""

from __future__ import print_function
import numpy as np
from stack import Stack


class AVLNode:
    def __init__(self, point, balance=0, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right
        self.balance = balance

    def __str__(self):
        # This performs an inorder traversal of the tree rooted at self,
        # using recursion.  Return the corresponding string.
        st = '({0}, {1:5.2f}) {2}\n'.format(self.point.x, self.point.y, self.balance)
        if self.left != None:
            st = str(self.left) + st    # A recursive call: str(self.left)
        if self.right != None:
            st = st + str(self.right)   # Another recursive call
        return st

    def rotateLeft(self):
        # Perform a left rotation of the subtree rooted at the
        # receiver.  Answer the root node of the new subtree.
        child = self.right
        if (child == None):
            print('Error!  No right child in rotateLeft.')
            return None  # redundant
        else:
            self.right = child.left
            child.left = self
            return child

    def rotateRight(self):
        # Perform a right rotation of the subtree rooted at the
        # receiver.  Answer the root node of the new subtree.
        child = self.left
        if (child == None):
            print('Error!  No left child in rotateRight.')
            return None  # redundant
        else:
            self.left = child.right
            child.right = self
            return child

    def rotateRightThenLeft(self):
        # Perform a double inside left rotation at the receiver.  We
        # assume the receiver has a right child (the bad child), which has a left
        # child. We rotate right at the bad child then rotate left at the pivot
        # node, self. Answer the root node of the new subtree.  We call this
        # case 3, subcase 2.
        child = self.right
        if not child:
            print('Error!  No right child in rotateRightThenLeft.')
            return None  # redundant
        else:
            # Perform single right rotation at bad child
            self.right = child.rotateRight()
            # Perform single rotation left at pivot (self)
            newroot = self.rotateLeft()
            return newroot

    def rotateLeftThenRight(self):
        # Perform a double inside right rotation at the receiver.  We
        # assume the receiver has a left child (the bad child) which has a right
        # child. We rotate left at the bad child, then rotate right at
        # the pivot, self.  Answer the root node of the new subtree. We call this
        # case 3, subcase 2.
        child = self.left
        if not child:
            print('Error!  No left child in rotateLeftThenRight.')
            return None  # redundant
        else:
            # Perform single left rotation at bad child
            self.left = child.rotateLeft()
            # Perform single rotation right at the pivot (self)
            newroot = self.rotateRight()
            return newroot

    def set_new_point(self, point):
        self.point = point


class AVLTree:
    def __init__(self, root=None):
        self.root = root
        self.count = 0

    def __str__(self):
        st = 'There are ' + str(self.count) + ' nodes in the AVL tree.\n'
        return st + str(self.root)  # Using the string hook for AVL nodes

    def insert(self, newItem):
        # Add a new node with item newItem, if there is not a match in the
        # tree.  Perform any rotations necessary to maintain the AVL tree,
        # including any needed updates to the balances of the nodes.  Most of the
        # actual work is done by other methods.
        (pivot, theStack, parent, found) = self.search(newItem)
        if found:
            return
        # Add node to tree
        self.count += 1
        newNode = AVLNode(newItem)
        if newItem.x < parent.point.x:
            parent.left = newNode
        else:
            parent.right = newNode
        # Adjust balances and pivot if necessary
        if pivot:
            # If to the left of pivot:
            if newItem.x < pivot.point.x:
                self.select_pivot_case(-1, theStack, pivot, newNode)
            # Else if to the right of pivot:
            else:
                self.select_pivot_case(1, theStack, pivot, newNode)
        else:
            self.case1(theStack, pivot, newNode)

    def select_pivot_case(self, signal, theStack, pivot, newNode):
        # This module performs the pivot in case 2 or case3 based on the signal
        # given
        if pivot.balance == signal:
            # In longer pivot branch
            self.case3(theStack, pivot, newNode)
        else:
            # In shorter pivot branch
            self.case2(theStack, pivot, newNode)

    def remove_node(self, node):
        # Find node in tree and create stack list
        pivot, theStack, parent = self.find_node(node.point)
        self.count -= 1
        if not node.left and not node.right:
            # No children
            self.delete1(theStack, pivot, node)
        elif node.left and node.right:
            # Two children
            self.delete3(theStack, pivot, node)
        else:
            # One child
            self.delete2(theStack, pivot, node)

    def adjustBalances_add(self, theStack, pivot, newNode):
        # We adjust the balances of all the nodes in theStack, up to and
        # including the pivot node, if any.  Later rotations may cause
        # some of the balances to change.
        child = newNode
        while not theStack.isEmpty():
            parent = theStack.pop()
            if child.point.x < parent.point.x:
                parent.balance -= 1
            else:
                parent.balance += 1
            if parent is pivot:
                return

    def adjustBalances_remove(self, theStack, pivot, node):
        # Adjust the balances of all the nodes in theStack. If a new balance
        # value reaches 2, a left rotation is required to rebalance the tree.
        # If it reaches -2, then a right rotation is required. These rotations
        # may cascade all the way up the path to the root of the tree.
        # child = node
        while not theStack.isEmpty():
            parent = theStack.pop()
            # if child.point.x <= parent.point.x:
            #     parent.balance += 1
            #     print(parent.balance)
            # else:
            #     parent.balance -= 1
            #     print(parent.balance)
            parent.balance = self.recalculate_balance(parent)
            if parent.balance == -2 or parent.balance == 2:
                if self.height(parent) >= 3:
                    if parent.balance == -2 and parent.left.balance <= 0:
                        self.subcaseA(theStack, parent)
                    elif parent.balance == 2 and parent.right.balance >= 0:
                        self.subcaseA(theStack, parent)
                    else:
                        self.subcaseB(theStack, parent)
                else:
                    self.subcaseA(theStack, parent)
            if parent is pivot:
                return

    def case1(self, theStack, pivot, newNode):
        # There is no pivot node.  Adjust the balances of all the nodes
        # in theStack.
        self.adjustBalances_add(theStack, pivot, newNode)

    def case2(self, theStack, pivot, newNode):
        # The pivot node exists.  We have inserted a new node into the
        # subtree of the pivot of smaller height.  Hence, we need to adjust
        # the balances of all the nodes in the stack up to and including
        # that of the pivot node.  No rotations are needed.
        self.adjustBalances_add(theStack, pivot, newNode)

    def case3(self, theStack, pivot, newNode):
        # The pivot node exists.  We have inserted a new node into the
        # larger height subtree of the pivot node.  Hence rebalancing and
        # rotations are needed.
        self.adjustBalances_add(theStack, pivot, newNode)
        # Determine direction of imbalance
        if pivot.balance < 0:
            imbalance = 'left'
            badchild = pivot.left
        else:
            imbalance = 'right'
            badchild = pivot.right
        # Determine direction of newNode
        if newNode.point.x < badchild.point.x:
            direction = 'left'
        else:
            direction = 'right'
        # Select subcase
        if imbalance == direction:
            self.subcaseA(theStack, pivot)
        else:
            self.subcaseB(theStack, pivot)

    def subcaseA(self, theStack, pivot):
        # The new node is added to the subtree of the bad child in the direction
        # of the imbalance. The solution is to rotate the pivot node in the opposite
        # direction of the imbalance.
        # Determine pivot direction; Update node balances in subtree
        if pivot.balance < 0:
            newroot = pivot.rotateRight()
            pivot.balance = self.recalculate_balance(pivot)
            newroot.balance = self.recalculate_balance(newroot)
        else:
            newroot = pivot.rotateLeft()
            pivot.balance = self.recalculate_balance(pivot)
            newroot.balance = self.recalculate_balance(newroot)
        self.reconnect_subtree(theStack, pivot, newroot)

    def subcaseB(self, theStack, pivot):
        # The new node is added to the subtree of the bad child in the opposite
        # direction of the imbalance.
        # Determine pivot direction; Update node balances in subtree
        if pivot.balance < 0:
            newroot = pivot.rotateLeftThenRight()
            pivot.balance = self.recalculate_balance(pivot)
            newroot.left.balance = self.recalculate_balance(newroot.left)
            newroot.balance = self.recalculate_balance(newroot)
        else:
            newroot = pivot.rotateRightThenLeft()
            pivot.balance = self.recalculate_balance(pivot)
            newroot.right.balance = self.recalculate_balance(newroot.right)
            newroot.balance = self.recalculate_balance(newroot)
        self.reconnect_subtree(theStack, pivot, newroot)

    def delete1(self, theStack, pivot, node):
        # Case 1 of removing a node: the node doesn't have children. Replace
        # node with None in parent node.
        if node is self.root:
            self.root = None
        else:
            parent = theStack.top()
            if node.point.x < parent.point.x:
                parent.left = None
            else:
                parent.right = None
            self.adjustBalances_remove(theStack, pivot, node)

    def delete2(self, theStack, pivot, node):
        # Case 2 of removing a node: the node has one child. Replace node
        # with child in node's parent.
        # Set child node
        if node.balance == -1:
            child = node.left
        else:
            child = node.right
        if node is self.root:
            self.root = child
        else:
            parent = theStack.top()
            # Connect to parent node
            if node.point.x < parent.point.x:
                parent.left = child
            else:
                parent.right = child
            self.adjustBalances_remove(theStack, pivot, node)

    def delete3(self, theStack, pivot, node):
        # Case 3 of removing a node: the node has two children. First determine
        # the right most value of the left subtree of the node to delete. Instead
        # of deleting node, replace its value with the value from the right-most
        # node of the subtree. Then delete the right-most node of the subtree.
        # Add node back to theStack to be adjusted
        theStack, pivot = self.add_node_to_stack(node, theStack, pivot)
        rightmost, theStack, rpivot = self.getRightMost(node.left, stack=theStack)
        if rpivot:
            pivot = rpivot
        node.set_new_point(rightmost.point)
        if rightmost is not node.left:
            # The right most node of the subtree won't have two children:
            parent = theStack.top()
            # right most node will be at the right and might have a left branch
            if rightmost.left:
                parent.right = rightmost.left
            else:
                parent.right = None
        else:
            if rightmost.left:
                node.left = rightmost.left
            else:
                node.left = None
        self.adjustBalances_remove(theStack, pivot, rightmost)

    def reconnect_subtree(self, theStack, pivot, newroot):
        # Reconnect subtree to tree
        if pivot is self.root:
            self.root = newroot
        else:
            prepivot = theStack.top()
            if pivot.point.x < prepivot.point.x:
                prepivot.left = newroot
            else:
                prepivot.right = newroot

    def add_node_to_stack(self, node, theStack, pivot):
        # This function adds a node to theStack that was left out before
        # (specifically for delete3). It also updates pivot.
        theStack.push(node)
        if node.balance == 0:
            pivot = node
        return theStack, pivot

    def search(self, newItem):
        # The AVL tree is not empty.  We search for newItem. This method will
        #   return a tuple: (pivot, theStack, parent, found).
        #   In this tuple, if there is a pivot node, we return a reference to it
        #   (or None). We create a stack of nodes along the search path -- theStack.
        #   We indicate whether or not we found an item which matches newItem.  We
        #   also return a reference to the last node the search examined -- referred
        #   to here as the parent.  (Note that if we find an object, the parent is
        #   reference to that matching node.)  If there is no match, parent is a
        #   reference to the node used to add a child in insert().
        pivot = None
        theStack = Stack()
        leaf = self.root
        while leaf:
            theStack.push(leaf)
            if leaf.balance != 0:
                pivot = leaf
            # Left if item is less than the node item
            if newItem.x < leaf.point.x:
                if not leaf.left:
                    return (pivot, theStack, leaf, False)
                leaf = leaf.left
            # Right if item is more than the node item
            elif newItem.x > leaf.point.x:
                if not leaf.right:
                    return (pivot, theStack, leaf, False)
                leaf = leaf.right
            else:
                return (pivot, theStack, leaf, True)

    def find_node(self, point):
        # The AVL tree is not empty.  We search for newItem. This method will
        #   return a tuple: (pivot, theStack, parent, found).
        #   In this tuple, if there is a pivot node, we return a reference to it
        #   (or None). We create a stack of nodes along the search path -- theStack.
        #   We indicate whether or not we found an item which matches newItem.  We
        #   also return a reference to the last node the search examined -- referred
        #   to here as the parent.  (Note that if we find an object, the parent is
        #   reference to that matching node.)  If there is no match, parent is a
        #   reference to the node used to add a child in insert().
        pivot = None
        theStack = Stack()
        leaf = self.root
        while leaf:
            if leaf.point is point:
                return pivot, theStack, leaf
            else:
                theStack.push(leaf)
                if leaf.balance == 0:
                    pivot = leaf
                # Left if item is less than the node item
                if point.x < leaf.point.x:
                    leaf = leaf.left
                # Right if item is more than the node item
                elif point.x > leaf.point.x:
                    leaf = leaf.right
        # If reached bottom of tree without finding point, throw error
        print('Error!  Cannot find point in tree.')
        print(point)

    def floor_x(self, p):
        # This function performs the search: the point q with the
        # greatest q_x <= p_x. This should always return a value since
        # the tree contains x='infinity' and x=-'infinity'.
        # Anything to the left of a node will be smaller than the current
        # node, so it is only necessary to search to the right.
        leaf = self.root
        while leaf:
            if leaf.point.x <= p.x:
                # Check that the right node is not also smaller
                if leaf.right:
                    child = leaf.right
                    # If the leaf at the right is also smaller, move that way
                    if child.point.x <= p.x:
                        leaf = child
                    # the leaf to the left of leaf's right hand could also be smaller
                    elif child.left and child.left.point.x <= p.x:
                        leaf = child.left
                    else:
                        return leaf.point
                else:
                    # Return greatest q_x found
                    return leaf.point
            else:
                # If leaf's x-value is > p_x, then need to move left
                leaf = leaf.left

    def higher_x(self, p):
        # This function performs the search: the point q with the least
        # q_x > p_x. This should always return a value since
        # the tree contains x='infinity' and x=-'infinity'.
        # Anything to the right will be greater than the current node,
        # so the search is concentrated toward the left.
        leaf = self.root
        while leaf:
            if leaf.point.x > p.x:
                # Check that the left node is not also greater
                if leaf.left:
                    child = leaf.left
                    # If the leaf to the left is also greater, move that way
                    # (toward smaller values that are greater than p.x)
                    if child.point.x > p.x:
                        leaf = child
                    # the leaf to the right could also be larger
                    elif child.right and child.right.point.x > p.x:
                        leaf = child.right
                    else:
                        return leaf.point
                else:
                    # Return smallest q_x found
                    return leaf.point
            else:
                # If leaf's x-value is < p_x, then need to move right
                leaf = leaf.right

    def lower_y(self, p):
        # This function performs the search: the point with the
        # greatest q_y < p_y. This should always return a value since
        # the tree contains y='infinity' and y=-'infinity'. If only
        # nondominated solutions added, the smaller values of y will be
        # located more to the right in the tree. So anything the right
        # of a node should have smaller y values than the current node,
        # so it is only necessary to search on the left.
        leaf = self.root
        while leaf:
            if leaf.point.y < p.y:
                # Check that the left node is not also smaller
                if leaf.left:
                    child = leaf.left
                    # If the leaf at the left is also smaller, move that way
                    if child.point.y < p.y:
                        leaf = child
                    # the leaf to the right of leaf's left hand could also be smaller
                    elif child.right and child.right.point.y < p.y:
                        leaf = child.right
                    else:
                        return leaf.point
                else:
                    # Return greatest q_x found
                    return leaf.point
            else:
                # If leaf's y-value is > p_y, then need to move right
                leaf = leaf.right

    def getRightMost(self, node, stack=Stack()):
        # This function determines the right-most point of the subtree
        # started at node.
        # Found the right-most point when it no longer has a right child
        pivot = None
        while node.right:
            if node.balance == 0:
                pivot = node
            stack.push(node)
            node = node.right
        return node, stack, pivot

    def list_nodes_domxy(self, u):
        # This module checks the tree to see if any nodes are dominated
        # by point u and returns the resulting list.
        # Possible area for improvement: once a subtree has all x < u.x
        # stop searching subtree
        uvec = np.array([u.x, u.y])
        nodes = []
        leaves = [self.root]
        while any(leaves):
            for f in leaves:
                if f:
                    vvec = np.array([f.point.x, f.point.y])
                    if dom2(uvec, vvec):
                        nodes.append(f)
            leaves = self.next_tree_row(leaves)
        return nodes

    def recalculate_balance(self, node):
        # This module calculates the new balance of a node
        balance = self.height(node.right) - self.height(node.left)
        return balance

    def height(self, node):
        # This function determines the height of any node in the tree.
        height = 0              # Each node has a height of 1
        while node:
            height += 1
            # If balance is -1, the longer branch is on the left
            if node.balance < 0:
                node = node.left
            # If balance is 1, the longer branch is on the right
            # If balance is 0, it doesn't matter which branch you use.
            else:
                node = node.right
        return height

    def set_newroot(self, approxpoint):
        root_node = AVLNode(approxpoint)
        self.root = root_node
        self.count += 1

    def print_astree(self, root=None):
        print('There are ' + str(self.count) + ' nodes in the AVL tree.\n')
        if root:
            leaves = [root]
        else:
            leaves = [self.root]
        while any(leaves):
            # Make level string
            st = ' '
            for f in leaves:
                if f:
                    st += '({0}, {1:5.2f}) {2}  '.\
                        format(f.point.x, f.point.y, f.balance)
                else:
                    st += '       '
            print(st)
            leaves = self.next_tree_row(leaves)

    def next_tree_row(self, leaves):
        # Set next level
        newleaves = []
        for f in leaves:
            if f:
                newleaves.append(f.left)
                newleaves.append(f.right)
            else:
                newleaves.append(None)
        return newleaves


def dom2(u, v):
    # Determines if fitness vector u dominates fitness vector v
    # This function assumes a minimization problem.
    # For u to dominate v, every fitness value must be either
    # equal to or less than the value in v AND one fitness value
    # must be less than the one in v

    equaltest = np.allclose(u, v)
    if equaltest is True:
        # If u == v then nondominated
        return False
    # less_equal returns boolean for each element u[i] <= v[i]
    domtest = np.less_equal(u, v)
    return np.all(domtest)


def main():
    class Point:
        # Quick class to approximate ApproxPoint in eaf3D
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    p1 = Point(3, 23.4623828, 6059.2348600000005)
    p2 = Point(4, 14.07345342, 5990.93696)
    p3 = Point(5, 10.90633272, 5965.522494)
    p4 = Point(7, 10.73267638, 5868.0173159999995)
    # Set initial points for sentinels (to simulate infinity)
    big_pos_value = 10E10
    big_neg_value = -1 * big_pos_value
    p0 = Point(big_neg_value, big_pos_value, big_neg_value)
    p5 = Point(big_pos_value, big_neg_value, big_neg_value)

    tree = AVLTree()
    node0 = AVLNode(p0)
    tree.root = node0
    for point in [p1, p2, p3, p4, p5]:
        tree.insert(point)
    tree.print_astree()


if __name__ == '__main__':
    main()
