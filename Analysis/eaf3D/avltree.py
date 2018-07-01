""" avltree.py

  The following script was copied and extended from Kent D. Lee and Steve
  Hubbard, Data Structures and Algorithms with Python, Chapter 10. This
  algorithm creates a height-balanced binary search tree.

"""

from stack import Stack


class AVLNode:
    def __init__(self, item, balance=0, left=None, right=None):
        self.item = item
        self.left = left
        self.right = right
        self.balance = balance

    def __str__(self):
        # This performs an inorder traversal of the tree rooted at self,
        # using recursion.  Return the corresponding string.
        st = str(self.item) + ' ' + str(self.balance) + '\n'
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
        if newItem < parent.item:
            parent.left = newNode
        else:
            parent.right = newNode
        # Adjust balances and pivot if necessary
        if pivot:
            # If to the left of pivot:
            if newItem < pivot.item:
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

    def adjustBalances(self, theStack, pivot, newNode):
        # We adjust the balances of all the nodes in theStack, up to and
        # including the pivot node, if any.  Later rotations may cause
        # some of the balances to change.
        child = newNode
        while not theStack.isEmpty():
            parent = theStack.pop()
            if child.item < parent.item:
                parent.balance -= 1
            else:
                parent.balance += 1
            if parent is pivot:
                return

    def case1(self, theStack, pivot, newNode):
        # There is no pivot node.  Adjust the balances of all the nodes
        # in theStack.
        self.adjustBalances(theStack, pivot, newNode)

    def case2(self, theStack, pivot, newNode):
        # The pivot node exists.  We have inserted a new node into the
        # subtree of the pivot of smaller height.  Hence, we need to adjust
        # the balances of all the nodes in the stack up to and including
        # that of the pivot node.  No rotations are needed.
        self.adjustBalances(theStack, pivot, newNode)

    def case3(self, theStack, pivot, newNode):
        # The pivot node exists.  We have inserted a new node into the
        # larger height subtree of the pivot node.  Hence rebalancing and
        # rotations are needed.
        self.adjustBalances(theStack, pivot, newNode)
        # Determine direction of imbalance
        if pivot.balance < 0:
            imbalance = 'left'
            badchild = pivot.left
        else:
            imbalance = 'right'
            badchild = pivot.right
        # Determine direction of newNode
        if newNode.item < badchild.item:
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
            pivot.balance += 2
            newroot.balance += 1
        else:
            newroot = pivot.rotateLeft()
            pivot.balance -= 2
            newroot.balance -= 1
        self.reconnect_subtree(theStack, pivot, newroot)

    def subcaseB(self, theStack, pivot):
        # The new node is added to the subtree of the bad child in the opposite
        # direction of the imbalance.
        # Determine pivot direction; Update node balances in subtree
        if pivot.balance < 0:
            newroot = pivot.rotateLeftThenRight()
            pivot.balance += 2
            newroot.balance -= 1
            newroot.left.balance -= 2
        else:
            newroot = pivot.rotateRightThenLeft()
            pivot.balance -= 2
            newroot.balance += 1
            newroot.right.balance += 2
        self.reconnect_subtree(theStack, pivot, newroot)

    def reconnect_subtree(self, theStack, pivot, newroot):
        # Reconnect subtree to tree
        if pivot is self.root:
            self.root = newroot
        else:
            prepivot = theStack.top()
            if pivot.item < prepivot.item:
                prepivot.left = newroot
            else:
                prepivot.right = newroot

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
            if newItem < leaf.item:
                if not leaf.left:
                    return (pivot, theStack, leaf, False)
                leaf = leaf.left
            # Right if item is more than the node item
            elif newItem > leaf.item:
                if not leaf.right:
                    return (pivot, theStack, leaf, False)
                leaf = leaf.right
            else:
                return (pivot, theStack, leaf, True)

    def height(self, node):
        # This function determines the height of any node in the tree.
        height = 0              # Each node has a height of 1
        while node:
            height += 1
            # If balance is -1, the longer branch is on the left
            if node.balance == -1:
                node = node.left
            # If balance is 1, the longer branch is on the right
            # If balance is 0, it doesn't matter which branch you use.
            else:
                node = node.right
        return height

    def print_astree(self):
        print('There are ' + str(self.count) + ' nodes in the AVL tree.\n')
        leaves = [self.root]
        while any(leaves):
            # Make level string
            st = ' '
            for f in leaves:
                if f:
                    st += '{0} {1}  '.format(f.item, f.balance)
                else:
                    st += '       '
            print(st)
            # Set next level
            newleaves = []
            for f in leaves:
                if f:
                    newleaves.append(f.left)
                    newleaves.append(f.right)
                else:
                    newleaves.append(None)
            leaves = newleaves


def main():
    a = AVLNode(20, -1)
    b = AVLNode(30, -1)
    c = AVLNode(-100)
    d = AVLNode(290)

    print("Testing Node Placement")
    expectation = 'There are 4 nodes in the AVL tree.\n' + \
                  '-100 0\n' + \
                  '20 -1\n' + \
                  '30 -1\n' + \
                  '290 0\n'
    t = AVLTree()
    t.root = b
    b.left = a
    a.left = c
    b.right = d
    t.count = 4
    tree_str = str(t)
    if tree_str == expectation:
        print(' - passed')
    else:
        print(' - failed. The following was returned:')
        print(tree_str)

    print("Testing rotateLeftThenRight()")
    expectation = '30 0\n' + '40 0\n' + '50 0\n'
    a = AVLNode(50)
    b = AVLNode(30)
    c = AVLNode(40)
    a.left = b
    b.right = c
    sequence_str = str(a.rotateLeftThenRight())
    if sequence_str == expectation:
        print(' - passed')
    else:
        print(' - failed. The following was returned:')
        print(sequence_str)

    print("Testing search function")
    exp1 = (20, -100, False)
    (pivot, theStack, parent, found) = t.search(-70)
    if (pivot.item, parent.item, found) == exp1:
        print(' - passed pivot and parent location for -70')
    else:
        print(' - failed pivot and parent location. Found:')
        print('pivot = {0}, parent = {1}, -70 exists in tree = {2}'.
              format(pivot.item, parent.item, found))
    exp2 = [-100, 20, 30]
    i = 0
    while not theStack.isEmpty():
        current = theStack.pop()
        if current.item != exp2[i]:
            print(' - failed stack test: expected {0}, received {1}'.
                  format(exp2[i], current.item))
        i += 1
    exp3 = (20, 20, False)
    (pivot, theStack, parent, found) = t.search(25)
    if (pivot.item, parent.item, found) == exp3:
        print(' - passed pivot and parent location for 25')
    else:
        print(' - failed pivot and parent location. Found:')
        print('pivot = {0}, parent = {1}, 25 exists in tree = {2}'.
              format(pivot.item, parent.item, found))
    exp4 = (20, -100, True)
    (pivot, theStack, parent, found) = t.search(-100)
    if (pivot.item, parent.item, found) == exp4:
        print(' - passed pivot and parent location for -100')
    else:
        print(' - failed pivot and parent location. Found:')
        print('pivot = {0}, parent = {1}, -100 exists in tree = {2}'.
              format(pivot.item, parent.item, found))


if __name__ == '__main__':
    main()
