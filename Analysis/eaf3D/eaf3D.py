""" eaf3D.py

   This python script performs Algorithm 4 described in C.M. Fonseca et. al.,
   "On the Computation of the Empirical Attainment Function," (2011):
   Algorithm 4. EAF computation in three dimensions

   Basic operations defined in article:
    - top(Q) returns the element at the top of a queue
    - pop(Q) retrieves the element at the top and removes it from Q
        --> same as python list function pop()
    - input_set(p) returns the index of the input set containing p

   Search operations defined in article:
    - floor_x(p, X*): the point q belonging to X* with the greatest q_x <= p_x
    - lower_x(p, X*): the point q belonging to X* with the greatest q_x < p_x
    - ceiling_x(p, X*): the point q belonging to X* with the least q_x >= p_x
    - higher_x(p, X*): the point q belonging to X* with the least q_x > p_x
    These and their y-coordinate partners can be performed in logarithmic time
    using 2n data structures on a height-balanced binary search tree. These are
    implemented in avltree_eaf3d.py.

   Indices in the algorithm are translated here from (1,...,n) to (0,...,n-1).


   kwargs:
        - SuperLevels: list of integers corresponding to the desired
          attainment surfaces to be present in the graph
        - MethodName: string naming the method being considered
        - opcat: list of 3 strings naming each objective
        - opal: integer correlating to color position in colors array

   Refer to license.txt for permissions.

"""

from __future__ import print_function
import avltree_eaf3d as bst
import numpy as np
import pandas as pd
import seaborn
from copy import deepcopy
from glob import glob
from matplotlib import pyplot, ticker
from operator import attrgetter
from stack import Stack

# Set environment for graphs
colors = ['#49ADA2', '#7797F4', '#C973F4', '#EF6E8B', '#FFAA6C']


class EAF_3D:
    # This class takes a sequence of nondominated point sets and transforms the
    # data into a sequence of summary attainment surfaces
    def __init__(self, sets):
        self.n = len(sets)
        self.x, m = multiset_sum(sets)
        self.a_tracker = []
        self.tmax = 0
        # Q is X sorted in ascending order of the z coordinate
        self.qstack = Stack()
        xintoq = sorted(self.x.values(), key=attrgetter('z'))
        for i in range(len(xintoq)):
            self.qstack.push(xintoq[i])

        # Set initial points for sentinels (to simulate infinity)
        big_pos_value = 10E10
        big_neg_value = -1 * big_pos_value
        p0_array = np.array([big_neg_value, big_pos_value, big_neg_value])
        self.p0 = ApproxPoint(None, None, p0_array)
        p1_array = np.array([big_pos_value, big_neg_value, big_neg_value])
        self.p1 = ApproxPoint(None, None, p1_array)
        lsa, lstar, xstar = init_surface_sentinels(self.n, self.p0, self.p1)
        self.lsa = lsa
        self.lstar = lstar
        self.xstar = xstar
        self.initialize()

    def initialize(self):
        # This module performs the initial steps of EAF-3D
        p = self.qstack.pop()
        nondominated = verify_nondominated(p.point, self.x)
        while not nondominated:
            p = self.qstack.pop()
            nondominated = verify_nondominated(p.point, self.x)
        j = p.input_set()
        # insert p into X*_j
        self.xstar[j].insert(p)
        # insert p into L*_1
        self.lstar[0].insert(p)
        self.a_tracker.append(j)

    def transform(self):
        # This module performs the while loop of Algorithm 4.
        while not self.qstack.isEmpty():
            p = self.qstack.pop()
            j = p.input_set()
            q = self.xstar[j].floor_x(p)
            if p.y < q.y:
                t, tmin = self.tmax, 0
                s, tmin = self.find_attainment_point(p, q, t, tmin)
                s = self.compare_p_to_surfaces(s, p, q, j, tmin)
                self.submit_points_lstar(s, p, q, tmin)
                self.submit_to_xstar(p, j)
            if j not in self.a_tracker:
                self.a_tracker.append(j)
                self.tmax = min(self.tmax + 1, self.n - 2)
        self.fill_attainment_surfaces()

    def find_attainment_point(self, p, q, t, tmin):
        # This module seeks output points r that X_j has not attained such
        # such that (px, pz) >= (rx, rz) and py < ry. Then s=(px, ry, pz) is
        # an element of J_t+1
        s = [None for _ in range(self.n)]
        while t >= tmin:
            r = self.lstar[t].floor_x(p)
            if r.y <= p.y:
                tmin = t + 1
            elif r.y < q.y:
                s[t] = ApproxPoint(None, None, np.array([p.x, r.y, p.z]))
            else:
                s[t] = self.lstar[t].lower_y(q)
            t -= 1
        return s, tmin

    def compare_p_to_surfaces(self, s, p, q, j, tmin):
        # This module seeks all output points belonging to L_t that Xj has not
        # attained and determines elements of L_t+1
        # Repeat this loop until q.y <= p.y
        while q.y > p.y:
            q = self.xstar[j].higher_x(q)
            b = max(p.y, q.y)
            for t in range(self.tmax, tmin - 1, -1):
                while s[t].y >= b and (s[t].y > b or b > p.y):
                    if s[t].x >= q.x:
                        s[t] = self.lstar[t].lower_y(q)
                    else:
                        # Make new point for submission
                        combo_point = ApproxPoint(None, None, np.array([s[t].x, s[t].y, p.z]))
                        self.submit_to_lstar(combo_point, t+1)
                        s[t] = self.lstar[t].higher_x(s[t])
        return s

    def submit_points_lstar(self, s, p, q, tmin):
        # This module finds output points similar to compare_p_to_surfaces,
        # except the roles of x and y are reversed.
        for t in range(self.tmax, tmin - 1, -1):
            if s[t].x < q.x:
                # Make new point for submission
                point = ApproxPoint(None, None, np.array([s[t].x, p.y, p.z]))
                self.submit_to_lstar(point, t+1)
        self.submit_to_lstar(p, tmin)

    def submit_to_lstar(self, u, t):
        # This algorithm (Algorithm 5 in article) submits point u to L*_t
        v = self.lstar[t].floor_x(u)
        if u.y < v.y:
            omegas = self.lstar[t].list_nodes_domxy(u)
            while omegas:
                if u.z > omegas[0].point.z:
                    self.lsa[t].append(omegas[0].point)
                self.lstar[t].remove_node(omegas[0])
                omegas = self.lstar[t].list_nodes_domxy(u)
            self.lstar[t].insert(u)
            self.check_node_balances(t)

    def submit_to_xstar(self, u, j):
        # This algorithm submits point u to X*_j
        v = self.xstar[j].floor_x(u)
        if u.y < v.y:
            omegas = self.xstar[j].list_nodes_domxy(u)
            while omegas:
                self.xstar[j].remove_node(omegas[0])
                omegas = self.xstar[j].list_nodes_domxy(u)
            self.xstar[j].insert(u)

    def fill_attainment_surfaces(self):
        # This algorithm performs line 48 (in the article) of the
        # overall algorithm.
        for t in range(self.n):
            leaves = [self.lstar[t].root]
            while any(leaves):
                for f in range(len(leaves)):
                    if leaves[f]:
                        approxpoint = leaves[f].point
                        sent1 = np.array_equal(approxpoint.point, self.p0.point)
                        sent2 = np.array_equal(approxpoint.point, self.p1.point)
                        # Do not include sentinels
                        if not sent1 and not sent2:
                            self.lsa[t].append(approxpoint)
                leaves = self.lstar[t].next_tree_row(leaves)

    def make_lsa_dataframe(self, **kwargs):
        # This module transforms the attainment surfaces into a pandas
        # dataframe with columns: x, y, z, super level set
        # If superlevel set numbers are not requested in kwargs, select all
        superlevels = list(range(self.n))
        opcat = ['Objective 1', 'Objective 2', 'Objective 3']
        if 'SuperLevels' in kwargs:
            superlevels = kwargs['SuperLevels']
        if 'opcat' in kwargs:
            opcat = kwargs['opcat']
        dlist_lsa = []
        for t in superlevels:
            for p in range(len(self.lsa[t])):
                # Add point to dict
                dlist_lsa.append({
                    opcat[0]: self.lsa[t][p].point[0],
                    opcat[1]: self.lsa[t][p].point[1],
                    opcat[2]: self.lsa[t][p].point[2],
                    'SuperLevel t/n [%]': int(100 * (t + 1.0) / self.n)
                })
        df_lsa = pd.DataFrame(dlist_lsa)
        return df_lsa

    def make_x_dataframe(self, **kwargs):
        # This module transforms X into a pandas dataframe with columns:
        # x, y, z
        opcat = ['Objective 1', 'Objective 2', 'Objective 3']
        if 'opcat' in kwargs:
            opcat = kwargs['opcat']
        dlist_x = []
        for p in range(len(self.x)):
            # Add point to dict
            dlist_x.append({
                opcat[0]: self.x[p].point[0],
                opcat[1]: self.x[p].point[1],
                opcat[2]: self.x[p].point[2],
                'Set': self.x[p].input_set()
            })
        df_x = pd.DataFrame(dlist_x)
        return df_x

    def graph_eaf(self, **kwargs):
        # This module creates a 2D scatter matrix plot of the empirical
        # attainment function inside folder.
        # Sort through keyword arguments
        opcat = ['Objective 1', 'Objective 2', 'Objective 3']
        plotname = 'eaf_ScatterMatrix'
        opal = 0
        if 'folder' in kwargs:
            plotname = kwargs['folder'] + plotname
        if 'MethodName' in kwargs:
            plotname = plotname + '_' + kwargs['MethodName']
        if 'opcat' in kwargs:
            opcat = kwargs['opcat']
        if 'opal' in kwargs:
            opal = kwargs['opal']
        df_lsa = self.make_lsa_dataframe(**kwargs)
        pal = seaborn.light_palette(colors[opal], reverse=True)
        scat = seaborn.PairGrid(df_lsa, vars=opcat, hue='SuperLevel t/n [%]',
                                palette=pal)
        scat = scat.map_diag(pyplot.hist)
        scat = scat.map_offdiag(pyplot.scatter, linewidths=1, edgecolor="w", s=40)
        # Set the tick labels to be at a 45 degree angle for better fit
        for ax in scat.axes.flat:
            ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda xax, p: format(int(xax))))
            pyplot.setp(ax.get_xticklabels(), rotation=45)
        scat.add_legend(frameon=True)
        scat.fig.get_children()[-1].set_bbox_to_anchor((0.995, 0.925, 0, 0))
        pyplot.savefig(plotname + '.eps', format='eps', dpi=4000)
        pyplot.savefig(plotname + '.pdf', format='pdf', dpi=4000)
        pyplot.close()

    def get_attainment_surfaces(self, t=None):
        if t or t == 0:
            return self.lsa[t]
        else:
            return self.lsa

    def check_node_balances(self, t):
        # This module performs a check of all the balances in the tree
        tree = self.lstar[t]
        leaves = [tree.root]
        while any(leaves):
            for f in range(len(leaves)):
                if leaves[f]:
                    correct_balance = tree.recalculate_balance(leaves[f])
                    if leaves[f].balance != correct_balance or \
                                    abs(leaves[f].balance) > 1:
                        tree.print_astree()
                        print(leaves[f].point)
                        message = 'Error!  Incorrect balance on tree {0}: {1}.' \
                                  'The balance should be {2} and always within ' \
                                  '[-1, 1].'.format(t, leaves[f].balance, correct_balance)
                        raise RuntimeError(message)
            leaves = tree.next_tree_row(leaves)


def multiset_sum(sets):
    # This function takes the sequence of nondominated point sets and returns
    # a multiset sum, allowing duplicate points
    x = {}
    m = 0
    idnum = 0
    # Add solutions to X
    for xi in range(len(sets)):
        m += len(sets[xi])
        for k, v in sets[xi].items():
            point = ApproxPoint(xi, k, v)
            x[idnum] = point
            idnum += 1
    return x, m


def verify_nondominated(v, x):
    # This module checks v against all the other fitness vectors in X,
    # removing any dominated solution
    for ki, pi in x.items():
        # Return False if dominated by another solution in x
        if dom1(pi.point, v):
            return False
    # If made it through the whole list, nondominated
    return True


def dom1(u, v):
    # dom(u, v) determines if fitness value u dominates fitness value v
    # to find the Pareto set
    nobj = 3
    equal = 0
    for i in range(nobj):
        if u[i] == v[i]:
            equal += 1
    u1, u2, u3 = u[0], u[1], u[2]
    v1, v2, v3 = v[0], v[1], v[2]
    return equal != nobj and u1 <= v1 and u2 <= v2 and u3 <= v3


def init_surface_sentinels(n, p0, p1):
    # This module initializes the empty attainment surfaces and the sentinels
    # Summary attainment surface
    lsa = [[] for _ in range(n)]
    # Make tree with Sentinels
    tree = bst.AVLTree()
    tree.set_newroot(p0)
    tree.insert(p1)
    # Copy tree n times to lstar and xstar
    lstar = []
    xstar = []
    for t in range(n):
        lstar.append(deepcopy(tree))
        xstar.append(deepcopy(tree))
    return lsa, lstar, xstar


def retrieve_input_sequences(folder):
    # This function imports the data in the text files in folder and returns a
    # dict of numpy arrays for every sequence, contained in a larger list.
    filenames = glob(folder + '*')
    sets = []
    for f in range(len(filenames)):
        set_f = import_approximate_set(filenames[f])
        sets.append(set_f)
    return sets


def import_approximate_set(fname):
    df_set = pd.read_csv(fname, sep='\t', index_col=0)
    approxset = df_set.to_dict(orient='index')
    # Transform items into np.arrays
    for k, v in approxset.items():
        fitvals = np.zeros(3, dtype=object)
        fitvals[0] = int(v.get('f[1] (B)'))
        fitvals[1] = float(v.get('f[2] (W)'))
        fitvals[2] = float(v.get('f[3] (s)'))
        approxset[k] = fitvals
    return approxset


class ApproxPoint:
    # This class maintains a connection between each point and its approximate set
    def __init__(self, xi, key, fitvals):
        self.set = xi
        self.key = key
        self.point = fitvals
        self.x = fitvals[0]
        self.y = fitvals[1]
        self.z = fitvals[2]

    def input_set(self):
        return self.set

    def __str__(self):
        return 'ApproxPoint at {0}.'.format(self.point)


def main():
    # Set the number of nondominated point sets to be input
    # Set the location of the input files
    folder = 'GAMMA-PC/Cookies1000/'
    sets = retrieve_input_sequences(folder)
    eaf_maker = EAF_3D(sets)
    eaf_maker.transform()

    kwargs = {'MethodName': 'GAMMA-PC',
              'opcat': ['No. of Bins', 'Avg. Initial Bin Heat (W)', 'Max. Time to Move (s)'],
              'folder': 'GAMMA-PC/',
              'opal': 0}
    df_lsa = eaf_maker.make_lsa_dataframe()
    print(df_lsa)
    eaf_maker.graph_eaf(**kwargs)


if __name__ == '__main__':
    main()
