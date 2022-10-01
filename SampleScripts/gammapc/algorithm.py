# algorithm.py
#    Script to develop algorithm for dynamic benchmark problems
#    Author: Kristina Spencer
#    g - generation index
#    i - bin index
#    j - item index
#    k - algorithm index
#    m - solution index
#    t - time index

from __future__ import print_function
import csv
import h5py
import numpy as np
import random
import sys
from datetime import datetime
from glob import glob
from operator import attrgetter
from os import mkdir, path

from . import coolcookies, ga, grasp, outformat as outf, mooproblem as mop


def algorithm(n, folder, datafile):
    existing_files = glob(folder + '*.out')
    fname = folder + 'run%d.out' % (len(existing_files) + 1)
    data = folder + datafile

    # Initialize algorithm
    batchsize = 100
    boxcap = 24
    rackcap = 300
    fillcap = 8
    pop = 100           # members per gen.
    end = 25000         # function evaluations
    # batchsize = 6
    # boxcap = 8
    # rackcap = 15
    # fillcap = 2
    # pop = 50            # members per gen.
    # end = 750           # function evaluations
    outf.startout(fname, n, end, data, 'GAMMA-PC')
    startt = datetime.now()
    print('                         ', startt)
    print('*******************************************************************************')
    print('     Method: GAMMA-PC\n')
    print('     data: ', datafile)
    print('*******************************************************************************')
    cookies = coolcookies.makeobjects(n, batchsize, data)
    moop = mop.MOCookieProblem(n, boxcap, rackcap, fillcap, cookies)
    bpp = grasp.BPP(n, cookies, moop)
    gen = Generation(n, pop, end, cookies, bpp, moop)
    gen.initialq(folder + 'seed.txt')
    # Remove this after finishing
    import warnings
    warnings.simplefilter('error', RuntimeWarning)

    # Algorithm
    while not gen.endgen():
        outf.gen_probabilities(fname, gen.gett() + 1, **gen.get_probabilities())
        gen.rungen()
        outf.genout(fname, gen.gett(), pop, gen.getq(), gen.getarch(),
                    onlyapprox=True)

    # Make output
    ndset = gen.finalapproxset()
    savefitvals(ndset, folder)
    savexys(ndset, folder)
    see(ndset, folder)
    outf.endout(fname)
    print('This algorithm has completed successfully.')


class Generation:
    def __init__(self, n, popsize, end, cookies, bpp, moop):
        self.n = int(n)             # Number of cookies to be sorted
        self.pop = int(popsize)     # Size of a generation population
        self.end = int(end)         # Number of function evaluations to perform
        self.items = cookies        # Dictionary of objects to be sorted
        self.bpp = bpp              # GRASP heuristic class
        self.moop = moop            # Multiobjective problem class
        self.alpha = 0.10           # value restriction
        self.idnum = 0              # Solution id number counter
        self.g = 0                  # Generation number counter
        self.p = []
        self.q = []
        self.newgenes = []
        self.archive = {}           # Nondominated archive
        self.funkeval = 0           # Function evaluation counter
        # ##    Initialize operation application rates  ##
        self.prob_xover = 0.8       # p. of crossover
        self.frac_random = 0.5      # fraction of pop sent to random crossover
        self.frac_clustr = 0.5      # fraction of pop sent to clustered crossover
        self.ideal_values = np.zeros(self.moop.nobj)
        self.obj_weights = np.zeros(self.moop.nobj)
        self.prob_mutat = 0.3       # p. of mutation
        self.prob_ls = 0.1          # p. of local search
        self.prob_grasp = np.zeros(self.moop.nobj)
        self.prob_ls1 = 0.4         # upper bound for local search 1
        self.prob_ls2 = 0.7         # upper bound for local search 2
        self.update_prob_grasp()

    def rungen(self):
        self.update_gen()
        # Create the parent pool:
        self.makep()
        if self.g == 1:
            self.q = ga.binsel(self.p, self.pop, 'elitism')
        else:
            self.q = ga.binsel(self.p, self.pop, 'cco')
        self.adaptive_crossover()
        # Mutation random or selected
        self.q, self.newgenes = \
            ga.mutat_mv(self.q, self.newgenes, self.pop, self.prob_mutat)
        self.fill_q()
        # Adaptive Local search
        self.local_search()
        self.cleanarchive()
        print(self.funkeval, 'function evaluations have been performed.\n')

    def makep(self):
        print('Selecting parent population...')
        r = self.p + self.q
        r, fronts = fnds(r)
        print('There are currently {0} solutions in the Approximate Set.'.
              format(len(fronts[0])))
        # Add nondominated solutions to the External Archive
        for m in range(len(fronts[0])):
            self.add_to_archive(fronts[0][m])
        if self.g == 1:
            self.p = r
        else:
            self.fill(fronts)

    def add_to_archive(self, solution):
        # This function determines if newsol should be added to the external
        # archive and if other solutions need to be removed
        solid = solution.getid()
        if solid in self.archive:
            if solution is self.archive.get(solid):
                return
        u = solution.getfits()
        addable = self.check_archive(u)
        if addable:
            if solid in self.archive:
                solution.updateid(self.idnum)
                self.updateid(1)
                self.archive[solution.getid()] = solution
            else:
                self.archive[solid] = solution

    def check_archive(self, u):
        # This module checks if fitness vector u can be added to the archive
        keys = [k for k, sol in self.archive.items()]
        for k in keys:
            v = self.archive.get(k).getfits()
            # If u dominates v, delete k from archive
            if mop.dom2(u, v):
                del self.archive[k]
            else:
                # If v dominates u, don't add it to the archive
                if mop.dom2(v, u):
                    return False
        return True

    def cleanarchive(self):
        # This function removes unwanted solutions from the archive
        # and calls the local search function.
        aslimit = 100           # Limit to size of approximation set
        # Every 50 generations check the archive
        if self.g % 50 == 0 or (self.end - self.funkeval) < 500:
            # Truncate approximate set
            if len(self.archive) > aslimit:
                self.update_cdvalues()
                self.fittruncation(aslimit)

    def update_cdvalues(self):
        # This function updates the archive solutions' cd values
            approxset = [sol for k, sol in self.archive.items()]
            approxset = cda(approxset, 3, rmovedupl=False)
            for m in range(len(approxset)):
                solid = approxset[m].getid()
                self.archive[solid] = approxset[m]

    def fittruncation(self, limit):
        # This function removes solutions from the approximate set that have
        # the same fitness values.
        print('Removing superfluous solutions from the archive.')
        clusters = self.cluster_solutions_bybin()
        # Remove one solution from each cluster before repeating
        for k in range(len(clusters[0]) - 1):
            if len(self.archive) <= limit:
                return
            clusters = self.remove_one_sol_from_clusters(clusters, limit)
        print('There are now {0} solutions in the Approximate Set.'.
              format(len(self.archive)))

    def remove_one_sol_from_clusters(self, clusters, limit):
        # Set cd value to be removed from each cluster
        cdremove = self.archive.get(clusters[0][0]).getcd()
        for c in range(len(clusters)):
            # If archive is below limit, stop removing solutions
            if len(self.archive) <= limit:
                return clusters
            if len(clusters[c]) > 1:
                clusters[c] = self.remove_one_solution(clusters[c], cdremove)
        return clusters

    def remove_one_solution(self, cluster, cdremove):
        # This function removes one solution in cluster from the archive
        for key in cluster:
            solution = self.archive.get(key)
            if solution.getcd() <= cdremove:
                if solution not in self.q:
                    del self.archive[key]
                    cluster.remove(key)
                    return cluster
            else:
                return cluster
        return cluster

    def cluster_solutions_bybin(self):
        # This function sorts a front into clusters of solutions
        # Each cluster has the same number of bins in the solution
        clusters = []
        for binsize in range(int(self.ideal_values[0]), int(self.obj_weights[0] + 1)):
            keys = [k for k, sol in self.archive.items()
                    if sol.getbins() == binsize]
            # Not a cluster if a solution is by itself
            if len(keys) > 1:
                keys = self.sort_cluster_bycd(keys)
                clusters.append(keys)
        orderedbylen = sorted(clusters, key=len, reverse=True)
        return orderedbylen

    def sort_cluster_bycd(self, cluster):
        # This function sorts an individual cluster by crowding distance values
        cd_values = []
        for key in cluster:
            cd_values.append(self.archive.get(key).getcd())
        cluster = [key for (cd, key) in sorted(zip(cd_values, cluster))]
        return cluster

    def fill(self, fronts):
        # This module fills the parent population based on the crowded
        # comparison operator.
        self.p = []
        k = 0
        fronts[k] = cda(fronts[k], 3)
        while (len(self.p) + len(fronts[k])) < self.pop:
            self.p = self.p + fronts[k]
            k += 1
            if not fronts[k]:
                fronts[k] = self.find_more_sols()
            fronts[k] = cda(fronts[k], 3)
        fronts[k].sort(key=attrgetter('cd'), reverse=True)
        fillsize = self.pop - len(self.p)
        for l in range(fillsize):
            self.p.append(fronts[k][l])

    def find_more_sols(self):
        # This module locates more solutions if too many were removed
        # during truncation procedures
        neighbors = []
        nls = self.pop - len(self.p)        # Number of neighbors to find
        keys = [k for k, m in self.archive.items()]
        # Pick random solution from external archive:
        m = random.choice(keys)
        rannum = random.random()
        # Find nondominated solution neighbors
        if rannum < self.prob_ls1:
            self.idnum, newset = self.call_ls1(self.archive.get(m), nls)
        elif rannum < self.prob_ls2:
            self.idnum, newset = self.bpp.ls2(self.idnum, nls, self.archive.get(m))
        else:
            self.idnum, newset = self.bpp.ls3(self.idnum, nls, self.archive.get(m))
        self.funkeval += nls
        neighbors.extend(newset)
        return neighbors

    def initialq(self, seedfile):
        # This module creates an initial generation randomly
        random.seed(self.getseedvalue(seedfile))
        for m in range(self.pop):
            p, newsol = self.bpp.generate_newsol(self.idnum, self.prob_ls1,
                                                 self.prob_ls2)
            self.updatefe(p - self.idnum)
            self.updateid(p - self.idnum)
            self.q.append(newsol)

    def fill_q(self):
        # This function transforms self.newgenes into new solutions to add to q
        for m in range(len(self.newgenes)):
            args = self.newgenes[m]
            p, newsol = self.bpp.generate_newsol(self.idnum, self.prob_ls1,
                                                 self.prob_ls2, *args)
            self.updatefe(p - self.idnum)
            self.updateid(p - self.idnum)
            self.q.append(newsol)

    def adaptive_crossover(self):
        # Decide adaptively if random crossover or selected crossover
        n_rand = int(self.pop * self.frac_random)
        if n_rand % 2 != 0:
            n_rand -= 1
        set_random = self.q[:n_rand]
        set_clustr = self.q[n_rand:]
        # Send fractions to their respective operators
        set_random, newgenes_r = ga.xover_mv(set_random, n_rand, self.prob_xover)
        set_clustr, newgenes_c = self.clustered_crossover(set_clustr)
        self.q = set_random + set_clustr
        self.newgenes = newgenes_r + newgenes_c

    def clustered_crossover(self, setc):
        # This module clusters set_c into m*2 groups and then performs crossover
        # operations within each group
        self.update_ideal_values()
        newgenes = []
        # Split set_c into m*2 groups
        n_clustr = self.moop.nobj * 2
        cluster_sets = self.cluster_solutions(n_clustr, setc)
        for c in range(n_clustr):
            # If some region of the objective space is undiscovered,
            # increase mutation rate
            if len(cluster_sets[c]) < 2:
                # Ensure probability does not goes past 1.0
                increase = min(0.01, 1.0 - self.prob_mutat)
                self.prob_mutat += round(increase, 2)
            else:
                c_pop = 2 * (len(cluster_sets[c]) // 2)
                cluster_sets[c], ngc = \
                    ga.xover_mv(cluster_sets[c], c_pop, self.prob_xover)
                newgenes.extend(ngc)
        setc = [sol for sublist in cluster_sets for sol in sublist]
        return setc, newgenes

    def cluster_solutions(self, n_clustr, setc):
        # This module clusters set_c into m*2 groups
        lambdas = self.random_tchebycheff_weights(n_clustr)
        cluster_sets = [[] for _ in range(n_clustr)]
        # Calculate the single-objective value for each weight vector and assign
        # to minimum value cluster
        for m in range(len(setc)):
            cluster_vector = self.calc_sofit_byweights(setc[m], lambdas)
            c = np.argmin(cluster_vector)
            cluster_sets[c].append(setc[m])
        return cluster_sets

    def calc_sofit_byweights(self, solution, lambdas):
        # This function calculates the Tchebycheff single-objective fitness values
        # for a solution for a given set of weights
        cluster_vector = np.zeros(len(lambdas), dtype=np.float64)
        u = solution.getfits()
        for c in range(len(lambdas)):
            diff = np.subtract(u, self.ideal_values)
            weights = np.divide(lambdas[c], self.obj_weights)
            fit_vector = np.multiply(weights, diff)
            cluster_vector[c] = np.amax(fit_vector)
        return cluster_vector

    def random_tchebycheff_weights(self, n_clustr):
        # This module returns n_clustr weight vectors.
        lambdas = []
        for k in range(n_clustr):
            high = 1.0
            lambdak = np.zeros(self.moop.nobj, dtype=np.float64)
            for w in range(self.moop.nobj - 1):
                lambdak[w] = random.uniform(0, high)
                high -= lambdak[w]
            lambdak[-1] = 1 - np.sum(lambdak)
            lambdas.append(lambdak)
        return lambdas

    def update_ideal_values(self):
        # Update ideal value for number of boxes open
        openbins = min(self.archive.values(), key=attrgetter('fit0')).fit0
        self.ideal_values[0] = openbins
        # Update ideal value for avg. initial box heat
        box_heat = min(self.archive.values(), key=attrgetter('fit1')).fit1
        self.ideal_values[1] = box_heat
        # Update ideal value for max tready
        tready = min(self.archive.values(), key=attrgetter('fit2')).fit2
        self.ideal_values[2] = tready
        self.update_objective_weights()

    def update_objective_weights(self):
        # This function updates the objective function weights for the Tchebycheff
        # clustering.
        max_bins = max(self.archive.values(), key=attrgetter('fit0')).fit0
        self.obj_weights[0] = max_bins
        # Update ideal value for avg. initial box heat
        max_heat = max(self.archive.values(), key=attrgetter('fit1')).fit1
        self.obj_weights[1] = max_heat
        # Update ideal value for max tready
        max_tready = max(self.archive.values(), key=attrgetter('fit2')).fit2
        self.obj_weights[2] = max_tready

    def local_search(self):
        # This module governs the local search phase of the algorithm
        print('Local search: probability of {0}'.format(self.prob_ls))
        numls = 10                  # Number of neighbors to find
        self.localsearch_bygrasp(numls)
        if self.g % 2 == 0:
            self.pareto_local_search(numls)
        if self.g > 2:
            self.update_local_search_probabilities()

    def localsearch_bygrasp(self, numls):
        # This function adaptively selects a local search mechanism for
        # members of q
        for m in range(self.pop):
            rannum = random.random()
            if rannum < self.prob_ls:
                ranls = random.random()
                if ranls < self.prob_ls1:
                    p, neighbors = self.call_ls1(self.q[m], numls)
                elif ranls < self.prob_ls2:
                    p, neighbors = self.call_ls2(self.q[m], numls)
                else:
                    p, neighbors = self.call_ls3(self.q[m], numls)
                self.updatefe(p - self.idnum)
                self.updateid(p - self.idnum)
                self.q.extend(neighbors)

    def call_ls1(self, solution, numls):
        # Only search for fewer bins if have room to squeeze out
        if solution.getopenbins() > self.bpp.lb:
            p, neighbors = self.bpp.ls1(self.idnum, numls, solution)
        else:
            p, neighbors = self.bpp.bin_mutation(self.idnum, numls, solution)
        return p, neighbors

    def call_ls2(self, solution, numls):
        # This module calls the ls2 operator numls / 2 times, then searches each
        # returned neighbor with ls3
        p, neighbors = self.bpp.ls2(self.idnum, numls / 2, solution)
        for m2 in range(len(neighbors)):
            p, more = self.bpp.ls3(self.idnum, 2, neighbors[m2])
            neighbors.extend(more)
        return p, neighbors

    def call_ls3(self, solution, numls):
        # This module calls the ls3 operator numls / 2 times, then searches each
        # returned neighbor with ls2
        p, neighbors = self.bpp.ls3(self.idnum, numls / 2, solution)
        for m2 in range(len(neighbors)):
            p, more = self.bpp.ls2(self.idnum, 2, neighbors[m2])
            neighbors.extend(more)
        return p, neighbors

    def pareto_local_search(self, numls):
        # This function randomly selects m solutions belonging to the edges of
        # the Pareto front and performs bin-specific mutations to find better
        # solutions
        print(' - performing Pareto local search')
        # Always search the solutions near the ideal values
        for theta in range(self.moop.nobj):
            # Identify ideal for objective theta
            zideal = self.ideal_values[theta]
            # Local search one of the lower Pareto members
            keys_low = [k for k, sol in self.archive.items()
                        if sol.fits[theta] - zideal <= self.alpha * zideal]
            self.pick_and_pls_solution(numls, keys_low)
        # Search along the front using clusters
        clusters = self.cluster_solutions_bybin()
        n_clustr = len(clusters)
        for c in range(n_clustr):
            if clusters[c]:
                key = random.choice(clusters[c])
                self.pls_sorting(numls, self.archive.get(key))
        if self.g % 4 == 0:
            self.pls_approximate_set(numls)

    def pls_approximate_set(self, numls):
        # This function randomly performs local search on the members of the
        # approximate set
        for k, sol in self.archive.items():
            rannum = random.random()
            if rannum < self.prob_ls:
                self.pls_sorting(numls, sol)

    def pick_and_pls_solution(self, numls, keys):
        # This module selects a random solution from keys and sends it to
        # pls_sorting
        search_key = random.choice(keys)
        self.pls_sorting(numls, self.archive.get(search_key))

    def pls_sorting(self, numls, solution):
        # This module sends a nondominated solution to the right ls routine and
        # adds its neighbors to population q
        theta = random.choice([0, 1, 2])
        if theta == 0:
            p, neighbors = self.call_ls1(solution, numls)
        elif theta == 1:
            ran2 = random.random()
            if ran2 < 0.5:
                p, neighbors = self.bpp.ls2(self.idnum, numls, solution)
            else:
                p, neighbors = self.call_ls2(solution, numls)
        else:
            ran3 = random.random()
            if ran3 < 0.5:
                p, neighbors = self.bpp.ls3(self.idnum, numls, solution)
            else:
                p, neighbors = self.call_ls3(solution, numls)
        self.updatefe(p - self.idnum)
        self.updateid(p - self.idnum)
        self.q.extend(neighbors)

    def update_local_search_probabilities(self):
        # This function updates the local search probabilities based on how
        # many archive solutions are included in the Pareto local search.
        pls_size = np.zeros(self.moop.nobj, dtype=int)
        for theta in range(self.moop.nobj):
            # Make quick list of options
            zideal = self.ideal_values[theta]
            keys = [k for k, sol in self.archive.items()
                    if sol.fits[theta] - zideal <= self.alpha * zideal]
            pls_size[theta] = len(keys)
        max_size = np.amax(pls_size)
        theta_max = np.argmax(pls_size)
        min_size = np.amin(pls_size)
        theta_min = np.argmin(pls_size)
        p_distribute = self.prob_grasp[theta_max] * (1 - 1/(max_size / min_size))
        theta_share = [theta for theta in range(self.moop.nobj)
                       if theta not in [theta_min, theta_max]][0]
        portion_share = min_size / (pls_size[theta_share] + min_size)
        self.update_prob_ls(pls_size)
        # Update the probability for ls1
        if 0 == theta_max:
            self.prob_ls1 -= round(p_distribute, 2)
        elif 0 == theta_share:
            self.prob_ls1 += round(portion_share * p_distribute, 2)
        else:
            self.prob_ls1 += round(p_distribute * (1 - portion_share), 2)
        # Update the probability for ls2
        if 1 == theta_max:
            self.prob_ls2 = self.prob_ls1 + self.prob_grasp[1] - p_distribute
        elif 1 == theta_share:
            self.prob_ls2 = self.prob_ls1 + self.prob_grasp[1] + \
                            round(portion_share * p_distribute, 2)
        else:
            self.prob_ls2 = self.prob_ls1 + self.prob_grasp[1] + \
                            round(p_distribute * (1 - portion_share), 2)
        self.update_prob_grasp()

    def update_prob_ls(self, pls_size):
        # Update the probability for local search overall
        if np.any(np.equal(pls_size, 5)):
            increase = min(0.02, 1.0 - self.prob_ls)
            self.prob_ls += increase
        elif np.any(np.equal(pls_size, 20)):
            decrease = min(0.02, self.prob_ls)
            self.prob_ls -= decrease

    def update_prob_grasp(self):
        # Update the probabilities for the grasp local search operators
        self.prob_grasp[0] = self.prob_ls1
        self.prob_grasp[1] = self.prob_ls2 - self.prob_ls1
        self.prob_grasp[2] = 1.0 - self.prob_ls2

    def finalapproxset(self):
        # This function finds the final approximate set.
        numsol = len(self.archive)
        ndset = []
        keys = [k for k, v in self.archive.items()]
        for p in range(numsol):
            np = 0
            u = self.archive.get(keys[p])
            for q in range(numsol):
                v = self.archive.get(keys[q])
                if u != v:
                    # If v dominates u, increase domination count
                    if mop.dom1(v.getfits(), u.getfits()):
                        np += 1
            if np == 0:
                ndset.append(u)
        return ndset

    def update_gen(self):
        self.g += 1
        # Print out generation info:
        if self.g % 50 == 0:
            print('                         ', datetime.now())
        print(' ')
        print('gen = ', self.g)

    def endgen(self):
        # Complete if number of function evaluations > end
        if self.g < 3:
            return False
        return self.funkeval > self.end

    def getseedvalue(self, seedfile):
        # This function retrieves the seed number from seedfile.
        infile = open(seedfile, 'r')
        seed = int(infile.readline())
        infile.close()
        return seed

    def updatefe(self, n_eval):
        # This function keeps track of the number of function evaluations.
        self.funkeval += n_eval

    def updateid(self, n_add):
        self.idnum += n_add

    def getid(self):
        return self.idnum

    def gett(self):
        return self.g

    def getp(self):
        return self.p

    def getq(self):
        return self.q

    def get_probabilities(self):
        probs = {'prob_crossover': self.prob_xover,
                 'prob_mutation': self.prob_mutat,
                 'prob_localsearch': self.prob_ls,
                 'prob_ls1': self.prob_ls1,
                 'prob_ls2': self.prob_ls2,
                 'prob_ls3': 1.0 - self.prob_ls2}
        return probs

    def getarch(self):
        return self.archive


def fnds(setp):
    # This module performs the fast-non-dominated-sort described in Deb(2002).
    # To run this module, enable the following line:
    # from mop import dom
    numsol = len(setp)
    fronts = []
    sp = []
    fhold = []
    nps = []
    for p in range(numsol):
        shold = []
        nump = 0
        for q in range(numsol):
            if setp[p] != setp[q]:
                if mop.dom1(setp[p].getfits(), setp[q].getfits()):
                    shold.append(setp[q])
                if mop.dom1(setp[q].getfits(), setp[p].getfits()):
                    nump += 1
        sp.append(shold)
        nps.append(nump)
        if nump == 0:
            fhold.append(setp[p])
            setp[p].updaterank(1)
    fronts.append(fhold)            # Pareto set
    i = 0
    while fronts[i] != []:
        q = []
        for j in range(numsol):
            if setp[j] in fronts[i]:
                for k in range(numsol):
                    if setp[k] in sp[j]:
                        nps[k] -= 1
                        if nps[k] == 0:
                            setp[k].updaterank(i+2)
                            q.append(setp[k])
        fronts.append(q)
        i += 1
    return setp, fronts


def cco(i, p, j, q):
    # This module guides the selection process using the crowded distance
    # assignment.
    prank = p.getrank()
    qrank = q.getrank()
    if prank < qrank:
        best = i
    elif prank > qrank:
        best = j
    else:
        pid = p.getcd()
        qid = q.getcd()
        if pid > qid:
            best = i
        else:
            best = j
    return best


def cda(front, nobj, rmovedupl=True):
    # This module performs the calculation of the crowding distance
    # assignment.
    if rmovedupl:
        front = rmdupl(front)
    l = len(front)
    fdist = []
    indexes = []
    for i in range(l):
        fdist.append(0)
        indexes.append([i, front[i].getbins(), front[i].getavgheat(), front[i].getmaxreadyt()])
    for o in range(nobj):
        indexes.sort(key=lambda pair: pair[o+1])
        fdist[indexes[0][0]] = 1000000  # represent infinity
        fdist[indexes[l-1][0]] = 1000000
        for i in range(1, l-1):
            fdist[indexes[i][0]] += (indexes[i+1][o+1]-indexes[i-1][o+1]) / \
                                    (indexes[l-1][o+1]-indexes[0][o+1])
    for p in range(l):
        idist = fdist[p]
        front[p].updatecd(idist)
    return front


def rmdupl(front):
    # This module takes a front produced by sorting a generation and
    # removes any duplicate individuals.
    r = 0
    for p in range(len(front)):  # remove duplicates
        if front.count(front[r]) > 1:
            del front[r]
            r -= 1
        r += 1
    return front


def savefitvals(ndset, folder):
    # This function saves the fitness values of the Approximation Set to
    # a csv file.
    with open(folder + 'ApproximationSet.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, dialect='excel', delimiter=',')
        csvwriter.writerow(['Solution ID', 'f[1] (B)', 'f[2] (W)', 'f[3] (s)'])
        for m in ndset:
            csvwriter.writerow([m.getid(), m.getbins(), m.getavgheat(),
                                m.getmaxreadyt()])


def savexys(ndset, folder):
    # This function saves the x, y, and tfill matrices into a .h5 file for later
    names = []
    xs = []
    ys = []
    tfills = []
    for m in range(len(ndset)):
        index, x, y = ndset[m].getid(), ndset[m].getx(), ndset[m].gety()
        tfill = ndset[m].gettfill()
        names.append(str(index))
        xs.append(x)
        ys.append(y)
        tfills.append(tfill)
    h5f = h5py.File(folder + 'xymatrices.h5', 'w')
    xgrp = h5f.create_group('xmatrices')
    ygrp = h5f.create_group('yarrays')
    tgrp = h5f.create_group('tfills')
    for m in range(len(names)):
        xgrp.create_dataset(names[m], data=xs[m])
        ygrp.create_dataset(names[m], data=ys[m])
        tgrp.create_dataset(names[m], data=tfills[m])
    h5f.close()


def see(ndset, folder):
    # see prints out a file to show how x and y for a gene
    dirname = folder + 'variables/'
    if not path.exists(dirname):
        mkdir(dirname)
    for m in ndset:
        x, y, tfill = m.getx(), m.gety(), m.gettfill()
        solid = str(m.getid())
        np.savetxt(dirname + solid + '_x.txt', x, fmt='%i', header='Item Location Matrix x:')
        np.savetxt(dirname + solid + '_y.txt', y, fmt='%i', header='Bins Used Matrix y:')
        np.savetxt(dirname + solid + '_tfill.txt', tfill, fmt='%G',
                   header='Time of Bins Used Matrix tfill:')


def checkformismatch(solution, out=sys.stdout):
    # This function identifies if the given solution does not have an x-matrix
    # and a variable length representation that match.
    x = solution.getx()
    vlrep = solution.getvlrep()
    for i in range(len(vlrep)):
        for j in vlrep[i]:
            if x[i, j] != 1:
                out.write('Error: Solution {0:d} is not coordinated on item {1:d}.'
                          .format(solution.getid(), j))


if __name__ == '__main__':
    n = eval(input('Please enter the number of items to be sorted: \n'))
    folder = input('Please enter the name of the folder where your input file is: \n')
    datafile = input('Please enter the name of the input file: \n')

    algorithm(n, folder, datafile)