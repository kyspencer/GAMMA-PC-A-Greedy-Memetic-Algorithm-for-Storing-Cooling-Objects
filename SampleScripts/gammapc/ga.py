# ga.py
#   This python script contains modules used in genetic algorithms.
#   Author: Kristina Spencer
#   Date: April 4, 2016


import numpy as np
import random
    

def binsel(rt, pop, mode):
    # binsel is a binary tournament selection module. It selects 2
    # members of population rt at random and chooses the best individual.
    print('GA operation: binary selection...')
    numsol = len(rt)
    q = []
    for m in range(pop):
        i = random.randint(0, numsol-1)
        j = random.randint(0, numsol-1)
        if mode == 'elitism':
            k = elitism(i, rt[i], j, rt[j])
            q.append(rt[k])
        elif mode == 'cco':  # crowding comparison operator
            k = cco(i, rt[i], j, rt[j])
            q.append(rt[k])
        else:
            print('An incorrect mode has been chosen for binary tournament selection.')
    return q
    
    
def elitism(i, p, j, q):
    # This module determines which individual has a lower
    # rank and sends that one to the next generation.
    prank = p.getrank()
    qrank = q.getrank()
    if prank < qrank:
        best = i
    elif prank > qrank:
        best = j
    else:
        opt = [i, j]
        best = random.choice(opt)
    return best


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


def xover(q, n, prob):
    # This module performs the crossover function on the next
    # generation. It is specific to bin packing.
    #  - q is the next generation
    #  - n is the number of members in a generation
    #  - prob is the crossover probability
    nover = int(n / 2)  # number of crossover matches to make
    newgenes = []
    for i in range(n):
        newgenes.append(q[i].getgenes().copy())

    for i in range(nover):
        a = random.randint(0, n-1)
        b = random.randint(0, n-1)
        p = random.random()
        if p <= prob:
            newgenes[a], newgenes[b] = xover_realgene(newgenes[a], newgenes[b])
    return newgenes


def xover_mv(q, pop, prob):
    # This module performs the crossover function for the mixed variable
    # chromosome + tdiff (continuous)
    #   - q is the selected breeding pool
    #   - pop is the number of member in a generation
    #   - prob is the crossover probability
    print('GA operation: crossover...')
    nover = int(pop / 2)        # number of crossover matches to make
    changed = []
    newgenes = []               # Format of list: tuples (genes, tfill)
    # Generate list of random order for crossover
    matchlist = random.sample(range(pop), pop)
    for m in range(nover):
        a = matchlist[m]
        b = matchlist[nover + m]
        p = random.random()
        if p <= prob:
            changed.append(q[a])
            changed.append(q[b])
            chroma, chromb = xover_realgene(q[a].getgenes().copy(),
                                            q[b].getgenes().copy())
            tfilla, tfillb = xover_tfill(q[a], q[b])
            newgenes.append((chroma, tfilla))
            newgenes.append((chromb, tfillb))
    # Remove solutions that have been crossed over from q
    for m in range(len(changed)):
        q.remove(changed[m])
    return q, newgenes


def xover_realgene(chroma, chromb):
    # This function performs the crossover of two real-valued chromosomes
    oldca = chroma[:]
    oldcb = chromb[:]
    cut = random.randint(1, len(oldca) - 1)
    cuta1 = oldca[:]
    cuta2 = oldca[cut:]
    cutb1 = oldcb[:]
    cutb2 = oldcb[cut:]
    for j in range(len(oldca)):
        if cutb2.count(oldca[j]) > 0:
            cuta1.remove(oldca[j])
    for j in range(len(oldcb)):
        if cuta2.count(oldcb[j]) > 0:
            cutb1.remove(oldcb[j])
    chroma = cuta1 + cutb2
    chromb = cutb1 + cuta2
    return chroma, chromb


def xover_tfill(sola, solb):
    # This function performs the crossover of two t_fill matrices
    nba = sola.getopenbins()            # Number of open bins in solution a
    tfilla = sola.gettfill().copy()
    nbb = solb.getopenbins()            # Number of open bins in solution b
    tfillb = solb.gettfill().copy()
    cut = random.randint(1, min(nba, nbb) - 1)
    for i in range(cut, min(nba, nbb)):
        tfilla[i], tfillb[i] = tfillb[i], tfilla[i]
    tfilla[:nba] = np.sort(tfilla[:nba], axis=0)
    tfillb[:nbb] = np.sort(tfillb[:nbb], axis=0)
    return tfilla, tfillb


def mutat(n, q, members):
    # This module mutates the new generation.
    #  - n is the number of items to be packed
    #  - q is the new generation
    #  - members is the number of individuals in a gen.
    prob = 0.3  # mutation probability
    for m in range(members):
        p = random.random()
        if p <= prob:
            q[m] = mutat_realgene(q[m])
    return q


def mutat_mv(q, newgenes, pop, prob):
    # This module mutates the mixed variable new generation.
    #   - q is the new generation (not crossed over)
    #   - newgenes are genes that have been crossed over
    #   - pop is the size of a generation population
    #   - prob is the probability of mutation
    print('GA operation: mutation...')
    changed = []
    gen_sample_tdiff = sampletfills(q, newgenes)
    tdiff_variability = calcstdevs(gen_sample_tdiff)
    # Mutate genes
    for m in range(pop):
        p = random.random()
        if p <= prob:
            if m < len(q):
                changed.append(q[m])
                chrom = mutat_realgene(q[m].getgenes().copy())
                tfill = mutat_tfill(q[m].gettfill().copy(), tdiff_variability)
                newgenes.append((chrom, tfill))
            else:
                index = m - len(q)
                chrom = mutat_realgene(newgenes[index][0])
                tfill = mutat_tfill(newgenes[index][1], tdiff_variability)
                newgenes[index] = (chrom, tfill)
    # Remove solutions that have been mutated from q
    for m in range(len(changed)):
        if changed[m] in q:
            q.remove(changed[m])
    return q, newgenes


def mutat_realgene(chromosome):
    """This module mutates the real-encoded chromosome by pair swap.

    Input
    -----
    chromosome : list
        List of cookie indices indicating the order in which they
        will be packed.

    Returns
    -------
    chromosome : list
        Reordered list with one pair swapped.
    """
    n = len(chromosome)
    i = random.randint(0, n - 1)
    j = random.randint(0, n - 1)
    while i == j:
        j = random.randint(0, n - 1)

    # Pair swap
    chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome


def mutat_tfill(tfill, variability):
    # This module mutates the continuous matrix t_fill based on a
    # normal distribution (0, sigma) where sigma is given for each
    # t_fill[i] by variability[i]
    tmin = 660      # Nothing smaller than the length of one batch + 10%
    stdev = 0
    for i in range(len(tfill)):
        if tfill[i] != 0:
            for tlow, thigh, sd in variability:
                if tlow <= tfill[i] < thigh:
                    stdev = sd
                    break
            deltat = random.normalvariate(0.0, stdev)
            # Don't let new tfill values go below tmin
            if tfill[i] + deltat <= tmin:
                tfill[i] = tmin
            else:
                tfill[i] += round(deltat, 1)
    return tfill


def calcstdevs(sample):
    # This module calculates the standard deviations of tfill in a generation
    # Set number of ranges of variation:
    mint = 600                  # Time to bake first batch
    maxt = max(sample)          # max. time value in sample
    groups = int(round((maxt - mint) / 600 + 0.5))
    # Add to variability from low t to high t
    variability = []
    tlow = 600
    thigh = 1200
    for g in range(groups):
        groupg = [t for t in sample if tlow <= t < thigh]
        if len(groupg) == 1:
            variability.append([tlow, thigh, 0.0])
        elif groupg:
            garray = np.array(groupg)
            gstdev = np.std(garray, ddof=1)
            variability.append([tlow, thigh, gstdev])
        tlow = thigh
        thigh += 600
    return variability


def sampletfills(q, newgenes):
    # This module returns only a list of tfill matrices
    sample = []
    for m in range(len(q)):
        tfill = q[m].gettfill()
        for i in range(len(tfill)):
            if tfill[i] != 0:
                sample.append(tfill[i])
    for m in range(len(newgenes)):
        tfill = newgenes[m][1]
        for i in range(len(tfill)):
            if tfill[i] != 0:
                sample.append(tfill[i])
    return sample


if __name__ == '__main__':
    print('This python script contains functions used in genetic algorithms:\n'
          '  - binsel(): binary tournament selection\n'
          '  - elitism(): select best individual\n'
          '  - xover(): single-point crossover\n'
          '  - oldnew(): no replacement check')
