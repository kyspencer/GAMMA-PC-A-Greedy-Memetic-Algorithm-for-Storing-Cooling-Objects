# outformat.py
#   Program to create pretty looking output.
#   Author: Kristina Spencer
#   Date: March 31, 2016

from __future__ import print_function
from datetime import datetime


def main():
    print(' This module automates formatting of output for this project.')


def startout(filename, n, end, data, method):
    import datetime
    startt = datetime.datetime.now()
    outfile = open(filename, 'w')
    print('                         ', startt, end='\n', file=outfile)
    print('*******************************************************************************', file=outfile)
    print('', file=outfile)
    print('     This output file records the optimization of a bin packing problem\n', file=outfile)
    print('     with ', n, ' items to pack. The optimization is terminated after\n', file=outfile)
    print('     ', end, ' fitness function evaluations.\n', file=outfile)
    print('', file=outfile)
    print('*******************************************************************************', file=outfile)
    print('', file=outfile)
    print('     Method: ', method, end='\n', file=outfile)
    print('     data: ', data, file=outfile)
    outfile.close()


def gen_probabilities(filename, t, **kwargs):
    outfile = open(filename, 'a')
    print('Generation ', t, ':', file=outfile)
    for k, v in kwargs.items():
        print('{0:16}  {1:4.2f}'.format(k, v), file=outfile)
    print('', file=outfile)
    outfile.close()


def genout(filename, t, pop, genomes, fronts, onlyapprox=False):
    outfile = open(filename, 'a')
    print('Generation ', t, ': \n', file=outfile)
    print('Gen. Member  Solution ID     f[1] (B)    f[2] (W)        f[3] (s)', file=outfile)
    print('------------------------------------------------------------------', file=outfile)
    for m in range(pop):
        print("{0:11d}  {1:11d}     {2:8.0f}    {3:8.2f}    {4:12.1f}".format(m, genomes[m].getid(),
                                                                              genomes[m].getbins(),
                                                                              genomes[m].getavgheat(),
                                                                              genomes[m].getmaxreadyt()),
              file=outfile)
    print('', file=outfile)
    if onlyapprox:
        # Will probably be a dictionary
        approxset = [v for k, v in fronts.items()]
    else:
        approxset = fronts[0]
    print('There are currently ', len(approxset), 'solutions in the Pareto Set:', file=outfile)
    for m in range(len(approxset)):
        print('  - Solution Number ', approxset[m].getid(), file=outfile)
    if not onlyapprox:
        i = 1
        while fronts[i] != []:
            flen = len(fronts[i])
            print('Front', i, ' has', flen, ' members in it.', file=outfile)
            i += 1
    print('', file=outfile)
    print('', file=outfile)
    outfile.close()


def endout(filename):
    outfile = open(filename, 'a')
    endt = datetime.now()
    print('*******************************************************************************', file=outfile)
    print('                         ', endt, file=outfile)
    outfile.close()


def printvlrep(filename, vlrep):
    # This function prints out the variable length representation of the
    # chromosome.
    output = open(filename, 'a')
    print('Variable Length Representation of Loading:', file=output)
    for i in range(len(vlrep)):
        print(vlrep[i], file=output)
    output.close()


if __name__ == '__main__':
    main()
