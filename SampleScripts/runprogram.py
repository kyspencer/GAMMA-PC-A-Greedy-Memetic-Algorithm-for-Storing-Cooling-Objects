# runs.py
#    This Python script automates running the bin packing experiments.
#    
#    Date: April 9, 2016

import nsgaii_dynamic as nsgaii
import moma_dynamic as moma
import algorithm
from TextMeWhenDone import TextMeWhenDone


def runalgorithms():
    n = 1000
    basefolder = 'Results_Dynamic/'
    fname = 'Cookies1000.txt'
    nsgaii.nsgaii(n, basefolder + 'NSGA-II/Cookies1000/Experiment01/', fname)
    moma.moma(n, basefolder + 'MOMA/Cookies1000/Experiment01/', fname)
    algorithm.algorithm(n, basefolder + 'GAMMA-PC/Cookies1000/Experiment01/',
                        fname)


if __name__ == '__main__':
    number = eval(input('Please enter your phone number without spaces or hyphens:\n'))
    email = input('Please enter your email address:')
    passwd = input('Please enter your email access password:\n')
    TextMeWhenDone('AT&T',number,email,passwd,runalgorithms)
