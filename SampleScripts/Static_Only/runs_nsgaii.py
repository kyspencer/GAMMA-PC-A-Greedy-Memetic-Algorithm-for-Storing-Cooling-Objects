# runs.py
#    This Python script automates running the bin packing experiments for my
#    research.
#    Author: Kristina Spencer
#    Date: April 9, 2016

import gammapc
from TextMeWhenDone import TextMeWhenDone

def runalgorithms():
    n = 500
    for i in range(18, 20):
        folder = 'GAMMA-PC/SBSBPP500/Experiment{0}/'.format(i + 1)
        datafile = 'SBSBPP500_run{0}.txt'.format(i + 1)
        gammapc.algorithm(n, folder, datafile)


if __name__ == '__main__':
    TextMeWhenDone('AT&T', 9794924082, 'kristina.yancey@gmail.com',
                   'fiib lvls sbtm utws', runalgorithms)
