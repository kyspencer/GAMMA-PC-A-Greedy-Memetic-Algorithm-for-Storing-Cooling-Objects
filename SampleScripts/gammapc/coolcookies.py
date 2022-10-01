# coolcookies.py
#   This file contains a class of cookie objects and modules
#   to make them from an input file.
#   Author: Kristina Spencer

from __future__ import print_function
from math import exp
from os import path
from scipy.stats import norm


class Cookie:
    volume = 1.2e-5  # m^3
    surfarea = 4.9e-3  # m^2
    def __init__(self, density, specificheat, batch):
        self.rho = density
        self.cp = specificheat
        self.rhocp = self.rho * self.cp
        self.batch = batch
        self.inittemp = 20.0 + 273  # room temp. (K)
        self.calcinitialstate()

    def calcinitialstate(self):
        oventemp = 175.0 + 273    # Kelvin
        t_oven = 600.0            # Seconds (10 min)
        h_oven = 62.0             # W/(m2-K)
        tau_oven = (self.rhocp * Cookie.volume * 1000)/(h_oven * Cookie.surfarea)
        self.inittemp = (self.inittemp - oventemp) * exp(-t_oven / tau_oven) \
                        + oventemp

    def gettemp(self, t, temp_amb, h):
        # This module calculates the temperature of the cookie
        # at time t during cooling (time from removal from oven).
        #   Input: t - time (s), temp_amb - ambient temp. (K)
        #          h - ambient heat transfer coefficient (W/m2)
        #   Output: Temp (K)
        tcool = t - self.batch * 600       # each batch 10 min cooktime
        timeconstant = (self.rhocp * Cookie.volume * 1000)/(h * Cookie.surfarea)
        temp = (self.inittemp - temp_amb) * exp(-tcool / timeconstant) + temp_amb
        return temp

    def getdensity(self):
        return self.rho

    def getcp(self):
        return self.cp

    def getbatch(self):
        return self.batch


def makeobjects(n, batchsize, filename):
    # This module is designed to turn a text file into a library of
    # cookie objects.
    cookiecollection = {}
    items = importfile(filename)
    # Organize items based on the third column
    batchorder = sorted(items, key=lambda x: x[-1])
    if n != len(batchorder):
        print('Your input file has an error: missing item data.')
        print('  n = {0} vs. batch order size of {1}'.format(n, len(batchorder)))
    # Make cookie objects
    for j in range(n):
        batch = j // batchsize + 1
        cookiecollection[j] = makecookie(batchorder[j], batch)
    makedatafile(n, filename, cookiecollection)
    return cookiecollection


def importfile(filename):
    # Retrieve data from filename, turn strings into integers
    infile = open(filename, "r")
    items = []
    for line in infile:
        if not line.strip():
            continue
        else:
            z1, z2, order = line.split("\t")
            items.append((int(z1), int(z2), int(order)))
    infile.close()
    return items


def makecookie(objectstats, batch):
    # This function transforms p1 into a density value,
    # p2 into a heat capacity, and creates a cookie object.
    # Average values of interest:
    rho_mean = 1252.3
    rho_stdev = 17.6
    cp_mean = 2.94
    cp_stdev = 0.17
    # Transform input into probabilities
    p1 = objectstats[0] / 100.0
    p2 = objectstats[1] / 100.0
    # Transform p-values into z-values
    if p1 == 1.0:
        z1 = 3
    else:
        z1 = norm.ppf(p1)
    if p2 == 1.0:
        z2 = 3
    else:
        z2 = norm.ppf(p2)
    # Transform into thermophysical values
    density = rho_mean + z1 * rho_stdev
    heatcap = cp_mean + z2 * cp_stdev
    # Make cookie
    cookie = Cookie(density, heatcap, batch)
    return cookie


def makedatafile(n, filename, cookies):
    # This function creates a text file containing the cookie data.
    pathname = path.dirname(filename)
    datafile = pathname + '/CookieData.txt'
    outfile = open(datafile, 'w')
    for j in range(n):
        density = cookies.get(j).getdensity()
        heatcap = cookies.get(j).getcp()
        batch = cookies.get(j).getbatch()
        print('Cookie added to batch{0:2d}: {1:7.2f} kg/m3, {2:4.2f} kJ/kgK'
              .format(batch, density, heatcap), file=outfile)
    outfile.close()

if __name__ == '__main__':
    #filename = input('Please enter the name of your input file: ')
    #n = eval(input('How many items should be present?'))
    n = 24
    batchsize = 6
    filename = 'tests/stubs/Cookies24.txt'
    cookies = makeobjects(n, batchsize, filename)
