# mooproblem.py
#   This file contains functions inherent to the multi-objective problem (MOP).
#   Author: Kristina Spencer

from __future__ import print_function
import numpy as np
import sys
from binpacking_dynamic import coordarrays
from random import choice, random, uniform
from grasp import RCLtime
from operator import attrgetter


class MOCookieProblem:
    # This class maintains the overall multi-objective problem for the
    # dynamic cookie benchmark.
    def __init__(self, n, bigc, bigr, bigf_t, cookies):
        self.nobj = 3           # 3 objectives
        self.h = 8.0            # Heat Transfer Coefficient, air [W/m2C]
        self.tempamb = 298.0    # ambient air temp. [K]
        self.r = 2.0e-6         # Discount rate
        self.boxcap = bigc      # Box capacity, number of cookies
        self.coolrack = bigr    # Cooling rack capacity
        self.fillcap = bigf_t   # Period fill limit
        self.tbatch = 600       # Time to cook one batch: 10 min [s]
        self.n = n              # n cookies to put away
        self.cookies = cookies
        self.nbatches = max(cookies.values(), key=attrgetter('batch')).batch

    def calcfits(self, solution):
        # This module calculates the fitness values for a solution.
        fitvals = np.zeros(self.nobj)
        # Objective Function 1: min. # of bins
        fitvals[0] = np.sum(solution.gety())
        # Objective Function 2: min. avg. initial heat in box
        fitvals[1] = self.avgheat(solution)
        # Objective Function 3: min. max. time to transport
        fitvals[2] = self.maxreadyt(solution)
        return fitvals

    def calcfeasibility(self, solution):
        # This module checks if a solution is within the bounds of the feasible
        # region.
        # Constraint 3: Time Viability
        timeviolations = self.timeconstraint(solution)
        # Constraint 4: Cooling Rack Capacity
        rackviolations = self.rackcapacity(solution.getx(), solution.gettfill())
        # We can fix cooling rack violations:
        if rackviolations:
            self.fixcoolingrack(rackviolations, solution)
        # Constraint 5: Period cask fill limit
        fill_violations = self.period_fill_limit(solution)
        if fill_violations:
            self.fix_tfill(fill_violations, solution)
        # Constraint 1: No replacement
        self.noreplacement(solution.getid(), solution.getx())
        # Constraint 2: Box Capacity
        self.boxcapacity(solution.getid(), solution.getx())
        # x and y in agreement?
        xycheck(solution.getid(), solution.getx(), solution.gety())
        coordarrays(solution)
        return solution

    def avgheat(self, solution):
        # This function calculates the average initial heat of the boxes
        x = solution.getx()
        y = solution.gety()
        tfill = solution.gettfill()
        q0bins = np.zeros(self.n)
        top = 0
        for i in range(self.n):
            if y[i] == 1:
                q0bins[i] = self.getinitialboxheat(i, x, tfill[i])
                top += q0bins[i] / ((1 + self.r)**tfill[i])
        solution.setq0bins(q0bins)
        bottom = np.sum(y)
        avgheat = top / bottom
        return avgheat

    def getinitialboxheat(self, i, x, tfilli):
        # This function calculates the initial heat in box i (W)
        boxheat = 0
        sa = self.cookies.get(0).surfarea
        for j in range(self.n):
            if x[i, j] == 1:
                # Get temperature of cookie at time tfill[i]
                tempcookie = self.cookies.get(j). \
                    gettemp(tfilli, self.tempamb, self.h)
                # Calculate convective heat from cookie
                heatfromcookie = self.h * sa * (tempcookie - self.tempamb)
                boxheat += heatfromcookie
        return boxheat

    def maxreadyt(self, solution):
        # This function calculates the maximum time until all boxes are
        # ready to be moved to the storefront.
        vlrep = solution.getvlrep()
        x = solution.getx()
        tfill = solution.gettfill()
        tavail = np.zeros(self.n)
        for i in range(self.n):
            if tfill[i] != 0.0:
                tavail[i] = self.tsearch(vlrep[i], tfill[i])
        maxtime = np.max(tavail)
        solution.settavailable(tavail)
        return maxtime

    def tsearch(self, vlrepi, tfilli):
        # This module uses the modified regula-falsi method to find the time
        # at which box i is ready to be moved to the storefront.
        # Set up bounds
        t0 = tfilli
        t1 = self.nbatches*600*2
        t0, t1 = self.prepareinput(vlrepi, t0, t1)
        # If t1 is a root, return value:
        if t0 is None:
            return t1
        else:
            troot = self.modregulafalsi(vlrepi, t0, t1)
            return troot

    def modregulafalsi(self, vlrepi, t0, t1):
        # The modified regula falsi method:
        # Initial conditions:
        k = 0
        ak, bk = t0, t1
        tolerance = 1.0
        ck = self.newx_regfalsi(vlrepi, ak, bk)
        while not (abs(self.readyfunction(vlrepi, ck)) < tolerance):
            k += 1
            alpha, beta = 1, 1
            fak = self.readyfunction(vlrepi, ak)
            fck = self.readyfunction(vlrepi, ck)
            # If root is between ak and ck:
            if (fak * fck) <= 0:
                # If ck is the root, return position:
                if fck == 0:
                    return ck
                bk = ck
                # Update alpha
                if k > 1 and (fck * fck1) > 0:
                    alpha = 0.5
            # If root is between ck and bk:
            else:
                ak = ck
                # Update beta
                if k > 1 and (fck * fck1) > 0:
                    beta = 0.5
            fck1 = fck
            ck = self.newx_modregfalsi(vlrepi, ak, bk, alpha, beta)
        return ck

    def prepareinput(self, vlrepi, t0, t1):
        # This function checks that initial guesses t0 and t1
        # will yield f(t0)f(t1) < 0.
        f0 = self.readyfunction(vlrepi, t0)
        f1 = self.readyfunction(vlrepi, t1)
        negcheck = f0 * f1
        # If passes test, return both values
        if negcheck < 0:
            return t0, t1
        else:
            # If one of the values is a root, return that value
            if f0 == 0:
                print('Your initial guess {0:4.2f} is a root.'.format(t0))
                return t0, None
            elif f1 == 0:
                print('Your initial guess {0:4.2f} is a root.'.format(t1))
                return None, t1
            # Find new initial guesses
            else:
                t0, t1 = self.findnewinput(vlrepi, t0, t1)
                return t0, t1

    def findnewinput(self, vlrepi, t0, t1):
        # This function returns x0 and x1, the initial input for
        # the regula falsi method.
        # 'for' instead of 'while' to avoid infinite loop
        for k in range(1000):
            f0 = self.readyfunction(vlrepi, t0)
            f1 = self.readyfunction(vlrepi, t1)
            if f0 * f1 < 0:
                return t0, t1
            # If one of the values is a root, return that value
            elif f0 == 0:
                return t0, None
            elif f1 == 0:
                return None, t1
            # Since exponential decay, we know decreasing function:
            else:
                if f0 < 0:
                    tcheck = t0 / 2
                    t0, t1 = tcheck, t0
                else:
                    tcheck = t1 * 2
                    t0, t1 = t1, tcheck

    def readyfunction(self, vlrepi, t):
        # This function calculates the difference of Tj(t)*xij and rhs
        # rhs = qready = 5 * self.box.cap * self.h * self.cookies.surfarea
        # delete common terms: h and surfarea
        # rhs = 5 * self.box.cap
        # add independent portion of left side to rhs to get:
        rhs = 5 * self.boxcap + self.tempamb * len(vlrepi)
        tempsum = 0
        for j in vlrepi:
            tempsum += self.cookies.get(j).gettemp(t, self.tempamb, self.h)
        # f(xi) = lhs - rhs
        fofxi = tempsum - rhs
        return fofxi

    def newx_regfalsi(self, vlrepi, ta, tb):
        # This function returns a new position via the regula falsi method
        fa = self.readyfunction(vlrepi, ta)
        fb = self.readyfunction(vlrepi, tb)
        ck = (ta * fb - tb * fa) / (fb - fa)
        return ck

    def newx_modregfalsi(self, vlrepi, ta, tb, alpha, beta):
        # This function returns a new position via modified regula falsi method
        fa = self.readyfunction(vlrepi, ta)
        fb = self.readyfunction(vlrepi, tb)
        ck = (ta * beta * fb - tb * alpha * fa) / (beta * fb - alpha * fa)
        return ck

    def noreplacement(self, solid, x):
        # This function ensures that x respects the "no replacement constraint
        # Return false if an error is detected; the code should never violate
        # this, so if error output is present, algorithm has a bug.
        itemspresent = np.sum(x, axis=0)
        for j in range(self.n):
            if itemspresent[j] > 1:
                raise RuntimeError('Solution {0} has a physicality error: '
                                   'item {1}'.format(solid, j))

    def boxcapacity(self, solid, x):
        # This function ensures that no box is filled beyond capacity
        # Return false if an error is detected; the code should never
        # violate this, so if error output is present, algorithm has a bug.
        boxitems = np.sum(x, axis=1)
        for i in range(self.n):
            if boxitems[i] > self.boxcap:
                raise RuntimeError('Error: Solution {0} has filled bin {1} '
                                   'beyond capacity.'.format(solid, i))

    def timeconstraint(self, solution):
        # This function ensures that no cookie is put in a box before it
        # even gets out of the oven. Returns a list of (i,j) tuples for each
        # violation.
        solid = solution.getid()
        x = solution.getx()
        tfill = solution.gettfill()
        violations = []
        for i in range(self.n):
            for j in range(self.n):
                if x[i, j] == 1:
                    baked = self.cookiedonebaking(j, tfill[i])
                    if baked is False:
                        print('Error: Solution', solid, 'has packed cookie', j,
                              'in box', i, 'before it finished baking.')
                        violations.append((i, j))
        return violations

    def rackcapacity(self, x, tfill):
        # This function checks that the cooling rack is never be filled
        # beyond capacity and collects a list of violations as (i,j) tuples.
        timeintervals = self.gettimeintervals(tfill)
        violations = []
        for t in timeintervals:
            cookiesonrack = self.find_cookies_on_rack(t, tfill, x)
            if len(cookiesonrack) > self.coolrack:
                violations.append([t, cookiesonrack])
        return violations

    def find_cookies_on_rack(self, t, tfill, x):
        # This module collects a list of cookies on the cooling rack at time t
        # Cookies from boxes filled after t might be on rack
        timecheckindices = np.where(tfill > t)
        cookiesonrack = []
        for i in timecheckindices[0]:
            for j in range(self.n):
                if x[i, j] == 1:
                    onrack = self.rackij(t, tfill[i], self.cookies.get(j))
                    if onrack == 1:
                        cookiesonrack.append((i, j))
        return cookiesonrack

    def period_fill_limit(self, solution):
        # This function checks that the number of casks filled in each time
        # period does not exceed the limit.
        tfill = solution.gettfill()
        t_t = self.get_filltime_periods(tfill)
        res_fill = self.make_filltime_residuals(t_t, tfill, solution.gety())
        # Check each time period
        violations = []
        for t in range(len(t_t) - 1):
            if res_fill[t] < 0:
                violations.append(t_t[t])
        return violations

    def make_filltime_residuals(self, t_t, tfill, y):
        # This module forms the residual matrix for the fill period limit.
        m = np.sum(y)
        res_fill = []
        for t in range(len(t_t) - 1):
            p_t = [i for i in range(m) if t_t[t] <= tfill[i] < t_t[t + 1]]
            res_fill.append(self.fillcap - len(p_t))
        return res_fill

    def get_filltime_periods(self, tfill):
        # Get time periods that define the fill limit
        t_end = max(np.amax(tfill), self.tbatch * self.nbatches)
        n_period = int(2 * (t_end - self.tbatch) // self.tbatch) + 2
        t_t = [self.tbatch * (1.0 + t / 2.0) for t in range(n_period)]
        return t_t

    def gettimeintervals(self, tfill):
        # This module collects a list of time intervals at which rackcapacity()
        # should check.
        times = np.unique(tfill)
        timeintervals = [self.tbatch]
        for i in range(len(times)):
            # Want at least 300 seconds in between intervals
            if times[i] >= timeintervals[-1] + 300:
                timeintervals.append(times[i])
        # Also want when cookies are removed from the oven
        for bj in range(self.nbatches):
            ti = (bj + 1) * self.tbatch
            if ti not in timeintervals:
                timeintervals.append(ti)
        timeintervals.sort()
        return timeintervals

    def rackij(self, t, tfill, cookie):
        # This function determines if cookie is present on the cooling rack
        # at time t. It returns 1 for present and 0 for absent.
        t0 = cookie.batch * self.tbatch
        if t0 <= t < tfill:
            return 1
        else:
            return 0

    def fixcoolingrack(self, violations, solution):
        # This module brings solutions inside the feasible region by
        # removing cookies from the cooling rack inside the time ranges
        # specified by the violations list.
        print('Fixing cooling rack for solution number', solution.getid())
        while violations:
            # First solution: change load time of a box:
            for tv in range(len(violations)):
                t = violations[tv][0]
                solution = self.edittfill2(t, solution)
            # Second solution: make new boxes:
            violations = self.rackcapacity(solution.getx(), solution.gettfill())
            for tv in range(len(violations)):
                t = violations[tv][0]
                solution = self.openonebox(t, solution)
            # Check new solution for more rack capacity violations
            violations = self.rackcapacity(solution.getx(), solution.gettfill())
            if violations:
                # Backup solution: move items to new box
                solution = self.movecookies(violations[0][0], violations[0][1], solution)
                violations = self.rackcapacity(solution.getx(), solution.gettfill())
        checkformismatch(solution)
        return solution

    def edittfill(self, t, solution):
        # This function edits the load time of some boxes to reduce the number
        # of cookies on the cooling rack at one time.
        clist = self.find_cookies_on_rack(t, solution.gettfill(), solution.getx())
        print('     Need to remove {0:d} cookies from the cooling rack at time '
              '{1:6.2f} sec.'.format(len(clist)-self.coolrack, t))
        i = -1                          # Initialize i as nonexistent box
        booleanbins = []
        overcapnum = len(clist)
        k = 0
        while k < overcapnum:
            if len(clist) <= self.coolrack:
                return solution
            # If box i hasn't been explored before:
            if clist[k][0] != i:
                vlrep = solution.getvlrep()
                # Check box i to see if can reduce tfill[i]
                i = clist[k][0]
                cookieboolean = self.packatt(vlrep[i], t)
                booleanbins.append([i, cookieboolean])
                # Try to swap items
                if len(booleanbins) > 1:
                    icb, kcb, solution, booleanbins, clist = \
                        self.swapandfill(t, booleanbins, solution, clist)
                else:
                    icb = i
                    kcb = 0
                # If true, change tfill[i] and remove from clist
                if all(booleanbins[kcb][1]) is True:
                    solution.edit_tfilli(icb, t)
                    vlrep = solution.getvlrep()
                    clist = self.removefromclist(icb, vlrep[icb], clist)
                    overcapnum = len(clist)
            k += 1
        return solution

    def updateclist(self, clist, solution):
        # This function updates clist to what is present in vlrep because
        # earlier edits might have changed locations.
        vlrep = solution.getvlrep()
        k = 0
        for tpl in range(len(clist)):
            # i = clist[k][0], j = clist[k][1]
            if clist[k][1] not in vlrep[clist[k][0]]:
                del clist[k]
                k -= 1
            k += 1
        return clist

    def swapandfill(self, t, booleanbins, solution, clist):
        # This function swaps items between boxes and then edits tfill to reduce
        # the number of cookies on the cooling rack at time t.
        cb = self.findcboolbin(booleanbins)
        i = booleanbins[cb][0]
        solution, booleanbins, clist = self.swapitems(i, cb, t, booleanbins,
                                                      solution, clist)
        booleanbins[cb][1] = self.packatt(solution.getvlrep()[i], t)
        return i, cb, solution, booleanbins, clist

    def findcboolbin(self, booleanbins):
        # This function returns the booleanbin with the minimum number of False
        # values
        numfalse = np.zeros(len(booleanbins))
        for b in range(len(booleanbins)):
            numfalse[b] = booleanbins[b][1].count(False)
        cb = np.argmin(numfalse)
        return cb

    def swapitems(self, i, cb, t, booleanbins, solution, clist):
        # Swap in solution: x-matrix, vlrep; swap in cookieboolean
        vlrep = solution.getvlrep()
        for j1 in range(len(booleanbins[cb][1])):
            # Swap the cookies in bin cb that are false:
            if booleanbins[cb][1][j1] is False:
                j = vlrep[i][j1]
                i2, j2, k1, k2 = self.findcforswap(i, j, booleanbins,
                                                       solution.gettfill(), vlrep)
                if i2:
                    # Swap in solution
                    solution.swapitems(i, j, i2, j2)
                    # Swap in clist
                    if (i, j) in clist:
                        clist.remove((i, j))
                        clist.append((i2, j))
                    if (i2, j2) in clist:
                        clist.remove((i2, j2))
                        clist.append((i, j2))
                    # Update vlrep and booleans to reflect swap
                    vlrep = solution.getvlrep()
                    booleanbins[k1][1] = self.packatt(vlrep[i2], t)
        return solution, booleanbins, clist

    def findcforswap(self, i, j, booleanbins, tfill, vlrep):
        # This function finds a cookie in another bin that can be swapped with one
        # in box i. booleanbins is a list of lists: [bin i, [true/false values]]
        # Find a cookie from another bin that's ready:
        for k1 in range(len(booleanbins)):
            i2 = booleanbins[k1][0]
            if i2 != i:
                for k2 in range(len(booleanbins[k1][1])):
                    if booleanbins[k1][1][k2] is True:
                        j2 = vlrep[i2][k2]
                        # Before swap, verify packable:
                        pack1 = self.cookiedonebaking(j, tfill[i2])
                        # Wanting to make box i pack at time t:
                        # pack2 = self.cookiedonebaking(j2, t)
                        if pack1 is True:
                            return i2, j2, k1, k2

    def fillbin(self, i, t, solution, booleanbins, clist):
        # This function attempts to fill up box i with other cookies that can
        # be packed at time t.
        vlrep = solution.getvlrep()
        if len(vlrep[i]) < self.boxcap:
            addx = self.boxcap - len(vlrep[i])
            for ad in range(addx):
                try:
                    i2, j2, k1, k2 = self.findcformove(i, booleanbins, vlrep)
                    solution.moveitem(i2, j2, i)
                    # Move in clist:
                    if (i2, j2) in clist:
                        clist.remove((i2, j2))
                        clist.append((i, j2))
                    # Update vlrep and booleans to reflect move
                    vlrep = solution.getvlrep()
                    booleanbins[k1][1] = self.packatt(vlrep[i2], t)
                except:
                    break
        return solution, booleanbins, clist

    def findcformove(self, i, booleanbins, vlrep):
        # This function finds a cookie in another bin that can be moved box i.
        for k1 in range(len(booleanbins)):
            i2 = booleanbins[k1][0]
            if i2 != i:
                for k2 in range(len(booleanbins[k1][1])):
                    if booleanbins[k1][1][k2] is True:
                        j2 = vlrep[i2][k2]
                        return i2, j2, k1, k2

    def removefromclist(self, i, vlrepi, clist):
        # This function removes the cookies in box i from the list of cooking
        # violating the cooling rack capacity.
        # Remove cookies from current list edit:
        for j in vlrepi:
            if (i, j) in clist:
                clist.remove((i, j))
        return clist

    def openonebox(self, t, solution):
        # This function moves some cookies into a new box at time t.
        clist = self.find_cookies_on_rack(t, solution.gettfill(), solution.getx())
        if len(clist) - self.coolrack <= 0:
            return solution
        print('     Need to remove {0:d} cookies from the cooling rack at time '
              '{1:6.2f} sec.'.format(len(clist) - self.coolrack, t))
        clist.sort(key=lambda pair: pair[1])
        numrm = 0
        for k in range(len(clist)):
            # Return solution if clist can now fit on rack
            if len(clist) - numrm <= self.coolrack:
                return solution
            j = clist[k][1]
            i = [i for i in range(solution.getopenbins()) if j in solution.vlrep[i]][0]
            pack = self.cookiedonebaking(j, t)
            if pack is True:
                if numrm % self.boxcap == 0:
                    solution.opennewbin(i, j, t)
                else:
                    inew = len(solution.getvlrep()) - 1
                    solution.moveitem(i, j, inew)
                numrm += 1
        return solution

    def movecookies(self, t, clist, solution):
        # This function moves some of the cookies on rack at time t into a new box
        # Make list of cookie temperatures at time t
        print('     Need to remove {0:d} cookies from the cooling rack at time '
              '{1:6.2f} sec.'.format(len(clist) - self.coolrack, t))
        templist = [self.cookies.get(j).gettemp(t, self.tempamb, self.h)
                    for (i, j) in clist]
        # for i, j in clist:
        #     jtemp = self.cookies.get(j).gettemp(t, self.tempamb, self.h)
        #     templist.append(jtemp)
        # Move coldest cookies into a new box
        nmove = len(clist) - self.coolrack
        clist = [(i, j) for (temp, (i, j)) in sorted(zip(templist, clist))]
        solution.repackitemsatt(t, clist[:nmove])
        return solution

    def packatt(self, binitems, t):
        # This function checks all the cookies in binitems to see if they can
        # be packed at time t.
        cookieboolean = []
        for j in binitems:
            cookieboolean.append(self.cookiedonebaking(j, t))
        return cookieboolean

    def cookedbyt(self, binitems, t):
        # This function goes through list binitems and removes any cookie index
        # of a cookie that is not done cooking by time t.
        cookedlist = list(binitems)
        for j in binitems:
            if self.cookiedonebaking(j, t) is False:
                cookedlist.remove(j)
        return cookedlist

    def cookiedonebaking(self, j, t):
        # This function checks if cookie j is out of the oven by time t
        # Return True if cookie is out of the oven, otherwise return False
        bk = self.cookies.get(j).getbatch()
        if bk * self.tbatch > t:
            return False
        else:
            return True

    def edittfill2(self, t, solution):
        # Faster alternative to edittfill()
        clist = self.find_cookies_on_rack(t, solution.gettfill(), solution.getx())
        nremove = len(clist) - self.coolrack
        if nremove <= 0:
            return solution
        print('     Need to remove {0:d} cookies from the cooling rack at time '
              '{1:6.2f} sec.'.format(nremove, t))
        # Make list of bins to visit
        ivals = [i for (i, j) in clist]
        ibins = list(set(ivals))
        rcl_tfill = self.get_move_restrictions(solution)
        # Edit one bin at a time
        for i in ibins:
            if len(clist) <= self.coolrack:
                return solution
            if solution.getopenbins() <= i:
                return solution
            # Remove hot cookies from bin i to allow earlier t_fill
            solution, clist, rcl_tfill \
                = self.removehotcookies(i, t, solution, clist, rcl_tfill)
            vlrep = solution.getvlrep()
            cookieboolean = self.packatt(vlrep[i], t)
            if all(cookieboolean) is True:
                t_new, rcl_tfill = self.find_new_tfilli(vlrep[i], rcl_tfill)
                if t_new and t_new <= t:
                    clist = self.removefromclist(i, vlrep[i], clist)
                    solution, clist = self.fillbin2(i, t_new, solution, clist)
                    solution.edit_tfilli(i, t_new)
                else:
                    clist = self.removefromclist(i, vlrep[i], clist)
                    solution, clist = self.fillbin2(i, t, solution, clist)
                    solution.edit_tfilli(i, t)
        return solution

    def removehotcookies(self, i, t, solution, clist, rcl_tfill):
        # This function goes through bin i to remove cookies that can't be
        # filled at time t.
        vlrep = solution.getvlrep()
        tfill = solution.gettfill()
        cookieboolean = self.packatt(vlrep[i], t)
        if any(cookieboolean) is False:
            return solution, clist, rcl_tfill
        k2 = 0
        for k1 in range(len(vlrep[i])):
            if cookieboolean[k1] is False:
                j = vlrep[i][k2]
                i1 = self.findbinformove(i, j, vlrep, tfill)
                if i1:
                    rcl_tfill.adapt_movebins(tfill[i], tfill[i1])
                    solution.moveitem(i, j, i1)
                    # Move in clist:
                    if (i, j) in clist:
                        clist.remove((i, j))
                        clist.append((i1, j))
                    k2 -= 1
            k2 += 1
        return solution, clist, rcl_tfill

    def findbinformove(self, i, j, vlrep, tfill):
        # This function finds an available box to move cookie j to that
        # is not box i.
        timecheckindices = np.where(tfill > tfill[i])
        for inew in timecheckindices[0]:
            if inew >= len(vlrep):
                break
            if len(vlrep[inew]) < self.boxcap:
                pack1 = self.cookiedonebaking(j, tfill[inew])
                if pack1 is True:
                    return inew
        return None

    def fillbin2(self, i, t, solution, clist):
        # This function attempts to fill up box i with other cookies that can
        # be packed at time t.
        vlrep = solution.getvlrep()
        if len(vlrep[i]) < self.boxcap:
            addx = self.boxcap - len(vlrep[i])
            for ad in range(addx):
                i2, j2 = self.findcformove2(i, t, clist)
                if j2 is None:
                    return solution, clist
                else:
                    try:
                        solution.moveitem(i2, j2, i)
                        # Move in clist:
                        if (i2, j2) in clist:
                            clist.remove((i2, j2))
                    except:
                        return solution, clist
        return solution, clist

    def findcformove2(self, i, t, clist):
        # This function finds a cookie in another bin that can be moved box i.
        # Sort clist by j
        clist.sort(key=lambda pair: pair[1])
        for (i2, j2) in clist:
            if i2 != i:
                pack2 = self.cookiedonebaking(j2, t)
                if pack2 is True:
                    return i2, j2
        return i, None

    def fix_tfill(self, violations, solution):
        # This function corrects violations in tfill to make an infeasible
        # solution feasible with regard to period_fill_limit.
        print('Fixing period limit violations for solution number {0}.'
              .format(solution.getid()))
        rcl_tfill = self.get_move_restrictions(solution)
        loop = 0
        while violations:
            solution, rcl_tfill = self.select_fix_mode(loop, violations,
                                                       rcl_tfill, solution)
            violations = self.period_fill_limit(solution)
            loop += 1
        return solution

    def select_fix_mode(self, loop, violations, rcl_tfill, solution):
        # This module sends solution instance to a fix module based on the
        # loop number.
        if loop == 0 or loop % 2 == 0:
            for t in violations:
                solution, rcl_tfill = self.new_tfill(t, rcl_tfill,  solution)
        else:
            t = violations[0]
            if (loop + 1) % 4 == 0:
                solution, rcl_tfill = self.open_colderbox(t, rcl_tfill, solution)
            solution, rcl_tfill = \
                self.remove_colder_cookies(t, rcl_tfill, solution)
            solution, rcl_tfill = self.empty_one_box(t, rcl_tfill, solution)
        return solution, rcl_tfill

    def new_tfill(self, t, rcl_tfill, solution):
        # This module finds new tfill values based on the restricted candidate
        # list and the residual matrix for t_t
        print('     Changing the number of boxes filled at time period', t)
        tfill = solution.gettfill()
        vlrep = solution.getvlrep()
        # Get set of boxes filled during the period starting with t
        p_t = [i for i in range(len(vlrep))
               if t <= tfill[i] < t + self.tbatch / 2.0]
        k2 = 0                              # Counter for changes to tfill
        nedit = len(p_t) - self.fillcap     # Number of changes needed
        # Move through list backward
        for p_i in range(len(p_t) - 1, 0, -1):
            i = p_t[p_i]
            if k2 == nedit:
                return solution, rcl_tfill
            told = tfill[i]
            t_new, rcl_fill = self.find_new_tfilli(vlrep[i], rcl_tfill)
            if t_new:
                solution.edit_tfilli(i, t_new)
                # Adapt Greedy Function
                rcl_tfill.adapt_changetime(told, t_new, len(vlrep[i]))
                k2 += 1
        return solution, rcl_tfill

    def find_new_tfilli(self, vlrepi, rcl_tfill):
        # This function finds a reasonable new value for tfill[i]
        tmin = self.get_box_tmin(vlrepi)
        args = (tmin, len(vlrepi))
        # Get new tfill value and find place in t_t periods
        t_new, p_t, rcl_tfill = self.get_new_fill_value(rcl_tfill, *args)
        if not t_new:
            return None, rcl_tfill
        kappa = 0                       # Counter to exit loop
        # Check if possible to load in period
        while rcl_tfill.res_fill[p_t] < 1:
            # If not possible, find new time value
            t_new, p_t, rcl_tfill = self.get_new_fill_value(rcl_tfill, *args)
            if not t_new or kappa == 10:
                return None, rcl_tfill
            kappa += 1
        # If returning t_new to open bin, reduce fill capacity by 1
        rcl_tfill.res_fill[p_t] -= 1
        return t_new, rcl_tfill

    def empty_one_box(self, t, rcl_tfill, solution):
        # This module empties the least-filled bin in time period t
        print('     - empty one box')
        tfill = solution.gettfill()
        vlrep = solution.getvlrep()
        # Get set of boxes filled during the period starting with t
        p_t = [i for i in range(len(vlrep))
               if t <= tfill[i] < t + self.tbatch / 2.0]
        i_empty = self.pick_bin_to_empty(p_t, vlrep)
        for j in list(vlrep[i_empty]):
            solution, rcl_tfill = \
                self.move_cookie_j(i_empty, j, solution, rcl_tfill)
        return solution, rcl_tfill

    def pick_bin_to_empty(self, p_t, vlrep):
        rannum = random()
        if rannum < 0.5:
            # Get lengths of bins in p_t
            bin_sizes = [len(vlrep[i]) for i in p_t]
            # Determine emptiest bin
            k = np.argmin(np.array(bin_sizes))
            return p_t[k]
        else:
            return choice(p_t)

    def remove_colder_cookies(self, t, rcl_tfill, solution):
        # This function removes colder cookies from boxes filled during period t
        print('     - remove cold cookies')
        tfill = solution.gettfill()
        vlrep = solution.getvlrep()
        # Get set of boxes filled during the period starting with t
        p_t = [i for i in range(len(vlrep))
               if t <= tfill[i] < t + self.tbatch / 2.0]
        b_max = self.get_max_batch(t)
        for i in p_t:
            if len(vlrep[i]) > 1:
                solution, rcl_tfill = \
                    self.move_colder_cookies_out(i, t, rcl_tfill, solution, b_max)
        return solution, rcl_tfill

    def move_colder_cookies_out(self, i, t, rcl_tfill, solution, b_max):
        # This module moves at most 25% of a bin to other boxes
        vlrep = solution.getvlrep()
        n_max = len(vlrep[i]) // 4
        nmove = 0
        for j in vlrep[i]:
            if self.cookies.get(j).getbatch() <= b_max:
                solution, rcl_tfill = \
                    self.move_cookie_j(i, j, solution, rcl_tfill, tmax=t)
                nmove += 1
                if nmove == n_max:
                    return solution, rcl_tfill
        return solution, rcl_tfill

    def open_colderbox(self, t, rcl_tfill, solution):
        # This function opens a new bin earlier than time t
        tfill = solution.gettfill()
        vlrep = solution.getvlrep()
        # Get set of boxes filled during the period starting with t
        p_t = [i for i in range(len(vlrep))
               if t <= tfill[i] < t + self.tbatch / 2.0]
        i, j = self.pick_cookie_to_move(t, p_t, vlrep)
        # Make sure there's a cookie to move
        if j:
            tnew = self.get_newbox_tvalue(t, j, rcl_tfill)
            if tnew:
                solution.opennewbin(i, j, tnew)
                rcl_tfill.adapt_greedy_function_newbin(tnew, add=0)
                rcl_tfill.adapt_movebins(tfill[i], tnew)
        return solution, rcl_tfill

    def move_cookie_j(self, i, j, solution, rcl_tfill, tmax=None):
        # This function moves cookie j out of its current box into a new one
        # Get list of bins to move to and pick random bin
        rcl_bins = self.get_rcl_bins(i, j, solution.getvlrep(),
                                     solution.gettfill(), rcl_tfill, tmax=tmax)
        if rcl_bins:
            kwargs = {'iold': i, 'j': j, 'rcl_bins': rcl_bins,
                      'solution': solution, 'rcl_tfill': rcl_tfill}
            solution, rcl_tfill = self.move_cookie_to_ranbox(**kwargs)
        return solution, rcl_tfill

    def get_rcl_bins(self, i_old, j, vlrep, tfill, rcl_tfill, tmax=None):
        # This module returns a list of boxes that cookie j could move to
        # Be careful about setting tmax!
        tmin = self.cookies.get(j).getbatch() * 600
        if tmax:
            options = [i for i in range(len(vlrep)) if tmin < tfill[i] < tmax
                       and len(vlrep[i]) < self.boxcap]
        else:
            tmax = rcl_tfill.get_tmax(tmin, 1)
            # If j was put in a box after tmax, it won't overcrowd rack until then
            if tmax < tfill[i_old]:
                tmax = tfill[i_old]
            options = [i for i in range(len(vlrep)) if tmin < tfill[i] <= tmax
                       and len(vlrep[i]) < self.boxcap]
            if not options:
                options = [i for i in range(len(vlrep))
                           if tfill[i] == tmin and len(vlrep[i]) < self.boxcap]
        if i_old in options:
            options.remove(i_old)
        return options

    def move_cookie_to_ranbox(self, iold, j, rcl_bins, solution, rcl_tfill):
        # This function moves cookie j to one of the bins listed in rcl_bins
        tfill = solution.gettfill()
        inew = choice(rcl_bins)
        # Move cookie to new bin
        rcl_tfill.adapt_movebins(tfill[iold], tfill[inew])
        solution.moveitem(iold, j, inew)
        return solution, rcl_tfill

    def pick_cookie_to_move(self, t, p_t, vlrep):
        # This function ensures that an older cookie in p_t is chosen
        b_max = self.get_max_batch(t)
        i = self.pick_random_box(p_t, vlrep)
        # Open new box with coldest cookie in box i
        j = vlrep[i][0]
        count = 0
        while self.cookies.get(j).getbatch() >= b_max:
            if count == len(p_t):
                b_max += 1
            if count == 2 * len(p_t):
                return None, None
            i = self.pick_random_box(p_t, vlrep)
            j = vlrep[i][0]
            count += 1
        return i, j

    def pick_random_box(self, p_t, vlrep):
        # This function returns a random box belonging to set P_t that has more
        # than one cookie in it
        i = choice(p_t)
        while len(vlrep[i]) <= 1:
            i = choice(p_t)
        return i

    def get_max_batch(self, t):
        # This function sets a maximum batch to consider based on t
        b_max = 0.8 * t // self.tbatch
        return int(b_max)

    def get_box_tmin(self, vlrepi):
        # Find minimum time for box i
        boxi_contents = {k: v for k, v in self.cookies.items() if k in vlrepi}
        maxbatch = max(boxi_contents.values(), key=attrgetter('batch')).batch
        tmin = maxbatch * 600
        return tmin

    def get_new_fill_value(self, rcl_tfill, *args):
        # This function retrieves a new time value from the restricted candidate
        # list and updates the residual matrix for the fill periods
        t_new = rcl_tfill.get_new_t(args[0], nmove=args[1])
        if not t_new:
            return None, None, rcl_tfill
        t_p, rcl_tfill = self.find_t_in_fill_periods(t_new, rcl_tfill)
        return t_new, t_p, rcl_tfill

    def get_newbox_tvalue(self, t, j, rcl_tfill):
        # This function retrieves a time value for a new box based on cookie j
        # and trying to open the box during a time period that has the least
        # boxes filled
        tmin = self.cookies.get(j).getbatch() * 600
        # Determine valid range
        p_min, rcl_tfill = self.find_t_in_fill_periods(tmin, rcl_tfill)
        p_min += 1      # avoid period cookie j finished baking
        # Removing the cookie earlier than tfill[i] not a problem:
        p_max = self.find_max_period(t, rcl_tfill, tmin, p_min)
        tnew = self.pick_tnew_wperiod(rcl_tfill, p_min, p_max)
        return tnew

    def pick_tnew_wperiod(self, rcl_tfill, p_min, p_max):
        t_t = rcl_tfill.t_t
        tnew = 0
        open_range = np.array(rcl_tfill.res_fill[p_min:p_max + 1])
        possible_ps = list(range(p_max + 1 - p_min))
        while not rcl_tfill.time_period_feasible(tnew):
            if not possible_ps:
                return None
            pbest = np.argmax(open_range[np.array(possible_ps)])
            p = possible_ps[pbest] + p_min
            tnew = round(uniform(t_t[p], t_t[p + 1]), 1)
            # Remove period from list after visiting
            possible_ps.remove(int(p - p_min))
        return tnew

    def find_max_period(self, t, rcl_tfill, tmin, p_min):
        # This module returns the maximum period for searching
        tmax = rcl_tfill.get_tmax(tmin, 1)
        # Want to avoid the current period being violated
        if t < tmax:
            tmax = t
        p_max, rcl_tfill = self.find_t_in_fill_periods(tmax, rcl_tfill)
        for p in range(p_min, p_max + 1):
            if rcl_tfill.res_fill[p] < 0:
                p_max = p - 1
                return p_max
        return p_max

    def get_move_restrictions(self, solution):
        # Get restricted candidate list for new tfill options
        m = solution.getopenbins()
        tfill = solution.gettfill()
        n_b = self.n // self.nbatches
        rcl_tfill = RCLtime(self.coolrack, self.fillcap, n_b,
                            self.tbatch, self.nbatches)
        rcl_tfill.initialize_withtfill(m, solution.getvlrep(), tfill)
        return rcl_tfill

    def find_t_in_fill_periods(self, t, rcl_tfill):
        while t > rcl_tfill.t_t[-1]:
            rcl_tfill.extend_fill_periods()
        # Find the period containing t
        tlist = np.where(t >= np.array(rcl_tfill.t_t))[0]
        return tlist[-1], rcl_tfill

    def extend_fill_periods(self, t_t, res_fill):
        # This function extends t_t by one period
        t_t.append(t_t[-1] + 0.5 * self.tbatch)
        res_fill.append(self.fillcap)
        return t_t, res_fill


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


def fnds(setp):
    # This module performs the fast-non-dominated-sort described in Deb(2002).
    numsol = len(setp)
    fronts = []
    sp = []
    fhold = []
    nps = []
    for p in range(numsol):
        shold = []
        np = 0
        for q in range(numsol):
            if setp[p] != setp[q]:
                if dom1(setp[p].getfits(), setp[q].getfits()):
                    shold.append(setp[q])
                if dom1(setp[q].getfits(), setp[p].getfits()):
                    np += 1
        sp.append(shold)
        nps.append(np)
        if np == 0:
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


def xycheck(m, x, y):
    # This function verifies that the number of open bins in x and y agree
    n = len(y)
    itembins = np.sum(x, axis=1)
    for i in range(n):
        if (itembins[i] > 0 and y[i] == 1) is False:
            if (itembins[i] == 0 and y[i] == 0) is False:
                print('Solution', m, 'has an open bin error: bin', i)


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
    print('This file contains modules and classes related to the multi-'
          'objective problem. It is meant to be used in conjunction with'
          'an optimization algorithm.')