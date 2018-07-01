# modregulafalsi.py
#   This python script gives a template for the modified regula falsi
#   method.
#   Author: Kristy Yancey Spencer


from __future__ import print_function
from random import random


class MyClass:
    def __init__(self):
        self.tolerance = 0.05  # Change this to be your preferred tolerance

    def runalgorithm(self):
        # Set up problem
        x0, x1 = 0.0, 1.0       # Change these to be your initial guesses
        x0, x1 = self.prepareinput(x0, x1)
        # If x0 is a root, return value:
        if x1 is None:
            return x0
        # If x1 is a root, return value:
        elif x0 is None:
            return x1
        else:
            xroot = self.modregulafalsi(x0, x1)
            return xroot

    def modregulafalsi(self, x0, x1):
        # The modified regula falsi method:
        # Initial conditions:
        k = 0
        ak, bk = x0, x1
        print('Initial range: a0 = {0:2.1f}, b0 = {1:2.1f} \n'.format(ak, bk))
        ck = self.newx_regfalsi(ak, bk)
        while not (abs(self.function(ck)) < self.tolerance):
            k += 1
            print('k = ', k, ':')
            alpha, beta = 1, 1
            fak = self.function(ak)
            fck = self.function(ck)
            print('c{0:2d} = {1:4.2f}: f(c{2:2d}) ~ {3:6.2f}'.format(k, ck, k, fck))
            # If root is between ak and ck:
            if (fak * fck) <= 0:
                # If ck is the root, return position:
                if fck == 0:
                    return ck
                bk = ck
                print('[a{0:2d}, b{1:2d}] = [{2:2.1f}, {3:2.1f}]'.format(k, k, ak, bk))
                # Update alpha
                if k > 1 and (fck * fck1) > 0:
                    alpha = 0.5
            # If root is between ck and bk:
            else:
                ak = ck
                print('[a{0:2d}, b{1:2d}] = [{2:2.1f}, {3:2.1f}] \n'.format(k, k, ak, bk))
                # Update beta
                if k > 1 and (fck * fck1) > 0:
                    beta = 0.5
            fck1 = fck
            ck = self.newx_modregfalsi(ak, bk, alpha, beta)
        return ck

    def prepareinput(self, x0, x1):
        # This function checks that initial guesses x0 and x1
        # will yield f(x0)f(x1) < 0.
        f0 = self.function(x0)
        f1 = self.function(x1)
        negcheck = f0 * f1
        # If passes test, return both values
        if negcheck < 0:
            return x0, x1
        else:
            # If one of the values is a root, return that value
            if f0 == 0:
                print('Your initial guess {0:4.2f} is a root.'.format(x0))
                return x0, None
            elif f1 == 0:
                print('Your initial guess {0:4.2f} is a root.'.format(x1))
                return None, x1
            # Find new initial guesses
            else:
                x0, x1 = self.findnewinput(x0, x1)
                return x0, x1

    def findnewinput(self, x0, x1):
        # This function returns x0 and x1, the initial input for
        # the regula falsi method.
        # 'for' instead of 'while' to avoid infinite loop
        for k in range(1000):
            f0 = self.function(x0)
            f1 = self.function(x1)
            if f0 * f1 < 0:
                break
            ck = self.newx_regfalsi(x0, x1)
            if ck < x0:
                x0, x1 = ck, x0
            elif ck > x1:
                x0, x1 = x1, ck
            else:
                rannum = random()
                if rannum < 0.5:
                    x0 = ck
                else:
                    x1 = ck
        return x0, x1

    def function(self, xi):
        # The function in question. You should change this to be
        # your specific function.
        fofxi = xi ** 3 + xi - 1
        return fofxi

    def newx_regfalsi(self, xa, xb):
        # This function returns a new position via the regula falsi method
        fa = self.function(xa)
        fb = self.function(xb)
        ck = (xa * fb - xb * fa) / (fb - fa)
        return ck

    def newx_modregfalsi(self, xa, xb, alpha, beta):
        # This function returns a new position via modified regula falsi method
        fa = self.function(xa)
        fb = self.function(xb)
        ck = (xa * beta * fb - xb * alpha * fa) / (beta * fb - alpha * fa)
        return ck


if __name__ == '__main__':
    instance = MyClass()
    root = instance.runalgorithm()
    print('You have a root at x=', root)
