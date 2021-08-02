# -*- coding: utf-8 -*-

"""
#import math
import numpy as np
import sympy as sym

from scipy.optimize import fsolve

def myFunction(z):
   x = z[0]
   y = z[1]
   w = z[2]

   F = np.empty((3))
   F[0] = x**2+y**2-20
   F[1] = y - x**2
   F[2] = w + 5 - x*y
   return F

zGuess = np.array([1,1,1])
z = fsolve(myFunction,zGuess)
print(z)


def solve_test():
    sym.init_printing()
    x,y,z = sym.symbols('x,y,z')
    c1 = 5# sym.Symbol('c1')
    f = sym.Eq(2*x**2+y+z,1)
    g = sym.Eq(x+2*y+z,c1)
    h = sym.Eq(-2*x+y,-z)
    
    print(sym.solve([f,g,h],(x,y,z)))
    
    print("res: ", x, y, z)

#solve_test()
    
def solve_test_1():
    
    sym.init_printing()

    a1, a2, a3, b1, b2, b3 = sym.symbols('a1,a2,a3,b1,b2,b3')
    
    eq1 = sym.Eq((a1 - a2 * sym.exp(-5/a3)) * (b1 - b2 * sym.exp(-8/b3)), 10)
    eq2 =  sym.Eq((a1 - a2 * sym.exp(-2/a3)) * (b1 - b2 * sym.exp(-1/b3)), 30)    
    eq3 =  sym.Eq((a1 - a2 * sym.exp(-6/a3)) * (b1 - b2 * sym.exp(-2/b3)), 35)

    
    eq4 = sym.Eq((a1 - a2 * sym.exp(-10/a3)) * (b1 - b2 * sym.exp(-8/b3)), 40)
    eq5 =  sym.Eq((a1 - a2 * sym.exp(-15/a3)) * (b1 - b2 * sym.exp(-5/b3)), 50)
    
    eq6 =  sym.Eq((a1 - a2 * sym.exp(-8/a3)) * (b1 - b2 * sym.exp(-21/b3)), 65)
    eq7 =  sym.Eq((a1 - a2 * sym.exp(-15/a3)) * (b1 - b2 * sym.exp(-25/b3)), 75)
    
    print("a1: ", a1)
    print(sym.solve([eq1, eq2, eq3, eq4, eq5, eq6, eq7],(a1, a2, a3, b1, b2, b3)))

    
#solve_test_1()


def solve_test_2():
    
    sym.init_printing()

    a1, a2, a3 = sym.symbols('a1,a2,a3')
    
    eq1 = sym.Eq((a1 - a2 * sym.exp(-5/a3)), 10)
    eq2 =  sym.Eq((a1 - a2 * sym.exp(-2/a3)),3)    
    eq3 =  sym.Eq((a1 - a2 * sym.exp(-6/a3)), 35)

    
    print("a1: ", a1)
    print(sym.solve([eq1, eq2, eq3],(a1, a2, a3)))

solve_test_2()

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return c - a * np.exp(-x/b)

x = np.linspace(0,4,3)
y = func(x, 1, 1, 1)
yn = y + 0.2*np.random.normal(size=len(x))

popt, pcov = curve_fit(func, x, yn)


y1 = func(1, *popt)
print("y1: ", y1)

plt.figure()
plt.plot(x, yn, 'ko', label="Original Noised Data")
plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
x = np.linspace(0,4,100)
plt.plot(x, func(x, *popt), 'r-', label="Fitted Exponential")

y2 = func(1, *popt)
print("y2: ", y2)

plt.legend()
plt.savefig('fit_curve.png')
plt.show()
