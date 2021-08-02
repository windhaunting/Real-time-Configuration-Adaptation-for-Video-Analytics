#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 10:19:24 2021

@author: fubao
"""

# fit accuracy with frame rate/resolution



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def expo_func(x, a, b, c):
    return c - a * np.exp(-x/b)

def fit_function(data_points):
    # N point,  data_points: N x 2  at least three data points;  the more, the better
    
    
    x = [x for x, y in data_points]
    y = [y for x, y in data_points]
    
    popt, pcov = curve_fit(expo_func, x, y)

    
    plt.figure()
    plt.plot(x, y, 'ko', label="Data Points")
    plt.plot(x, expo_func(x, *popt), 'r-', label="Fitted Curve")
    
    x = np.linspace(0,4,100)
    plt.plot(x, expo_func(x, *popt), 'r-', label="the Exponential")
    
    plt.legend()
    plt.savefig('fit_accuracy_fr.png')
    #plt.show()
    
    return popt