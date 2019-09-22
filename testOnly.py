#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 17:04:51 2019

@author: fubao
"""

import numpy as np
#from blist import blist 
arr = np.zeros((3, 2))

a = [1,2,3,4,3]
#arr[0:2, 0] = a

print (arr, a[:-1])


a = np.arange(10, 100)

print (a, np.where(a < 15))


print (np.argwhere(a<15))
print("len a:" , len(a))

