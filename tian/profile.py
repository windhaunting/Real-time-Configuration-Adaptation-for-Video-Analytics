# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:00:03 2019

@author: yanxi
"""

import numpy as np


def boundOks(oks, oks_min, preferHigher=True):
    assert oks.ndim == 2
    assert 0 <= oks_min <= 1
    n,m = oks.shape
    if preferHigher:
        fun = lambda x: np.nonzero(x > oks_min)[0][-1]
    else:
        fun = lambda x: np.nonzero(x > oks_min)[0][0]
    res = np.apply_along_axis(fun, 0, oks)
    return res
