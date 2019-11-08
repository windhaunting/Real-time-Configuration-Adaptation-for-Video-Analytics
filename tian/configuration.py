# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:00:03 2019

@author: yanxi
"""

import numpy as np


def boundOks(oks, ptm, oks_min, approx=False):
    '''
    Compute the configuration ID using the strategy of Bound-Oks and Minimize-Time
    Input:
        <oks> the OKS matrix (2d: conf-time)
        <ptm> the processing time matrix (2d: conf-time)
        <oks_min> the bound of OKS
        <approx> whether to use quick but approximate method (assume PT is linear to OKS)
    Output:
        A 1d int array of the selected configurations
    '''
    assert oks.ndim == 2
    assert 0 <= oks_min <= 1
    if approx:
        fun = lambda x: x[np.nonzero(x > oks_min)].argmin().astype(int)
        res = np.apply_along_axis(fun, 0, oks)
    else:
        assert oks.shape == ptm.shape
        n,m = oks.shape
        res=np.zeros(m, dtype=int)
        for i in range(m):
            nz = np.nonzero(oks[:,i] > oks_min)[0]
            res[i] = ptm[nz,i].argmin()
    return res


def boundPT(oks, ptm, pt_max, approx=False):
    '''
    Compute the configuration ID using the strategy of Bound-Time and Maximize-Oks
    Input:
        <oks> the OKS matrix (2d: conf-time)
        <ptm> the processing time matrix (2d: conf-time)
        <pt_min> the bound of processing time
        <approx> whether to use quick but approximate method (assume PT is linear to OKS)
    Output:
        A 1d int array of the selected configurations
    '''
    assert ptm.ndim == 2
    assert 0 <= pt_max
    if approx:
        fun = lambda x: x[np.nonzero(x < pt_max)].argmax().astype(int)
        res = np.apply_along_axis(fun, 0, ptm)
    else:
        assert oks.shape == ptm.shape
        n,m = ptm.shape
        res=np.zeros(m, dtype=int)
        for i in range(m):
            nz = np.nonzero(ptm[:,i] < pt_max)[0]
            res[i] = oks[nz,i].argmax()
    return res


def boundDelay(oks, spf, delay_min):
    pass

