# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:29:15 2019

@author: yanxi
"""

import numpy as np


def cart2pol(data):
    s = data.shape
    assert s[-1] == 2
    # x and y
    t = data.reshape(-1,2)
    res = np.zeros_like(t)
    # rho
    res[:,0] = np.sqrt(t[:,0]**2 + t[:,1]**2)
    # phi
    res[:,1] = np.arctan2(t[:,1], t[:,0])
    return res.reshape(s)

def pol2cart(data):
    s = data.shape
    assert s[-1] == 2
    # r and p
    t = data.reshape(-1,2)
    res = np.zeros_like(t)
    # x
    res[:,0] = t[:,0] * np.cos(t[:,1])
    # y
    res[:,1] = t[:,0] * np.sin(t[:,1])
    return res.reshape(s)
    

def featureAbsSpeed(kpm, fps, unit, method='mean', alpha=0.8, weight=None):
    '''
    Input:
        <kpm> key point matrix (4d: conf-frame-kp-xyv)
        <fps> the FPS of the kpm data
        <unit> generate one feature using <unit> frames
        <method> how to merge the features of different frames
    '''
    assert kpm.ndim == 4
    assert kpm.shape[-2:] == (17,3)
    assert method in ['max', 'min', 'mean', 'ema', 'weight']
    nconf, nfrm, nkp = kpm.shape[:3]
    if nfrm % unit == 0:
        nfrm-=1
    nfrm = nfrm - (nfrm % unit) + 1
    diff = np.diff(kpm[:,:nfrm,:,[0,1]],n=1,axis=1)
    #utime = 1.0/fps
    diff = diff.reshape([nconf, -1, unit, nkp, 2])
    if method == 'mean':
        m = diff.mean(2)
    elif method == 'min':
        m = diff.min(2)
    elif method == 'max':
        m = diff.max(2)
    elif method == 'ema':
        f = np.ones(unit)
        for i in range(unit-1):
            f[:unit-i-1] *= alpha
        m = np.average(diff, axis=2, weights=f)
    elif method == 'weight':
        assert isinstance(weight, np.ndarray) and np.shape == (unit,)
        m = np.average(diff, axis=2, weights=weight)
    return m


def featureRelSpeed(kpm, fps, pairs, unit, method='mean', alpha=0.8, weight=None):
    '''
    Input:
        <pairs> a list of ID pairs
    '''
    assert kpm.ndim == 4
    assert kpm.shape[-2:] == (17,3)
    assert pairs.ndim ==2 and pairs.shape[1] == 2
    assert method in ['max', 'min', 'mean', 'ema', 'weight']
    nconf, nfrm, nkp = kpm.shape[:3]
    npair = pairs.shape[0]
    if nfrm % unit == 0:
        nfrm-=1
    nfrm = nfrm - (nfrm % unit) + 1
    r =[None for _ in range(npair)]
    for i in range(npair):
        x,y = pairs[i]
        r[i] = kpm[:,:,x,[0,1]] - kpm[:,:,y,[0,1]]
    ref = np.stack(r, axis=2)
    ref = ref.reshape([nconf, -1, unit, npair, 2])
    if method == 'mean':
        m = ref.mean(2)
    elif method == 'min':
        m = ref.min(2)
    elif method == 'max':
        m = ref.max(2)
    elif method == 'ema':
        f = np.ones(unit)
        for i in range(unit-1):
            f[:unit-i-1] *= alpha
        m = np.average(ref, axis=2, weights=f)
    elif method == 'weight':
        assert isinstance(weight, np.ndarray) and np.shape == (unit,)
        m = np.average(ref, axis=2, weights=weight)
    return m


def kp2feature(kpm):
    pass


def kp2featureConf(kpmList, conf):
    pass

