# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:52:17 2019

@author: tian
"""

import numpy as np

try:
    from . import utilPose
except:
    import utilPose


def __calcSpeed(v1, v2, n):
    #assert v1.ndim == 3
    #assert v1.shape[-2:] == (17,3)
    return (v1[:,:,:2] - v2[:,:,:2])/n
    

def getGroupedKpm(kpm, srate, method='ema', alpha=0.8):
    '''
    <srate> the number of frames in each group. <picked>,<pred>,...,<pred>
    <method> the method used to filter the speed
    '''
    assert isinstance(srate, int) and srate > 1
    assert kpm.shape[-2:] == (17,3)
    if kpm.ndim == 3:
        kpm = kpm[np.newaxis, :]
    assert kpm.ndim == 4
    assert method in ['keep', 'linear', 'ema']
    if method == 'keep':
        funCalc = lambda i : np.zeros((17,2))
        funDelta = lambda speed,j : speed
        funUpdate = lambda speed : speed
    if method == 'linear':
        funCalc = lambda i : __calcSpeed(kpm[:,i], kpm[:,i-srate], srate)
        funDelta = lambda speed,j : speed*j
        funUpdate = lambda speed : speed
    if method == 'ema':
        assert 0 <= alpha <= 1
        speedOld = np.zeros((17,2))
        funCalc = lambda i : __calcSpeed(kpm[:,i], kpm[:,i-srate], srate)
        funDelta = lambda speed,j : speed*j
        funUpdate = lambda speed : speed*alpha + speedOld*(1-alpha)
    n, ms = kpm.shape[:2]
    rkpm = np.zeros_like(kpm)
    for i in range(0,srate):
        rkpm[:,i] = kpm[:,0]
    speed = funCalc(srate)
    for i in range(srate, ms, srate):
        # copy sampled frames
        rkpm[:,i] = kpm[:,i]
        # update speed
        speedOld = speed
        speedNew = funCalc(i)
        speed = funUpdate(speedNew)
        # predict unsampled frames
        for j in range(1, srate+1):
            ii = i + j
            if ii >= ms:
                break
            rkpm[:,ii,:,0:2] = kpm[:,i,:,0:2] + funDelta(speed, j)
            rkpm[:,ii,:,2] = kpm[:,i,:,2]
    return rkpm


def calcGroupOks(kpm, rkpm, srate, refConf=0):
    assert kpm.ndim == 4
    assert kpm.shape[-2:] == (17,3)
    assert kpm.shape == rkpm.shape
    n,m = kpm.shape[:2]
    rkpm = getGroupedKpm(kpm, srate)
    oks = np.zeros((n,m))
    for i in range(m):
        oks[:,i] = utilPose.computeOKS_list(kpm[refConf,i], rkpm[:,i])
    if m%srate != 0:
        oks = oks[:,:-(m%srate)]
    roks = oks.reshape(n, m//srate, srate)
    return roks.mean(2)


def calcGroupPtm(ptm, srate):
    assert ptm.ndim == 2
    m = ptm.shape[1]
    return ptm[:,range(0,m-(m%srate),srate)]


def segment(kpm, ptm, srate, method='eam', alpha=0.8):
    assert kpm.ndim == 4 and kpm.shape[-2:] == (17,3)
    assert isinstance(srate, int) and srate > 1
    rkpm = getGroupedKpm(kpm, srate, method, alpha)
    roks = calcGroupOks(kpm, rkpm, srate)
    rptm = calcGroupPtm(ptm, srate)
    return roks, rptm

