# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:12:33 2019

@author: tian
"""

import numpy as np

try:
    from . import utilPose
    from . import fpsconv
except:
    import utilPose
    import fpsconv


def getDiff(kpm3, rkpm):
    assert kpm3.ndim == 3
    assert rkpm.ndim == 4
    assert kpm3.shape[0] == rkpm.shape[1]
    nConf, nFrm = rkpm.shape[:2]
    diff = np.zeros((nConf, nFrm, 17, 2))
    for i in range(nFrm):
        diff[:,i] = utilPose.computeDiff_1toN(kpm3[i], rkpm[:,i])
    return diff


def segmentDiff(diff, unit, method='simple'):
    assert diff.ndim == 4
    assert diff.shape[-2:] == (17,2)
    assert 1 < unit
    assert method in ['simple', 'exp']
    nConf, nFrm = diff.shape[:2]
    n = nFrm - (nFrm % unit)
    nGrp = n // unit
    if n != nFrm:
        diff = diff[:,:n,:,:]
    diff = diff.reshape(nConf, nGrp, unit, 17, 2)
    res = np.zeros((nConf, nGrp, 17, 2))
    res[:,:,:,1] = diff[:,:,:,:,1].mean(2)
    if method == 'simple':
        res[:,:,:,0] = diff[:,:,:,:,0].mean(2)
    else:
        res[:,:,:,0] = np.apply_along_axis(
                lambda x: np.log(np.exp(x).sum()/unit), 2, diff[:,:,:,:,0])
    return res
    

def speedDiff(kpm, fpsList):
    assert kpm.ndim == 4
    assert kpm.shape[-2:] == (17,3)
    nRsl, nFrm = kpm.shape[:2]
    nFps = len(fpsList)
    nConf - nRsl*nFps
    rkpm = fpsconv.generateConfsFPS_KPM(kpm, fpsList, 25)
    diff = getDiff(kpm[3], rkpm)
    df = segmentDiff(diff, 25)
        


