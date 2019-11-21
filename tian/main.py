# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:29:24 2019

@author: yanxi
"""

import numpy as np
import preprocess
import fpsconv
import feature
import configuration


__KPM_FILENAME__ = 'kpm.npy'
__PTM_FILENAME__ = 'ptm.npy'
__CSM_FILENAME__ = 'csm.npy'


def processRecord2mat(dpath, fl):
    kpm,ptm,csm=preprocess.records2mat(dpath+'pose_estimation/', fl)
    np.save(dpath+__KPM_FILENAME__,kpm)
    np.save(dpath+__PTM_FILENAME__,ptm)
    np.save(dpath+__CSM_FILENAME__,csm)


def processMat2conf(dpath, name, fpsList, resolutionList, segSec=1, method='ema:0.8',
                    kpm=None, ptm=None):
    if not dpath.endswith('/'):
        dpath += '/'
    if kpm is None:
        kpm = np.load(dpath+__KPM_FILENAME__)
    if  ptm is None:
        ptm = np.load(dpath+__PTM_FILENAME__)
    roks, rptm = fpsconv.generateConfsFPS(kpm, ptm, fpsList, 25, segSec, method)
    np.savez(dpath+name, oks=roks, ptm=rptm, fpsList=fpsList, rslList=resolutionList)


def loadConf(fname):
    if not fname.endswith('.npz'):
        fname += '.npz'
    data = np.load(fname, allow_pickle=True)
    return data['oks'], data['ptm'], data['fpsList'], data['rslList']


def processFeatures(dpath, name, kpm, srcFps, unit, method='ema'):
    if not dpath.endswith('/'):
        dpath += '/'
    if isinstance(kpm, str):
        kpm = np.load(dpath+__KPM_FILENAME__)
    fas = feature.featureAbsSpeed(kpm[[0]], srcFps, unit, method)
    f = feature.cart2speed(fas[0])
    np.save(dpath+name, f)


def processGroundTruth(dpath, name, confName, oks_min):
    oks, ptm, _, _ = loadConf(dpath+confName)
    c = configuration.boundOks(oks, ptm, oks_min)
    np.save(dpath+name, c)


def main(start=1, end=13):
    base='E:/Data/video-pose/'
    rl = preprocess.listResolution('cmu', False) # high to low
    fl = [preprocess.makeERFilename(r,25,'cmu') for r in rl]
    print(fl)
    # step 1, raw records to key-points
    for i in range(start, end+1):
        folder = '%03d'%i
        print(folder)
        dpath = base + folder + '/'
        processRecord2mat(dpath, fl)
    
    # step 2, oks and ptm for all configurations
    fpsList = [25, 15, 10, 5, 2, 1]
    for i in range(start, end+1):
        folder = '%03d'%i
        print(folder)
        dpath = base + folder + '/'
        processMat2conf(dpath, 'conf-1s-cmu', fpsList, rl, 1)
    
    # step 3, feature
    for i in range(start, end+1):
        folder = '%03d'%i
        print(folder)
        dpath = base + folder + '/'
        processFeatures(dpath, 'feature.npy', __KPM_FILENAME__, 25, 25, 'ema')
    
    # step 4, optimal output
    for i in range(start, end+1):
        folder = '%03d'%i
        print(folder)
        dpath = base + folder + '/'
        processGroundTruth(dpath, 'gt', 'conf-1s-cmu', 0.95)
    
    