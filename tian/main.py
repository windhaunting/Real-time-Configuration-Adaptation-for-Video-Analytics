# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:29:24 2019

@author: yanxi
"""

import numpy as np
import preprocess
import fpsconv
import feature


def processRecord2mat(dpath, fl):
    kpm,ptm,csm=preprocess.records2mat(dpath+'pose_estimation/', fl)
    np.save(dpath+'kpm',kpm)
    np.save(dpath+'ptm',ptm)
    np.save(dpath+'csm',csm)


def processMat2conf(dpath, name, fpsList, resolutionList, segSec=1, method='eam:0.8',
                    kpm=None, ptm=None):
    if not dpath.endswith('/'):
        dpath += '/'
    if kpm is None:
        kpm = np.load(dpath+'kpm.npy')
    if  ptm is None:
        ptm = np.load(dpath+'ptm.npy')
    roks, rptm = fpsconv.generateConfsFPS(kpm, ptm, fpsList, 25, segSec, method)
    np.savez(dpath+'name', roks, rptm, fpsList, resolutionList)


def extractFeatures(dpath, kpm):
    pass


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
    
    # step 2, all configurations
    fpsList = [25, 15, 10, 5, 2, 1]
    for i in range(start, end+1):
        folder = '%03d'%i
        print(folder)
        dpath = base + folder + '/'
        processMat2conf(dpath, 'conf-1s-cmu', fpsList, rl)
    
    # step 3, feature
    
    