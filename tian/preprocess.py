# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:49:38 2019

@author: yanxi
"""

import pandas as pd
import re
import numpy as np

REGEX_KEY_POINT_ONE=re.compile(r'(0(?:\.\d*)?), (0(?:\.\d*)?), ([\d])')
REGEX_KEY_POINT_LIST=re.compile(r'\[(.+?)\],(\d+(?:\.\d+))')
NUM_KEY_POINTS=17


def kpString2Info(kpstring):
    m = REGEX_KEY_POINT_LIST.match(kpstring)
    if m:
        kpList = [ [float(t[0]), float(t[1]), int(t[2])] 
                 for t in REGEX_KEY_POINT_ONE.findall(m[1]) ]
        score = float(m[2])
        return kpList, score
    return None

'''
Process one KP file, contains key point lists and processing time.
Return three numpy.array:
    1, key points (3d-vector: x, y, v for each KP) for each frame (3d: frame-kp-xyv)
    2, processing time for each frame
    3, confidence score for each frame
'''
def record2mat(filename):
    df = pd.read_csv(filename, delimiter='\t')
    n = len(df)
    nh = df['numberOfHumans'].to_numpy()
    kplist = df['Estimation_result']
    kps=[None for _ in range(n)]
    cs=np.zeros(n)
    for i in range(n):
        if nh[i] == 1:
            kps[i], cs[i] = kpString2Info(kplist[i])
        elif nh[i] > 1:
            l = kplist[i].split(';')
            linfo = [ kpString2Info(t) for t in l ]
            maxScore=-1
            for t,s in linfo:
                if s > maxScore:
                    maxScore = s
                    kps[i] = t
            cs[i] = maxScore
        else:
            kps[i] = [[0.0, 0.0, 0] for _ in range(NUM_KEY_POINTS)]
            #cs[i] = 0.0
    pt = df['Time_SPF'].to_numpy()
    return np.array(kps), pt, np.cs


'''
Process a list of KP files, to get 2 big matrices
Return two numpy.array:
    1, key points (4d: file-frame-kp-xyv)
    2, processing time (2d: file-frame)
    3, confidence score (2d: file-frame)
'''
def records2mat(prefix, fnList):
    kpm = []
    ptm = []
    csm = []
    for fn in fnList:
        kp, pt, cs = record2mat(prefix+fn)
        kpm.append(kp)
        ptm.append(pt)
        csm.append(cs)
    n=min(len(t) for t in ptm)
    for i in range(len(fnList)):
        if len(ptm[i]) > n:
            kpm[i]=kpm[i][:n]
            ptm[i]=ptm[i][:n]
            csm[i]=csm[i][:n]
    return np.stack(kpm), np.stack(ptm), np.stack(csm)


RESOLUTION_LIST_DEFAULT = ['320x240','480x352','640x480','960x720','1120x832']
RESOLUTION_LIST_CMU = ['320x240','480x352','640x480','960x720','1120x832']
RESOLUTION_LIST_MOBILENET_V2 = ['320x240','480x352','640x480','960x720','1120x832']


def listResolution(model = None):
    if model is None:
        return RESOLUTION_LIST_DEFAULT
    elif model.lower() == 'cmu':
        return RESOLUTION_LIST_CMU
    elif model.lower() == 'mobilenet_v2':
        return RESOLUTION_LIST_MOBILENET_V2
    else:
        return []


def makeERFilename(resolution, fps, model):
    return '%s_%d_%s_estimation_result.tsv' % (resolution, fps, model.lower())



