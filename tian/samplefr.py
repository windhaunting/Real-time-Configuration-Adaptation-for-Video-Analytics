# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:52:17 2019

@author: tian
"""

import numpy as np

try:
    from . import utilPose
    from .feature import cart2speed
except:
    import utilPose
    from feature import cart2speed


def calcSpeed(v1, v2, n):
    #assert v1.ndim == 3
    #assert v1.shape[-2:] == (17,3)
    return (v1[:,:,:2] - v2[:,:,:2])/n


def calcSpeedRatio(v1, v2, n):
    #assert v1.ndim == 3
    #assert v1.shape[-2:] == (17,3)
    v = (v1[:,:,:2] - v2[:,:,:2])/n
    return np.sqrt(np.sum(v**2, 2))
    

# -------- fixed sample rate --------
    
class SampleMethod:
    def __init__(self, srate, method='ema', alpha=0.8):
        assert isinstance(srate, int) and srate >= 1
        assert method in ['keep', 'linear', 'ema']
        self.srate = srate
        self.method = method
        self.alpha = alpha
        if method == 'keep':
            self.speedFun = lambda kpm,i,sr : np.zeros((17,2))
            self.deltaFun = lambda speed, offset : speed
            self.updateFun = lambda spOld, spNew : spNew
        elif method == 'linear':
            self.speedFun = lambda kpNew,kpOld : calcSpeed(kpNew, kpOld, self.srate)
            self.deltaFun = lambda speed, offset : speed*offset
            self.updateFun = lambda spOld, spNew : spNew
        elif method == 'ema':
            self.speedFun = lambda kpNew,kpOld : calcSpeed(kpNew, kpOld, self.srate)
            self.deltaFun = lambda speed, offset : speed*offset
            self.updateFun = lambda spOld, spNew : spNew*alpha + spOld*(1-alpha)

    def calcSpeed(self, kpm, newIdx):
        return self.speedFun(kpm[:,newIdx], kpm[:,newIdx-self.srate])
    
    def calcDelta(self, speed, offset):
        return self.deltaFun(speed, offset)
    
    def calcUpdateSpeed(self, spOld, spNew):
        return self.updateFun(spOld, spNew)


def getGroupSpeed(kpm, srate, method='ema', alpha=0.8):  
    '''
    Compute the speed of each group. A group is sampled every <srate> frames.
        <srate> the number of frames in each group. <picked>,<pred>,...,<pred>
        <method> the method used to filter the speed
    '''
    assert isinstance(srate, int) and srate >= 1
    assert kpm.shape[-2:] == (17,3)
    if kpm.ndim == 3:
        kpm = kpm[np.newaxis, :]
    assert kpm.ndim == 4
    if srate == 1:
        s=np.diff(kpm[:,:,:,:2],1,1,prepend=0)
        s[:,0] = s[:,1]
        s = np.sqrt(np.sum(s**2,3))
        return s
    sm = SampleMethod(srate, method, alpha)
    n, ms = kpm.shape[:2]
    res = np.zeros((n, (ms+srate-1)//srate, 17))
    speed = sm.calcSpeed(kpm, srate)
    ii = 0
    res[:,0] = cart2speed(speed)
    for i in range(srate, ms, srate):
        speedOld = speed
        speedNew = sm.calcSpeed(kpm, i)
        speed = sm.calcUpdateSpeed(speedOld, speedNew)
        ii += 1
        res[:,ii] = cart2speed(speed)
        
    print ("getGroupSpeed, kp:", res.shape)
    
    return
    return res



def getTransKpm(kpm, srate, method='ema', alpha=0.8):
    '''
    <srate> the number of frames in each group. <picked>,<pred>,...,<pred>
    <method> the method used to filter the speed
    '''
    assert isinstance(srate, int) and srate >= 1
    assert kpm.shape[-2:] == (17,3)
    if kpm.ndim == 3:
        kpm = kpm[np.newaxis, :]
    assert kpm.ndim == 4
    assert method in ['keep', 'linear', 'ema']
    if srate == 1:
        return kpm.copy()
    sm = SampleMethod(srate, method, alpha)
    n, ms = kpm.shape[:2]
    rkpm = np.zeros_like(kpm)
    for i in range(0,srate):
        rkpm[:,i] = kpm[:,0]
    speed = sm.calcSpeed(kpm, srate)
    for i in range(srate, ms, srate):
        # copy sampled frames
        rkpm[:,i] = kpm[:,i]
        # update speed
        speedOld = speed
        speedNew = sm.calcSpeed(kpm, i)
        speed = sm.calcUpdateSpeed(speedOld, speedNew)
        # predict unsampled frames
        for j in range(1, srate+1):
            ii = i + j
            if ii >= ms:
                break
            rkpm[:,ii,:,0:2] = kpm[:,i,:,0:2] + sm.calcDelta(speed, j)
            rkpm[:,ii,:,2] = kpm[:,i,:,2]
    return rkpm


def calcGroupOks(kpm, rkpm, srate, refConf=0):
    assert kpm.ndim == 4
    assert kpm.shape[-2:] == (17,3)
    assert kpm.shape == rkpm.shape
    n,m = kpm.shape[:2]
    end = m - (m%srate)
    oks = utilPose.computeOKS_NtoMN(kpm[refConf,:end], rkpm[:,:end])
    roks = oks.reshape(n, m//srate, srate)
    return roks.mean(2)


def calcGroupPtm(ptm, srate):
    assert ptm.ndim == 2
    m = ptm.shape[1]
    return ptm[:,range(0,m-(m%srate),srate)]


def segment(kpm, ptm, srate, refConf=0, method='ema', alpha=0.8):
    assert kpm.ndim == 4 and kpm.shape[-2:] == (17,3)
    assert isinstance(srate, int) and srate >= 1
    if srate == 1:
        return calcGroupOks(kpm, kpm, 1, refConf), ptm.copy()
    rkpm = getTransKpm(kpm, srate, method, alpha)
    roks = calcGroupOks(kpm, rkpm, srate, refConf)
    rptm = calcGroupPtm(ptm, srate)
    return roks, rptm


# -------- slice --------

def sliceFR_sample(roks, goks, srate, gspeed, threshold):
    '''
    Get the average OKS by slice slow groups of <goks> and fast groups of <roks>
    Assume <roks> is reference OKS with 25 frame/sec.
    <goks> and <gspeed> are the OKS and speed at group level.
    <gspeed> is 1D
    '''
    assert roks.ndim == 1 and goks.ndim == 1
    assert isinstance(srate, int) and 1 <= srate <= 25
    assert len(goks) == len(gspeed) or len(goks) == len(gspeed) - 1
    if len(gspeed) == len(goks):
        flag = gspeed <= threshold
    else:
        flag = gspeed[:-1] <= threshold
    nS = sum(flag)
    acc1 = goks[flag].mean() if nS != 0 else 0.0
    n = len(roks)
    idx = np.nonzero(np.invert(flag))[0]*srate
    acc2 = 0
    if len(idx) > 0:
        for i in range(srate):
            acc2 += roks[idx+i].mean()
    acc2/=srate
    return (acc1 * nS + acc2 * (n - nS))/n, nS


def sliceFR_on2(hOks, hSrate, hSpeed, hTh, lOks, lSrate, lSpeed, lTh):
    '''
    Get the average OKS by slice slow groups of <srcOks> and fast groups of <tgtOks>
    <tgtOks>, <srcOks> and <srcSpeed> are at frame-level.
    <srcOks> are <srcSpeed> the OKS and speed of the fast groups.
    <tgtSrate> is larger than <srcSrate>
    <srcSpeed> is 1D
    '''
    assert tgtOks.ndim == 1 and srcOks.ndim == 1
    assert isinstance(tgtSrate, int) and isinstance(srcSrate, int)
    assert tgtSrate > srcSrate
    assert len(srcOks) == len(srcSpeed) or len(srcOks) == len(srcSpeed) - 1
    if len(gspeed) == len(goks):
        flag = gspeed <= threshold
    else:
        flag = gspeed[:-1] <= threshold
    nS = sum(flag)
    acc1 = goks[flag].mean() if nS != 0 else 0.0
    n = len(roks)
    idx = np.nonzero(np.invert(flag))[0]*srate
    acc2 = 0
    if len(idx) > 0:
        for i in range(srate):
            acc2 += roks[idx+i].mean()
    acc2/=srate
    return (acc1 * nS + acc2 * (n - nS))/n, nS

# -------- analysis --------

def __test__():
    dpath=''
    kpm=np.load(dpath+'kpm.npy')
    ptm=np.load(dpath+'ptm.npy')
    sigmas=utilPose.__KPT_OKS_SIGMAS__
    
    # data
    speedList=[]
    snList=[] # normalized speed
    sminList=[]
    smaxList=[]
    smeanList=[]
    for i in range(1,25+1):
        s=getGroupSpeed(kpm,i)
        speedList.append(s)
        snList.append(s/sigmas)
        smin=s.min(2)
        sminList.append(smin)
        smax=s.max(2)
        smaxList.append(smax)
        smean=s.mean(2)
        smeanList.append(smean)
    
    kpmList=[]
    for i in range(25):
        rkpm=getTransKpm(kpm,i+1)
        kpmList.append(rkpm)
        
    oksList=[]
    foksList=[]
    for i in range(25):
        roks=calcGroupOks(kpm, kpmList[i], i+1)
        oksList.append(roks)
        foks=utilPose.computeOKS_NtoMN(kpm[0], kpmList[i])
        foksList.append(foks)
        rptm=calcGroupPtm(ptm, i)
        
    ptmList=[]
    for i in range(25):
        ptmList.append(rptm)
    
    # dump
    import pickle
    with open(dpath+'sfr-analys.pkl','wb') as f:
        pickle.dump([speedList,kpmList,oksList,foksList],f,pickle.HIGHEST_PROTOCOL)
    # load
    with open(dpath+'sfr-analys.pkl', 'rb') as f:
        speedList,kpmList,oksList,foksList = pickle.load(f)
    
    # analysis
    smat=np.zeros((25,5,17))
    for i in range(25):
        smat[i]=speedList[i].mean(1)
    
    import matplotlib.pyplot as plt
    rId=1 # resolution id
    plt.plot([s[rId].max(1).mean() for s in snList])
    plt.plot([np.quantile(s[rId],16/17,1).mean() for s in snList])
    plt.plot([np.quantile(s[rId],15/17,1).mean() for s in snList])
    
    