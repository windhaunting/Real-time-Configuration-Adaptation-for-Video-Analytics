# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:09:56 2019

@author: yanxi
"""

import numpy as np
import utilPose


# -------- part 1: methods for sampling frames when converting FPS --------

class FrameSampler:
    def __init__(self, tgtFps, srcFps=25, srcTime=None):
        assert 0 < tgtFps < srcFps
        assert srcTime is None or len(srcTime) == srcFps
        if srcTime is not None:
            assert np.abs(sum(srcTime) - 1.0) < 1e9
        else:
            srcTime = np.linspace(0.0, 1.0, srcFps, endpoint=False)
        self.tgtFps = tgtFps
        self.srcFps = srcFps
        self.scrTime = srcTime
        
        self.sfx = None
        self.sfi = None
        self.sti = None
        
        
    @staticmethod
    def calcSampleFrameId(tgtFps, srcFps=25):
        stime = np.linspace(0.0, 1.0, srcFps, endpoint=False)
        ttime = np.linspace(0.0, 1.0, tgtFps, endpoint=False)
        sf = [np.argmax(ttime[i]<=stime) for i in range(tgtFps)]
        return np.array(sf)
    
    @staticmethod
    def calcSampleFrameInterval(tgtFps, srcFps=25):
        sf = FrameSampler.calcSampleFrameId(tgtFps, srcFps)
        sf = np.append(sf, sf[0] + srcFps)
        return np.array([ sf[i+1] - sf[i] for i in range(tgtFps) ])
    
    @staticmethod
    def calcSampleTimeInterval(tgtFps, srcFps=25, srcTime=None):
        assert srcTime is None or len(srcTime) == srcFps
        if srcTime is not None:
            assert np.abs(sum(srcTime) - 1.0) < 1e9
        sv = FrameSampler.calcSampleFrameInterval(tgtFps, srcFps)
        if srcTime is None:
            t0 = 1.0/srcFps
            return sv*t0
        svc = np.cumsum(sv)
        res=np.zeros(tgtFps)
        last=0
        for i in range(tgtFps):
            res[i] = sum(srcTime[last:svc[i]])
            last=svc[i]
        return res

    def getSampleFrameId(self):
        if self.sfx is None:
            self.sfx = FrameSampler.calcSampleFrameId(self.tgtFps, self.srcFps)
        return self.sfx
    
    def getSampleFrameInterval(self):
        if self.sfi is None:
            self.sfi = FrameSampler.calcSampleFrameInterval(self.tgtFps, self.srcFps)
        return self.sfi
    
    def getSampleTimeInterval(self):
        if self.sti is None:
            self.sti = FrameSampler.calcSampleTimeInterval(
                    self.tgtFps, self.srcFps, self.srcTime)
        return self.sti
    
    def calcConvertedNumFrame(self, nf):
        m = nf // self.srcFps
        r = nf % self.srcFps
        if r == 0:
            rt = 0
        else:
            sfx = self.getSampleFrameId()
            rt = sum(sfx<r) - 1
        return m*self.tgtFps + rt

# -------- part 2: convert fps --------

def convertFPS(kpm, ptm, tgtFps, srcFps=25, offset=0,
               method='keep', refConf=0, alpha=0.8):
    '''
    Convert OKS and SPF to another frame-rate by sampling frames
    input:
        <kpm> key-point-matrix (4d or 3d: (conf)-frame-kp-xyv)
        <ptm> processing-time-matrix (2d or 1d: (conf)-frame)
        <tgtFps> target frame rate (frame/second)
        <srcFps> the frame rate (frame/second) of input data
        <offset> start from the <offset>-th frame
        <method> method for estimating pose of the unprocessed frames.
            including: keep-old-pose, linear, exponential average
        <refConf> (for keep method) the configuration used as OKS reference
        <alpha> (for eam method) the factor for keeping lastest value
    output:
        <roks> 2d (conf-frame) OKS matrix for the target frame rate (average oks)
        <rptm> 2d (conf-frame) PT matrix for the target frame rate (sampled frame)
    '''
    assert 0 < tgtFps < srcFps
    assert kpm.shape[-2:] == (17,3)
    if kpm.ndim == 3 and ptm.ndim == 1:
        kpm = kpm[np.newaxis, :]
        ptm = ptm[np.newaxis, :]
    assert kpm.ndim == 4 and ptm.ndim == 2
    assert kpm.shape[:2] == ptm.shape
    assert method in ['keep', 'linear', 'eam']
    sampler = FrameSampler(tgtFps, srcFps)
    sfi = sampler.getSampleFrameInterval()
    n, ms = ptm.shape
    mt = sampler.calcConvertedNumFrame(ms - offset)
    roks = np.zeros((n, mt))
    rptm = np.zeros((n, mt))
    if method != 'keep':
        kpmFill = utilPose.fillUnseen(kpm, 'linear')
        last = kpmFill[:,0,:,0:2]
    if method == 'eam':
        speedOld = np.zeros_like(last)
        assert 0 <= alpha <= 1
    fused = offset
    for i in range(mt):
        span = sfi[i % tgtFps]
        ftouse = range(fused, fused+span)
        pose=kpm[:,ftouse,:,:]
        if method == 'keep':
            oksm = utilPose.computeOKS_mat(kpm[refConf,fused], pose)
        else: # linear and eam
            #speed = diff[:,ftouse,:,:].mean(1)
            speed = (kpm[:,fused,:,0:2] - last) / sfi[(i-1) % tgtFps]
            last = kpm[:,fused,:,0:2]
            if method == 'eam':
                speed = speed*alpha + speedOld*(1-alpha)
                speedOld = speed
            if span > 1:
                delta = np.stack([speed*i for i in range(1,span)], axis=1)
                pose[:,1:,:,[0,1]] += delta
            oksm = utilPose.computeOKS_pairMat(kpm[:,ftouse], pose)
        roks[:,i] = oksm.mean(1)
        rptm[:,i] = ptm[:,i]
        fused += span
    return roks, rptm


# -------- part 3: unify to time unit to second--------


def mergeData(oksm, ptm, unit):
    '''
    Unify to second. If the last second is not complete, it is cut off.
    Input:
        <oksm> object keypoint similarity matrix (2d or 1d: (conf)-frame)
        <ptm> processing time matrix (2d or 1d: (conf)-frame)
        <unit> the unit of the input <oksm> and <ptm>
    '''
    assert oksm.shape == ptm.shape
    ndim = oksm.ndim
    assert ndim == 1 or ndim == 2
    assert 0 < unit
    n = oksm.shape[-1]
    if n % unit !=0:
        m = n // unit
        n2 = m*unit
        if ndim == 1:
            oksm=oksm[:n2]
            ptm=ptm[:n2]
        else:
            oksm=oksm[:,:n2]
            ptm=ptm[:,:n2]
    if ndim == 1:
        resOks = np.mean(oksm.reshape(-1, unit), 1)
        resPtm = np.sum(ptm.reshape(-1, unit), 1)
    else:
        k = ptm.shape[0]
        resOks = np.mean(oksm.reshape(k, -1, unit), 2)
        resPtm = np.sum(ptm.reshape(k, -1, unit), 2)
    return resOks, resPtm

