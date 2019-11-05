# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 02:33:57 2019

@author: yanxi
"""

import numpy as np


# -------- part 1: pose fill --------

def fillUnseen(kpm, method='same'):
    assert kpm.dim == 4
    assert kpm.shape[-2:] == (17,3)
    assert method in ['same', 'estimate']
    n, m = kpm.shape[:-2]
    res = kpm.copy()
    if method == 'same':
        for i in range(17):
            nz = kpm.nonzero(kpm[:,:,i,2] - 2)
            for x,y in zip(nz[0],nz[1]):
                p=y-1
                while p >= 0 and res[x,p,i,2] == 0:
                    p-=1
                if p >= 0:
                    res[x,y,i,:2] = res[x,p,i,:2]
    else:
        for i in range(17):
            nz = kpm.nonzero(kpm[:,:,i,2] - 2)
            for x,y in zip(nz[0],nz[1]):
                p=y-1
                while p >= 0 and res[x,p,i,2] == 0:
                    p-=1
                q=p-1
                while q >= 0 and res[x,q,i,2] == 0:
                    q-=1
                if q >= 0:
                    s = res[x,p,i,:2] - res[x,q,i,:2]
                    res[x,y,i,:2] = res[x,p,i,:2] + s / (p-q) * (y-p)
    return res

# -------- part 2: OKS --------

__KPT_OKS_SIGMAS__ = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
__NUM_KPT__ = 17

'''
Object Keypoint Similarity
http://cocodataset.org/#keypoints-eval
OKS = Σi[exp(-di2/2s2κi2)δ(vi>0)] / Σi[δ(vi>0)]
https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
input:
    <gts>: ground truth key points. Format: 2d array with shape (17,3) for the (x,y) coordinates of the 17 keypoints.
    <dts>: destination key points. Format: the same as <gts>.
output:
    a scalar of oks
'''
def computeOKS_pair(gts, dts, sigmas = None):
    assert isinstance(gts, np.ndarray) and gts.shape == (17,3)
    assert isinstance(dts, np.ndarray) and dts.shape == (17,3)
    sigmas = np.array(__KPT_OKS_SIGMAS__ if sigmas is None else sigmas)
    assert sigmas.shape == (17,)
    k=len(sigmas)
    vars = (sigmas * 2)**2

    xg = gts[:,0]
    yg = gts[:,1]
    vg = gts[:,2]
    k1 = np.count_nonzero(vg > 0)

    xmin = xg.min(); xmax = xg.max(); xdif = xmax - xmin;
    ymin = yg.min(); ymax = yg.max(); ydif = ymax - ymin;
    area = (xmax - xmin)*(ymax - ymin)
    
    xd = dts[:,0]
    yd = dts[:,1]
    #vd = np.zeros_like(dg) + 2
    #k2 = np.count_nonzero(vd > 0)

    if k1>0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg
    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        #bb = gt['bbox']
        x0 = xmin - xdif; x1 = xmax + xdif;
        y0 = ymin - ydif; y1 = ymax + ydif;
        z = np.zeros((k))
        dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
        dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
    e = (dx**2 + dy**2) / vars / (area+np.spacing(1)) / 2
    if k1 > 0:
        e=e[vg > 0]
    return np.sum(np.exp(-e)) / e.shape[0]


'''
return a list of OKS for each dts in <dtsList>
'''
def computeOKS_list(gts, dtsList, sigmas = None):
    assert isinstance(gts, np.ndarray) and gts.shape == (17,3)
    assert isinstance(dtsList[0], np.ndarray) and dtsList[0].shape == (17,3)
    sigmas = np.array(__KPT_OKS_SIGMAS__ if sigmas is None else sigmas)
    assert sigmas.shape == (17,)
    k=len(sigmas)
    vars = (sigmas * 2)**2

    xg = gts[:,0]
    yg = gts[:,1]
    vg = gts[:,2]
    k1 = np.count_nonzero(vg > 0)

    xmin = xg.min(); xmax = xg.max(); xdif = xmax - xmin;
    ymin = yg.min(); ymax = yg.max(); ydif = ymax - ymin;
    area = (xmax - xmin)*(ymax - ymin)
    
    n = dtsList.shape[0] if isinstance(dtsList, np.ndarray) else len(dtsList)
    res=np.zeros(n)
    # normal case
    if k1>0:
        for i in range(n):
            dts = dtsList[i]
            xd = dts[:,0]
            yd = dts[:,1]
            dx = xd - xg
            dy = yd - yg
            e = (dx**2 + dy**2) / vars / (area+np.spacing(1)) / 2
            e = e[vg > 0]
            res[i] = np.sum(np.exp(-e)) / e.shape[0]
    else:
        x0 = xmin - xdif; x1 = xmax + xdif;
        y0 = ymin - ydif; y1 = ymax + ydif;
        z = np.zeros((k))
        for i in range(n):
            dts = dtsList[i]
            xd = dts[:,0]
            yd = dts[:,1]
            dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
            dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
            e = (dx**2 + dy**2) / vars / (area+np.spacing(1)) / 2
            res[i] = np.sum(np.exp(-e)) / k
    return res


def computeOKS_mat(gts, dtsMat, sigmas = None):
    assert isinstance(gts, np.ndarray) and gts.shape == (17,3)
    assert isinstance(dtsMat, np.ndarray) and dtsMat.shape[-2:] == (17,3)
    
    matShape = dtsMat.shape[:-2]
    oksl = computeOKS_list(gts, dtsMat.reshape(-1,17,3))
    return oksl.reshape(matShape)