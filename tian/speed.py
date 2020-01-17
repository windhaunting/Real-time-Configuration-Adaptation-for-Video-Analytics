# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:55:01 2019

@author: yanxi
"""

import numpy as np

try:
    from . import utilPose
except:
    import utilPose

# -------- part 1 conversion --------

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

def cart2speed(data):
    s = data.shape
    assert s[-1] == 2
    t = data.reshape(-1,2)
    res = np.sqrt(t[:,0]**2 + t[:,1]**2)
    r = s[:-1]
    return res.reshape(r)

# -------- part 2 estimator --------

class SpeedEstimator:
    def __init__(self, method, **param):
        method=method.lower()
        assert method in ['keep', 'linear', 'ema']
        self.poseLast = np.zeros(17,3)
        self.itvLast =  np.zeros(17,1)
        self.speed= np.zeros(17,2)
        if method == 'keep':
            self.speedFun = self.__update_keep__
        elif method == 'linear':
            self.speedFun = self.__update_linear__
        else: # ema
            self.alpha = param['alpha']
            self.speedLast = np.zeros(17,2)
            self.speedFun = self.__update_ema__
    
    def __update_keep__(self, pose, itv):
        self.poseLast = pose
        return pose
    
    def __update_linear__(self, pose, itv):
        pdiff = (pose - self.poseLast)
        idiff = (itv - self.itvLast)
        flag = pose[:,2] == 2
        self.poseLast[flag] = pose[flag]
        self.itvLast[flag] = itv
        self.speed[flag] = np.apply_along_axis(lambda x:x/idiff, 0, pdiff)[flag]
        
    def __update_ema__(self, pose, itv):
        pdiff = (pose - self.poseLast)
        idiff = (itv - self.itvLast)
        flag = pose[:,2] == 2
        self.poseLast[flag] = pose[flag]
        self.itvLast[flag] = itv
        speedNew = np.apply_along_axis(lambda x:x/idiff, 0, pdiff)[flag]
        self.speed = self.speedLast*(1.0-self.alpha) + speedNew*self.alpha
        self.speedLast[flag] = speedNew[flag]
    
    def update(self, pose, iterval):
        '''
        Input the new pose and iterval since last pose.
        <pose> is the x-y-v tuple of the 17 key points.
        <iterval> is in number of frames.
        '''
        assert pose.shape == (17,3)
        return self.speedFun(pose, iterval)
        
    def getSpeed(self):
        return self.speed
    
    def getSpeedRatio(self):
        return np.sqrt(np.sum(self.speed**2, 1))
    
    def guessPose(self, iterval):
        assert iterval > 0
        speed = self.getSpeed()
        pose = self.poseLast + speed*iterval
        return pose
    
    
