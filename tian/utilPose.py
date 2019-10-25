# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 02:33:57 2019

@author: yanxi
"""

import numpy as np


__KPT_OKS_SIGMAS__ = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0


'''
Object Keypoint Similarity
input:
	<gts>: ground truth key points. Format: 2d array with shape (17,3) for the (x,y) coordinates of the 17 keypoints.
	<dts>: destination key points. Format: the same as <gts>.
'''
def computeOKS_mat(gts, dts, sigmas = None):
	assert isinstance(gts, np.array) and gts.shape == (17,3)
	assert isinstance(dts, np.array) and dts.shape == (17,3)
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
