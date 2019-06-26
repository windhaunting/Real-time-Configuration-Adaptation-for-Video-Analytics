#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 11:58:31 2019

@author: fubao
"""


# tensorflow open_pose estimation model

import argparse
import logging
import sys
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'     # Just disables the warning, doesn't enable AVX/FMA if you have GPU

currentDir = '../poseEstimation/tf_pose_estimation/'
sys.path.insert(0, currentDir)

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.WARNING)      # ogging.DEBUG
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)           # ogging.DEBUG
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def load_model(model, reso):
    #print (" tf_open_pose_inference begin :", test_image, reso, model)
    
    w, h = model_wh(reso)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

   
    return e, w, h


def tf_open_pose_inference(test_image, reso, e, w, h):
    '''
    set the interface for pose estimation 
    '''
    #print (" tf_open_pose_inference begin :", test_image, reso)
    resize_out_ratio = 4.0
    
     # estimate human poses from a single image !
    if not common.verify_image(test_image):
        # delete the image
        os.remove(test_image)
        return None, None
    image = common.read_imgfile(test_image, w, h)
    if image is None:
        logger.error('Image can not be read, path=%s' % test_image)
        #sys.exit(-1)
        return None, None
    
    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
    elapsed = time.time() - t

    #logger.warning('inference image: %s in %.4f seconds and res %s' % (test_image, elapsed, reso))
    #logger.warning('output human: type %s and %s' % (type(humans), type(humans[0])))
    
    #print (" test result :", test_image, elapsed, reso)
    
    return humans,  elapsed
    
    

if __name__ == '__main__':
    x = 1
