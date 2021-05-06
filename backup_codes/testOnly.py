#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:05:56 2019

@author: fubao
"""


import cv2
from skimage import io

def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True

def read_imgfile(path, width=None, height=None):
    
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    if val_image is not None and width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image


w = 300
h = 300
test_image =  "/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/videoAnalytics_poseEstimation/input_output/diy_video_dataset/video_dancing_01_frames/011434.jpg"

print("valid: ", verify_image(test_image))
image = read_imgfile(test_image, w, h)
if image is None:
    print('Image can not be read, path=%s' % test_image)
    #sys.exit(-1)
    