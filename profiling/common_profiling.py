#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:54:58 2019

@author: fubao
"""

# common profiling

import os
import cv2
import logging
from sys import stdout
from collections import defaultdict


inputVideoDir = "input_output/video_dataset/"

def readVideo(inputVideoPath):
    '''
    read a video
    '''
    cap = cv2.VideoCapture(inputVideoPath)      # 0 to camera in the file

    return cap


def extractVideoFrames(inputVideoPath, outFramesPath):
    '''
    extracframes from a video
    and save into file or dictionary
    
    outFramesPath: if it's "dict", save into a dictionary
    '''
    
    cap = readVideo(inputVideoPath)
    
    if (not cap.isOpened):
        print ('cam not opened: %s ', cap.isOpened())
        return 

    FPS = cap.get(cv2.CAP_PROP_FPS)
    WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    NUMFRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    print('cam stat: ', FPS, WIDTH, HEIGHT, NUMFRAMES)
    
    count = 1
    imageDict = defaultdict(int)
    while True:
      
        ret, img = cap.read()

          
        if not ret:
            print ("no frame exit here 1, total frames ")
            break
      
        # test resize resolution
        #img = cv2.resize(img, (300, 300));    
        if outFramesPath == "dict":
            imageDict[count] = img
        else:
          
            if count < 10:
                outName = "00000" +str(count)
            elif count < 100:
                outName = "0000" + str(count)
            elif count < 1000:
                outName = "000" + str(count)
            elif count < 10000:
                outName = "00" + str(count)
            elif count < 100000:
                outName = "0" + str(count)
            else:
                outName = str(count)
            cv2.imwrite(outFramesPath + outName + '.jpg', img)     # save frame as JPEG file

        count += 1
  
      #cv2.waitKey( 1000 // 100)
    if outFramesPath == "dict":
        return imageDict
    else:
        return None


if __name__== "__main__":
    inputVideoPath = inputVideoDir + "video_dancing_01.mp4"
    
    outDir = inputVideoDir + 'video_dancing_01_frames/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    extractVideoFrames(inputVideoPath, outDir)
    
    