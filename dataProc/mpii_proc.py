#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:33:12 2019

@author: fubao
"""

# MPII dataset preprocess


# https://github.com/sausax/pose_estimation/tree/master/mpii/annotation


# https://stackoverflow.com/questions/51748087/training-a-keras-classifier-on-mpii-human-pose-dataset

# use a multiple video from MPII

import scipy.io as sio

dataDir='/media/fubao/TOSHIBAEXT/data_pose_estimation/MPII/'    


batchVideosId = 1          # # batch 1

def getVideoImageAnnotation():
    '''
    parse the video image annotation from each batch
    '''
    
    matPath = dataDir + 'mpii_human_pose_v1_u12_2_annotation/' + 'mpii_human_pose_v1_u12_1.mat'
    #print ("matph: ", matPath)
    mat = sio.loadmat(matPath, struct_as_record=False) # add here

    
    #print ("mat: ", mat)
    
    release = mat['RELEASE']
    object1 = release[0, 0]
    print(object1._fieldnames)       #   ['annolist', 'img_train', 'version', 'single_person', 'act', 'video_list']
    annolist = object1.__dict__['annolist']
    print(annolist, type(annolist), annolist.shape)        # (1, 24987) 24987 item
    
    anno1 = annolist[0, 0] 
    print(anno1._fieldnames)         #  ['image', 'annorect', 'frame_sec', 'vididx']

    fileName = anno1.__dict__['image']
    print ("anno1: ", fileName, type(fileName), fileName.shape, fileName[0, 0].__dict__['name'])    

    annorect = anno1.__dict__['annorect']
    print(annorect, type(annorect), annorect.shape)
    anno2 = annorect[0,0]
    print(anno2._fieldnames)          # ['scale', 'objpos']

    objpos = anno2.__dict__['objpos']

    objpos1 = objpos[0,0]
    print(objpos1._fieldnames)     #['x', 'y']
    y = objpos1.__dict__['y']
    print(y, type(y), y.shape)

if __name__== "__main__": 
    getVideoImageAnnotation()