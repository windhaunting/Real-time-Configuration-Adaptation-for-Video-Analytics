#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:11:53 2019

@author: fubao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:54:58 2019

@author: fubao
"""

# common profiling

import os
import cv2
import re
import logging
import numpy as np

from glob import glob
from sys import stdout
from collections import defaultdict
import pandas as pd


# old file
frameRates = [25, 15, 10, 5, 2, 1]    # test only [25, 10, 5, 2, 1]   # [5]   #          #  [25]    #  [25, 10, 5, 2, 1]    # [30],  [30, 10, 5, 2, 1] 
resoStrLst_OpenPose = ["1120x832", "960x720", "640x480",  "480x352", "320x240"]   # for openPose models [720, 600, 480, 360, 240]   # [240] #     # [240]       # [720, 600, 480, 360, 240]    #   [720]     # [720, 600, 480, 360, 240]  #  [720]    # [720, 600, 480, 360, 240]            #  16: 9
resoStrLst_cpn = ["384x288", "256x192"]   # for cpn models, only two resolutions pretrained available

modelMethods_openPose = ['cmu', 'mobilenet_v2_small']
# a_cpn,   "a" is just to make it alphabetically order first, to make it as ground truth conviniently for programming
modelMethods_cpn = ['a_cpn']  #  'cmu']   # , 'mobilenet_v2_small'] # ['a_cpn']   #     ['a_cpn', 'cmu', 'mobilenet_v2_small']  #  ['mobilenet_v2_small']      # ['mobilenet_thin']  # ['cmu']  #  ["openPose"]



# simulate without buffer to check how many accuracy we can achieve and the lag with the segment number
dataDir1 = '../input_output/mpii_dataset/'
dataDir2 = '../input_output/diy_video_dataset/'

PLAYOUT_RATE = 25


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
            print ("no more frame, exit here 1, total frames ", count-1)
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


def round_int(val):
    return int(round(val))

def transfer_coco_keyPoint_format(humanDict, image_w, image_h):
    
    #transfer to coco format of keypoint
    #https://www.tensorflow.org/lite/models/pose_estimation/overview
    #https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
        
    keypoints = []
    coco_ids = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    for coco_id in coco_ids:
        if coco_id not in humanDict.keys():
            keypoints.extend([0, 0, 0])
            continue
        body_part = humanDict[coco_id]
        keypoints.extend([round_int(body_part[0] * image_w), round_int(body_part[1]* image_h), 2])
    return keypoints
    

def oneHumanToDicthuman(human, w, h):
    
    #one human ; transfer to dictionary of body parts, score
    #BodyPart:0-(0.77, 0.25) score=0.80 BodyPart:1-(0.79, 0.28) score=0.75 BodyPart:2-(0.77, 0.29) score=0.58...

    
    humBodyPartsDict = defaultdict()
    bodyPartsLst = human.split(' Body')
    
    avrScore = 0.0
    #print ("bodyPartsLst: ", human)
    for i, bdScore in enumerate(bodyPartsLst):
        
        # split
        bdStr = bdScore.split('score=')[0]
        # get index
        #print ("bdStr: ", bdScore, bdStr)
        k = int(bdStr.split(':')[1].split('-')[0])   # body part id
        
        v = bdStr.split(':')[1].split('-')[1]  # str
        humBodyPartsDict[k] = (float(v.split(',')[0].replace('(', '')), float(v.split(',')[1].replace(')', '')))
        
        #print ("bdScore:", bdScore)
        sc = bdScore.split('score=')[1]
        avrScore += float(sc)
    humBodyPartsDict['score'] = round(avrScore/len(bodyPartsLst), 3)
    
    # calculate the bounding box, estimate
    minX = 2**32
    maxX = -2**32
    minY = 2**32
    maxY = -2**32
    for k, v in humBodyPartsDict.items():
        if k != 'score':
            if v[0] < minX:
                minX = v[0]
            if v[0] > maxX:
                maxX = v[0]
            if v[1] < minY:
                minY = v[1]
            if v[1] > maxY:
                maxY = v[1]
    
    humBodyPartsDict['bbox'] = [minX*w, minY*h, (maxX-minX)*w, (maxY-minY)*h]
    #print ("humBodyPartsDict bbox:", humBodyPartsDict['bbox'])
    humBodyPartsDict['area'] = (maxX-minX)*w*(maxY-minY)*h
    return humBodyPartsDict
      
def humanBodiesToLstDict(est_result, w, h):

    #multiple humans into a list of dictionary
    #image width, height

    # extract humans
    est_result = est_result.split('\t')[-1].replace('[', '').replace(']', '')    # preprocess
    #print ("est_result: ", est_result)
    
    humansLst = []
    begin = 0
    for m in re.finditer(', BodyPart', est_result):
        #print(' found', m.start(), m.end())
    
        human = est_result[begin:m.start()].strip()
        
        # human transfer to dictionary
        humansLst.append(oneHumanToDicthuman(human, w, h))
        
        begin = m.start()+1
        
    human = est_result[begin::].strip()
    humansLst.append(oneHumanToDicthuman(human, w, h))
    
    #print ("humansLst: ", humansLst)
    
    return humansLst


def computeOKSAP(est_result, gt_result, img_path, w, h):
    #calculate OKS and AP as accuracy with different threshold[.5:.05:.95]
    est_lst = humanBodiesToLstDict(est_result, w, h)
   
    dts = []
    det_scores = 0.0
    for human in est_lst:
        item = {
            'image_id': img_path,
            'category_id': 1,
            'keypoints': transfer_coco_keyPoint_format(human, w, h),
            'score': human['score']
        }
        dts.append(item)
        det_scores += item['score']
        
    det_avg_score = det_scores / len(est_lst) if len(est_lst) > 0 else 0
    
    
    gt_lst = humanBodiesToLstDict(gt_result, w, h)

    gts = []
    gt_scores = 0.0
    for human in gt_lst:
        item = {
            'image_id': img_path,
            'category_id': 1,
            'keypoints': transfer_coco_keyPoint_format(human, w, h),
            'score': human['score'],
            'bbox': human['bbox'],
            'area': human['area']
        }
        gts.append(item)
        gt_scores += item['score']
        
    gt_avg_score = gt_scores / len(gt_lst) if len(gt_lst) > 0 else 0
    
    
    # compute oks
    #print ("len(gts), dts:", len(gts), len(dts))
    ious = computeOKS(gts, dts)      # ((len(dts), len(gts)))
 
    #print ("ious:", ious)
    
    ious = np.transpose(ious)         # transpose
    
    thresholds = np.arange(0.5, 0.95, 0.05)
    aver_prec = 0.0
    aver_acc = 0.0
    for thres in thresholds:
        # calculate the AP as the accuracy
        TP = 0
        FP = 0
        FN = 0
        for m in range(0, len(ious)):
            eachGT_Detected = 0
            for n in range(0, len(ious[m])):
                if ious[m][n] > thres:
                    if eachGT_Detected >= 1:
                        FP += 1
                    TP += 1
                    eachGT_Detected += 1
            if eachGT_Detected == 0:
                FN += 1
        if (TP+FP) == 0:
            prec = 0
        else:
            prec = TP/(TP+FP)     #len(gts)
        if (TP+FN) == 0:
            recall = 0
        else:
            recall = TP/(TP+FN)
            
        if (prec+recall) == 0:
            acc = 0
        else:
            acc = 2*prec*recall/(prec+recall)
        
        #print ("precision:", prec, recall, acc)
        aver_prec += prec
        aver_acc += acc
    return aver_prec/thresholds.size


def computeOKS(gts, dts):
       
    kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    maxDets = [20]

    # dimention here should be Nxm
    inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
    dts = [dts[i] for i in inds]
    if len(dts) > maxDets[-1]:
        dts = dts[0:maxDets[-1]]
    # if len(gts) == 0 and len(dts) == 0:
    if len(gts) == 0 or len(dts) == 0:
        return []
    ious = np.zeros((len(dts), len(gts)))
    sigmas = kpt_oks_sigmas
    vars = (sigmas * 2)**2
    k = len(sigmas)
    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):
        # create bounds for ignore regions(double the gt bbox)
        g = np.array(gt['keypoints'])
        xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
        #print ("computeOKS type vg: ", type(vg), vg)
        k1 = np.count_nonzero(vg > 0)
        bb = gt['bbox']
        x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
        y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
        for i, dt in enumerate(dts):
            d = np.array(dt['keypoints'])
            xd = d[0::3]; yd = d[1::3]
            if k1>0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((k))
                dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
            e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
            if k1 > 0:
                e=e[vg > 0]
            ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
    return ious



def executeVideoToFrames():
    
    '''
    inputVideoPath = dataDir2 + "006-cardio_condition-20mins.mp4"
    
    outDir = dataDir2 + '006-cardio_condition-20mins_output_frames/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    extractVideoFrames(inputVideoPath, outDir)
    '''
    
    
    inputDir = "/media/fubao/TOSHIBAEXT/research_bakup/data_poseEstimation/diy_video_dataset/"
    filePathLst = sorted(glob(inputDir + "*.mp4"))[7:9]          # [5:6]
    
    outParentDir = "/media/fubao/TOSHIBAEXT/research_bakup/data_poseEstimation/diy_video_dataset/"
    for filePath in filePathLst:
        
        outDir =  outParentDir + filePath.split("/")[-1].split(".")[0] + "_frames/"      # "/media/fubao/TOSHIBAEXT/research_bakup/data_poseEstimation/2-soccer-20mins-frames/"
        if not os.path.exists(outDir):
            os.mkdir(outDir)
        #print ("filePath: ", filePath, outDir)
        extractVideoFrames(filePath, outDir)
    
    
if __name__== "__main__":
    executeVideoToFrames()

    
    
    