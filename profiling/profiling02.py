#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:53:39 2019

@author: fubao
"""

# profiling  02

# every other 1 minute to get 
# for frame rate less than maximum frame rate, use current extracted frame to replace the rest of frames unextracted


# get a profiling result and store into csv with dataframe
import sys 
import os 
import re
import pandas as pd
import numpy as np

from glob import glob
from collections import defaultdict
from common_prof import computeOKSAP

#sys.path.insert(0, '../openPoseEstimation')
sys.path.insert(0,'..')

from poseEstimation.pytorch_Realtime_Multi_Person_Pose_Estimation.openPose_interface import openPose_estimation_one_image

from poseEstimation.tf_pose_estimation.tf_openPose_interface import tf_open_pose_inference
from poseEstimation.tf_pose_estimation.tf_openPose_interface import load_model

dataDir1 = '../input_output/mpii_dataset/'

dataDir2 = '../input_output/diy_video_dataset/'

PLAYOUT_RATE = 25


# define a class including each clip's profile result
class cls_profile_video(object):
    # cameraNo is the camera no. for multiple camera streaming. 
    # queryNo is the query type, currently human pose estimation
    # 'frameStartNo', 
    __slots__ = ['cameraNo', 'queryNo', 'resolution', 'frameRate','modelMethod', 'accuracy', 'costFPS']


frameRates = [25, 15, 10, 5, 2, 1]    # test only [25, 10, 5, 2, 1]   # [5]   #          #  [25]    #  [25, 10, 5, 2, 1]    # [30],  [30, 10, 5, 2, 1] 
resoStrLst= ["1120x832", "960x720", "640x480",  "480x352", "320x240"]   #  [720, 600, 480, 360, 240]   # [240] #     # [240]       # [720, 600, 480, 360, 240]    #   [720]     # [720, 600, 480, 360, 240]  #  [720]    # [720, 600, 480, 360, 240]            #  16: 9
resolutions = [(int(res.split("x")[0]), int(res.split("x")[1])) for res in resoStrLst]
modelMethods = ['cmu', 'mobilenet_v2_small']  #  ['mobilenet_v2_small']      # ['mobilenet_thin']  # ['cmu']  #  ["openPose"]



def profilingOneVideoWithMaxFrameRate(inputDir, outDir):
    '''
    for a video with 14 mintues only for each video, not online detection

    use each config to read each segment and alculate the accuracy, average detection speed(SPF) for 
    for each frame
    
    first we get different resolution with frame rates 25's result of each frame and store into csv file
    
    #Configuration_C_index, Frame_rate, Resolution, Model, image_Id, category_Id,  detection_keyPoint, scores

    '''
        
    #get a video length 1 mins 
    #video_time_len = 60
    #total_frames_len = video_time_len * PLAYOUT_RATE         # 5 is test value only   video_time_len * PLAYOUT_RATE         # 60*25 = 1500
    
    #segmentNum = total_frames_len//segment_frames        # 
        
    #every one minute to draw a config out
    #plotIntervals = [120*PLAYOUT_RATE, 240*PLAYOUT_RATE, 360*PLAYOUT_RATE, 480*PLAYOUT_RATE, 600*PLAYOUT_RATE, 720*PLAYOUT_RATE, 840*PLAYOUT_RATE]
    configIndex = 0
    for res in resoStrLst:    # resolutions
        fr = frameRates[0]        # use only maximum frame rate is enough
        for mod in modelMethods:
            #'resolution', 'frameRate','modelMethod',
            
            configIndex += 1
            # each file is a config of  the video
            out_file = outDir + str(res)+ '_' + str(fr) + '_' + str(mod) + '_estimation_result.tsv'
                
            headStr = 'Configuration_C_index' + '\t' + 'Resolution' +'\t' + 'Frame_rate' +'\t' + 'Model' +   \
             '\t' + 'Image_path' +'\t' + 'Estimation_result' + '\t' + 'numberOfHumans' + '\t'+ 'Time_SPF' + '\n'  # '\t' + 'image_Id'+ '\t' + 'category_Id' +'\t' + 'detection_keyPoint' + '\t' + 'scores' + '\n'
            

            # load model
            e, w, h = load_model(mod, res)
            
            with open(out_file, 'a') as f:
                f.write(headStr)

                frmCnt = 1        # fram step size
                
                filePathLst = sorted(glob(inputDir + "*.jpg"))
                #print ("filePath: ", filePath)
                for imgPath in filePathLst:      # iterate each subdirect's frames and compose a video
                    #if frmCnt >= total_frames_len:   # until 1min
                    #    print ("frmCnt: ", frmCnt, imgPath)
                    #    break
                    
                
                    # apply different configuration
                    # estimate pose of this frame  call the open_pose etc. interface
                    preWriteStr = str(configIndex) + '\t' + str(res) +'\t' + str(fr) +'\t' + \
                    str(mod) + '\t'
                    #if mod == "cmu": # "openPose":
                    #output_lst, elapsedTime = openPose_estimation_one_image(imgPath, res) #mulitple person
                    output_lst, elapsedTime = tf_open_pose_inference(imgPath, mod, res, e, w, h)
                    
                    if frmCnt % (25*4) == 0:        # every other 4s to print only
                        print ("profilingOneVideoWithMaxFrameRate inference result: ", imgPath, mod, res, elapsedTime)
                    #save into file
                    if output_lst is None or len(output_lst) == 0:
                        continue
                    
                    #print ("output_lst: ", type(output_lst))
                    #for outDic in output_lst:
                    #    print ("outDic:" , type(outDic), outDic)
                    preWriteStr += imgPath + '\t' + str(output_lst)  + '\t'  + str(len(output_lst)) + '\t' + str(elapsedTime) + '\n'      # str(outDic['image_Id']) + '\t' + str(outDic['category_Id']) + '\t' + ','.join(outDic['detection_keyPoint']) + '\t' + str(outDic['scores']) + "\n"
    
                    f.write(preWriteStr)
    
                    frmCnt += 1
                    



def getEachSegmentProfilingAPTime(inputDir, segment_time,  outDir):
    '''
    get each frames's avearge precision(AP) and detection time
    
    read each file's detection result of each config, compare with the most expensive configuration and
    calculate the AP and time

    outProfFile:
        ['streamingNo', 'imagePath', 'resolution', 'frameRate','modelMethod', 'accuracy', 'costFPS']

    '''
    
    segment_frames = PLAYOUT_RATE*segment_time  # 25*4 = 100   
    profilingTime = segment_time//4       
    profiling_frames = int(PLAYOUT_RATE*profilingTime)    # frames in the profiling time
    

    gtDic = defaultdict(list)        # ground truth detection file for each frame
    
    
        
    out_file = outDir + 'test_profiling_segment_time' + str(segment_time)+ '.tsv'
    
    headStr = 'Resolution' +'\t' + 'Frame_rate' +'\t' + 'Model' +   \
             '\t' + 'Segment_no' +'\t' + 'Detection_speed_FPS'  + '\t' + 'Acc' +'\n'   # '\t' + 'image_Id'+ '\t' + 'category_Id' +'\t' + 'detection_keyPoint' + '\t' + 'scores' + '\n'
             
        
    print ("inputDir: ", inputDir)
    filePathLst = sorted(glob(inputDir + "*.tsv"))  # must read ground truth file(the most expensive config) first
    with open(out_file, 'w') as f:
        f.write(headStr)
        
        for fileCnt, filePath in enumerate(filePathLst):
           # read poste estimation detection result file
           
            df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)  # det-> detection
            print ("filePath: ", fileCnt, filePath, df_det.columns)
                       

            framesTotalNum = df_det.shape[0]
    
            segment_no = 1
            startFrmCnt = 1
            endFrmCnt = segment_frames
    
            #config_index = df_det.iat[0, 0]
            resolution = df_det.iat[1, 1]
            #print ("resolution: ", type(resolution), resolution)
            w = int(resolution.split('x')[0])
            h = int(resolution.split('x')[1])

            model = df_det.iat[0, 3]
                            
            if fileCnt == 0:      # '1120x832_25_cmu' in fileName:  # this file has the most expensive config result
                # get the ground truth for each image_path
                # pose estimation result is ground truth, accuracy is 1, add a new column "Acc"
                # get ground truth dictionary with estimation result
                gtDic = dict(zip(df_det.Image_path, df_det.Estimation_result))
                #df_det['Acc'] = 1             
            
            while (startFrmCnt < framesTotalNum):

                for frmRate in frameRates:
                    profileFrmInter = int(PLAYOUT_RATE//frmRate)
                    
                    if frmRate == frameRates[0]:     # PLAYOUT_RATE:  # ground truth
                        gtDic = dict(zip(df_det.Image_path, df_det.Estimation_result))
                        #df_det['Acc'] = 1 
                        #print ("profileFrmInter: ",profiling_frames, profileFrmInter, type( df_prof.iloc[startFrmCnt:profiling_frames]))
                        #get det_speed in the profiling time
                    
                    # no else: also calculate the segment's accuracy for ground truth scenario
                    #print ("det_speed_seg: ", df_det.iloc[startFrmCnt:startFrmCnt+profiling_frames].iloc[::profileFrmInter, 7].astype(float).values)
                    #calculate  detection speed for this segment
                    det_speed_seg= sum(df_det.iloc[startFrmCnt:startFrmCnt+profiling_frames].iloc[::profileFrmInter, 7].astype(float).values)/PLAYOUT_RATE
               
                    det_speed_seg = round(1/det_speed_seg, 2)
                    # accuracy use the extracted frame estimation result for the result of frame, tha a little complicated,
                    #df_extracted_det_est = df_prof.iloc[startFrmCnt:startFrmCnt+profiling_frames].iloc[::profileFrmInter, 4, 5]
                    df_extracted_det_est = df_det.iloc[startFrmCnt:startFrmCnt+profiling_frames]   # Image_path and Estimation_result
                    
                    iterFrmRateIndx = 0
                    acc_seg = 0.0
                    for indx, row in df_extracted_det_est.iterrows():
                        #print ("getEachSegmentAPTime row: ", indx, type(row), row['Estimation_result'])
                        #get the estimation_result for this frame
                        img = row['Image_path']
                        gt_result_curr_frm = gtDic[img]
                        
                        if img not in gtDic:
                            acc_seg += 0
                            print ("img not found in ground truth")
                            continue
            
                        if iterFrmRateIndx % profileFrmInter == 0:
                            lead_frame_est_res  = row['Estimation_result']
                            #get accuracy
                            acc_curr_frm = computeOKSAP(lead_frame_est_res, gt_result_curr_frm, img, w, h)
                            acc_seg += acc_curr_frm
                        else:
                            curr_frm_est_res = lead_frame_est_res
                            acc_curr_frm = computeOKSAP(curr_frm_est_res, gt_result_curr_frm, img, w, h)
                            acc_seg += acc_curr_frm
                            
                        iterFrmRateIndx += 1
                    acc_seg /= profiling_frames
                    acc_seg = round(acc_seg, 3)
                    #print ("iterFrmRateIndx: ", iterFrmRateIndx)
                    #break       # test only
                
                    rowStr =  str(resolution) + '\t' + str(frmRate) + '\t' + str(model) +   \
                        '\t'  + str(segment_no) + '\t' + str(det_speed_seg) + '\t' + str(acc_seg) + '\n'
                    f.write(rowStr)
                    
                startFrmCnt = endFrmCnt
                endFrmCnt += segment_frames
                if endFrmCnt > framesTotalNum:   # not enough
                    endFrmCnt = framesTotalNum
                    
                segment_no += 1
                


    df_prof_segment = pd.read_csv(out_file, delimiter='\t', index_col=False)  # det-> detection

    #df_prof_segment['Config_index'] = df_prof_segment.index
    
    # get each segment and store into file
    
    #df_prof_segment.to_csv(out_file, sep='\t', index=False)


    for seg_no in range(1, segment_no):
        out_seg_no_file = outDir + 'profiling_segmentTime' + str(segment_time)+ '_segNo' + str(seg_no) + '.tsv'
        df_seg_no = df_prof_segment.loc[df_prof_segment['Segment_no'] == seg_no]
        print ("df_seg_no shape: ", df_seg_no.shape, type(df_seg_no))
        df_seg_no.insert(0, column="Config_index", value = range(1, len(df_seg_no) + 1))
        df_seg_no.to_csv(out_seg_no_file, sep='\t', index=False)
           

if __name__== "__main__":
    
    '''
    inputDir = dataDir1 +  'output/'  #  dataDir2 + 'output_video_dancing_01/' 
    outDir = dataDir1 + 'output/profiling_result/'    #  dataDir2+ 'output_video_dancing_01/profiling_result/' 
    segment_time = 4

    getEachSegmentProfilingAPTime(inputDir, segment_time, outDir)
    '''
    
    
    inputDir =  dataDir2 + 'output_video_dancing_01/' 
    outDir = dataDir2+ 'output_video_dancing_01/profiling_result/' 
    segment_time = 4

    getEachSegmentProfilingAPTime(inputDir, segment_time, outDir)
    