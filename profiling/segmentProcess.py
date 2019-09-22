#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:09:07 2019

@author: fubao
"""

'''
after we profiling, we get each config's detection/pose_estimation result for each frame in a video
here we further process to get each segment's accuracy, detection speed compared with the most expensive config (as ground truth)
'''

import os 
import math
import pandas as pd

from glob import glob
from collections import defaultdict

from common_prof import PLAYOUT_RATE
from common_prof import dataDir2

from common_prof import frameRates
from common_prof import computeOKSAP
#from common_prof_02 import computeOKSAP

#from common_prof_01 import computeOKSAP


def getEachFrameProfilingAPTtime(inputDir,  outDir):
    '''
    #get each config, each frame's accuracy and detection speed/frame
    in total each config corresponds a file
    '''
      
    filePathLst = sorted(glob(inputDir + "*estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    for fileCnt, filePath in enumerate(filePathLst):
       # read poste estimation detection result file
       
        df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)         # det-> detection
        print ("getEachSegmentProfilingAPTime filePath: ", fileCnt, filePath, df_det.columns)
                   
        
        #config_index = df_det.iat[0, 0]
        resolution = df_det.iat[1, 1]
        sample_rate = df_det.iat[1, 2]
        model = df_det.iat[1, 3]
        #print ("resolution: ", type(resolution), resolution)
        img_w = int(resolution.split('x')[0])
        img_h = int(resolution.split('x')[1])
            
        if fileCnt == 0:      # 384x288_a_cpn or '1120x832_25_cmu' in fileName:  # this file has the most expensive config result
            # get the ground truth for each image_path
            # pose estimation result is ground truth, accuracy is 1, add a new column "Acc"
            # get ground truth dictionary with estimation result
            gtDic = dict(zip(df_det.Image_path, df_det.Estimation_result))
            #df_det['Acc'] = 1             
            gt_w = img_w             # ground truth image's width
            gt_h = img_h            # ground truth image's heigth 
            #continue                 # ground truth not need to calculate accuracy? or still use it as "1"
        # get the accuracy of each frame
        #acc = 0.0
        list_acc_cur = []
        for indx, row in df_det.iterrows():
            #print ("getEachSegmentAPTime row: ", indx, type(row), row['Estimation_result'])
            #get the estimation_result for this frame
            img = row['Image_path']
                    
            if img not in gtDic:
                acc = 0
                print ("getEachSegmentProfilingAPTime img not found in ground truth", img)
                list_acc_cur.append(acc)
                continue
            gt_result_curr_frm = gtDic[img]
               
            curr_frm_est_res  = row['Estimation_result']
            #print ("getEachSegmentProfilingAPTime curr_frm_est_res", curr_frm_est_res)
            acc = computeOKSAP(curr_frm_est_res, gt_result_curr_frm, img, img_w, img_h, gt_w, gt_h)
            #acc = computeOKSAP(curr_frm_est_res, gt_result_curr_frm, img,  gt_w, gt_h)
            list_acc_cur.append(acc)
        
        #print ("acc: ", len(list_acc_cur), len(df_det), df_det.count(),  df_det.shape)
        
        
        df_det['Acc'] = list_acc_cur
        
        # write into file
        del df_det['Estimation_result']         # delete the column file
        del df_det['numberOfHumans']         # delete the column file
        out_file = outDir + str(resolution) + '_' + str(sample_rate) + '_' + str(model) + '_frame_result.tsv'
        df_det.to_csv(out_file, sep='\t', index=False)
         

def getEachSegmentProfilingAPTime(inputDir, segment_time,  outDir):
    '''
    get each frames's avearge precision(AP) and detection time
    
    read each file's detection result of each config, compare with the most expensive configuration and
    calculate the AP and time

    outProfFile:
        [resolution', 'frameRate','modelMethod', 'segment_no', 'costFPS', 'accuracy']

    '''
    
    segment_frames = PLAYOUT_RATE*segment_time  # 25*4 = 100   
    profilingTime = segment_time//4       
    profiling_frames = int(PLAYOUT_RATE*profilingTime)    # frames in the profiling time
    

    gtDic = defaultdict(list)        # ground truth detection file for each frame
    
    
    out_file = outDir + 'test_profiling_segment_time' + str(segment_time)+ '.tsv'
    
    headStr = 'Resolution' +'\t' + 'Frame_rate' +'\t' + 'Model' +   \
             '\t' + 'Segment_no' +'\t' + 'Detection_speed_FPS'  + '\t' + 'Acc' +'\n'   # '\t' + 'image_Id'+ '\t' + 'category_Id' +'\t' + 'detection_keyPoint' + '\t' + 'scores' + '\n'
             
        
    print ("getEachSegmentProfilingAPTime inputDir: ", inputDir)
    filePathLst = sorted(glob(inputDir + "*.tsv"))  # must read ground truth file(the most expensive config) first
    with open(out_file, 'w') as f:
        f.write(headStr)
        
        #get each config, each image's accuracy
        out_frm_config_dir = outDir + 'frames_config_result/' 
        if not os.path.exists(out_frm_config_dir):
            os.mkdir(out_frm_config_dir)
            
        for fileCnt, filePath in enumerate(filePathLst):
           # read poste estimation detection result file
           
            df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)         # det-> detection
            print ("getEachSegmentProfilingAPTime filePath: ", fileCnt, filePath, df_det.columns)
                       
            framesTotalNum = df_det.shape[0]
            
            segment_no = 1
            startFrmCnt = 1
            endFrmCnt = segment_frames
    
            #config_index = df_det.iat[0, 0]
            resolution = df_det.iat[1, 1]
            #print ("resolution: ", type(resolution), resolution)
            img_w = int(resolution.split('x')[0])
            img_h = int(resolution.split('x')[1])

            model = df_det.iat[0, 3]
            
            if fileCnt == 0:      # 384x288_a_cpn or '1120x832_25_cmu' in fileName:  # this file has the most expensive config result
                # get the ground truth for each image_path
                # pose estimation result is ground truth, accuracy is 1, add a new column "Acc"
                # get ground truth dictionary with estimation result
                gtDic = dict(zip(df_det.Image_path, df_det.Estimation_result))
                #df_det['Acc'] = 1             
                gt_w = img_w             # ground truth image's width
                gt_h = img_h            # ground truth image's heigth 
            while (startFrmCnt < framesTotalNum):

                for frmRate in frameRates:             
                    profileFrmInter = math.ceil(PLAYOUT_RATE/frmRate)+1          # frame rate sampling frames in interval, +1 every other
                    
                    #if frmRate == frameRates[0]:     # PLAYOUT_RATE  # ground truth
                    #   gtDic = dict(zip(df_det.Image_path, df_det.Estimation_result))     # this should be cpn model as ground truth
                        #df_det['Acc'] = 1 
                        #print ("profileFrmInter: ",profiling_frames, profileFrmInter, type( df_prof.iloc[startFrmCnt:profiling_frames]))
                        #get det_speed in the profiling time
                    
                    # no else: also calculate the segment's accuracy for ground truth scenario
                    #print ("det_speed_seg: ", df_det.iloc[startFrmCnt:startFrmCnt+profiling_frames].iloc[::profileFrmInter, 7].astype(float).values)
                    #calculate  detection speed for this segment based on current frame rate
                    det_speed_seg= sum(df_det.iloc[startFrmCnt:startFrmCnt+profiling_frames].iloc[::profileFrmInter, 7].astype(float).values)/PLAYOUT_RATE
               
                    det_speed_seg = round(1/det_speed_seg, 3)
                    
                    #print ("det_speed_seg", resolution,  frmRate, profileFrmInter, PLAYOUT_RATE, det_speed_seg)
                    
                    # accuracy use the extracted frame estimation result for the result of frame, tha a little complicated,
                    #df_extracted_det_est = df_prof.iloc[startFrmCnt:startFrmCnt+profiling_frames].iloc[::profileFrmInter, 4, 5]
                    df_extracted_det_est = df_det.iloc[startFrmCnt:startFrmCnt+profiling_frames]   # Image_path and Estimation_result
                    
                    iterFrmRateIndx = 0
                    acc_seg = 0.0
                    
                    for indx, row in df_extracted_det_est.iterrows():
                        #print ("getEachSegmentAPTime row: ", indx, type(row), row['Estimation_result'])
                        #get the estimation_result for this frame
                        img = row['Image_path']
                        
                        if img not in gtDic:
                            acc_seg += 0
                            print ("getEachSegmentProfilingAPTime img not found in ground truth", img)
                            continue
                        gt_result_curr_frm = gtDic[img]
                        
                        if iterFrmRateIndx % profileFrmInter == 0:
                            lead_frame_est_res  = row['Estimation_result']
                            #get accuracy
                            #print ("filePath222333: ", filePath, img , lead_frame_est_res, gt_result_curr_frm)
                            acc_curr_frm = computeOKSAP(lead_frame_est_res, gt_result_curr_frm, img)
                            #acc_curr_frm = computeOKSAP(lead_frame_est_res, gt_result_curr_frm, img, gt_w, gt_h)

                            acc_seg += acc_curr_frm
                        else:
                            curr_frm_est_res = lead_frame_est_res
                            acc_curr_frm = computeOKSAP(curr_frm_est_res, gt_result_curr_frm, img)
                            #acc_curr_frm = computeOKSAP(lead_frame_est_res, gt_result_curr_frm, img, gt_w, gt_h)
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
        #print ("df_seg_no shape: ", df_seg_no.shape, type(df_seg_no))
        df_seg_no.insert(0, column="Config_index", value = range(1, len(df_seg_no) + 1))
        df_seg_no.to_csv(out_seg_no_file, sep='\t', index=False)
        
        


def execute_frame_performance():
    '''
    after profiling
    calculate each frame's performance accuracy, detection speed for each config
    
    '''
    #lst_input_video_frms_dir = ['001-dancing_10mins_frames/', '002-soccer-20mins-frames/', \
    #                    '003-bike_race-20mins_frames/', '004-Marathon-20mins_frames/',  \
    #                    '006-cardio_condition-20mins_frames/', ]
    
    lst_input_video_frms_dir = ['001-dancing-10mins_frames/',  '002-video_soccer-20mins_frames/',
                                '003-bike_race-20mins_frames/', '006-cardio_condition-20mins_frames/',
                                '008-Marathon-20mins_frames/']
      
    
    for input_frm_dir in lst_input_video_frms_dir[0:1]:  # [4::]:       # run 006 first
        
        out_dir = dataDir2 + 'output_' + '_'.join(input_frm_dir.split('_')[:-1]) +'/'      # 004-output_Marathon-20mins_01/' 
        #transfer to accuracy and detection speed in each segment
    
        input_dir = out_dir          # out_dir is created from profiling API

        out_dir = input_dir + 'frames_config_result/' 
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        getEachFrameProfilingAPTtime(input_dir, out_dir)
        
        
def execute_segment_performance(segment_time):
    '''
    after profiling
    calculate each segment's performance accuracy, detection speed
    
    '''
    #lst_input_video_frms_dir = ['001-dancing_10mins_frames/', '002-soccer-20mins-frames/', \
    #                    '003-bike_race-20mins_frames/', '004-Marathon-20mins_frames/',  \
    #                    '006-cardio_condition-20mins_frames/', ]
    
    lst_input_video_frms_dir = ['006-cardio_condition-20mins_frames/']
      
    for input_frm_dir in lst_input_video_frms_dir[0:1]:  # [4::]:       # run 006 first
        
        out_dir = dataDir2 + 'output_' + '_'.join(input_frm_dir.split('_')[:-1]) +'/'      # 004-output_Marathon-20mins_01/' 
    
        #transfer to accuracy and detection speed in each segment
    
        input_dir = out_dir          # out_dir is created from profiling API
        
        out_dir = input_dir + 'profiling_result/' 
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        getEachSegmentProfilingAPTime(input_dir, segment_time, out_dir)
        
        
if __name__== "__main__":
    
    #segment_time = 4
    #execute_segment_performance(segment_time)
    
    
    execute_frame_performance()
