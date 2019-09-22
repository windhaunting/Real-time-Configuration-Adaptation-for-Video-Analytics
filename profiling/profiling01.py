#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:46:04 2019

@author: fubao
"""

# get a profiling result and store into csv with dataframe
import sys 
import os 
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

dataDir3 = '../
PLAYOUT_RATE = 25


# define a class including each clip's profile result
class cls_profile_video(object):
    # cameraNo is the camera no. for multiple camera streaming. 
    # queryNo is the query type, currently human pose estimation
    # 'frameStartNo', 
    __slots__ = ['cameraNo', 'queryNo', 'resolution', 'frameRate','modelMethod', 'accuracy', 'costFPS']


frameRates = [25, 10, 5, 2, 1]    # test only [25, 10, 5, 2, 1]   # [5]   #          #  [25]    #  [25, 10, 5, 2, 1]    # [30],  [30, 10, 5, 2, 1] 
resoStrLst= ["1120x832", "960x720", "640x480",  "480x352", "320x240"]   #  [720, 600, 480, 360, 240]   # [240] #     # [240]       # [720, 600, 480, 360, 240]    #   [720]     # [720, 600, 480, 360, 240]  #  [720]    # [720, 600, 480, 360, 240]            #  16: 9
resolutions = [(int(res.split("x")[0]), int(res.split("x")[1])) for res in resoStrLst]
modelMethods = ['cmu', 'mobilenet_v2_small']  #  ['mobilenet_v2_small']      # ['mobilenet_thin']  # ['cmu']  #  ["openPose"]

#print ("resolutions:" , resolutions)


def profilingVideoWithMaxFrameRateMPII(inputDir, outDir):
    '''
    for a video with 1mins only for each video, not online detection from MPII dataset
    
    read multiple video's frames (in each vidoe batch sub directories, there are only 41 frames available,
    therefore, we need to read multiple subdirectories and combine them together as a video of 1 minutes)
    
    1min => 25*60 = 1500 frames 
    use each config to read each segment and alculate the accuracy, average detection speed(SPF) for each segment
    also restore the detection result in a file for the video
    
    first we get different resolution with frame rates 25's result of each frame and store into csv file
    
    #Configuration_C_index, Frame_rate, Resolution, Model, image_Id, category_Id,  detection_keyPoint, scores

    '''
        
    #get a video length 1 mins 
    video_time_len = 60
    
    total_frames_len = video_time_len * PLAYOUT_RATE         # 5 is test value only   video_time_len * PLAYOUT_RATE         # 60*25 = 1500
    
    #segmentNum = total_frames_len//segment_frames        # 
        
    configIndex = 0
    for res in resoStrLst:    # resolutions
        fr = frameRates[0]        # use only maximum frame rate is enough
        for mod in modelMethods:
            #'resolution', 'frameRate','modelMethod',
            
            configIndex += 1
            # each file is a config of  the video
            out_file = outDir + str(res)+ '_' + str(fr) + '_' + str(mod) + '_estimation_result.tsv'
                
            headStr = 'Configuration_C_index' + '\t' + 'Resolution' +'\t' + 'Frame_rate' +'\t' + 'Model' +   \
             '\t' + 'Image_path' +'\t' + str('Estimation_result') + '\t' + 'numberOfHumans' + '\t'+ 'Time_SPF' + '\n'  # '\t' + 'image_Id'+ '\t' + 'category_Id' +'\t' + 'detection_keyPoint' + '\t' + 'scores' + '\n'
            
            # load model
            e, w, h = load_model(mod, res)
            
            with open(out_file, 'a') as f:
                f.write(headStr)

                frmCnt = 1        # fram step size
                subDirLst = sorted(glob(inputDir + "*/"))
                
                finishOneMinVideoFlag = False
                for subDir in subDirLst:          # execute one by one? memory is not enough
                    filePathLst = sorted(glob(subDir + "*.jpg"))
                    #print ("filePath: ", filePath)
                    for imgPath in filePathLst:      # iterate each subdirect's frames and compose a video
                        if frmCnt >= total_frames_len:   # until 1min
                            
                            print ("frmCnt: ", frmCnt, imgPath)
                            finishOneMinVideoFlag = True
                            break
                            
                        # apply different configuration
                        # estimate pose of this frame  call the open_pose etc. interface
                        preWriteStr = str(configIndex) + '\t' + str(res) +'\t' + str(fr) +'\t' + \
                        str(mod) + '\t'
                        #if mod == "cmu": # "openPose":
                        #output_lst, elapsedTime = openPose_estimation_one_image(imgPath, res) #mulitple person
                        output_lst, elapsedTime = tf_open_pose_inference(imgPath, mod, res, e, w, h)
                        #save into file
                        if output_lst is None or len(output_lst) == 0:
                            continue
                        
                        print ("output_lst: ", type(output_lst))
                        #for outDic in output_lst:
                        #    print ("outDic:" , type(outDic), outDic)
                        preWriteStr += imgPath + '\t' + str(output_lst)  + '\t'  + str(len(output_lst)) + '\t' + str(elapsedTime) + '\n'      # str(outDic['image_Id']) + '\t' + str(outDic['category_Id']) + '\t' + ','.join(outDic['detection_keyPoint']) + '\t' + str(outDic['scores']) + "\n"

                        f.write(preWriteStr)
                                
                        frmCnt += 1
                    if finishOneMinVideoFlag:
                        break
                   




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
        
   
    configIndex = 0
    for res in resoStrLst:    # resolutions
        fr = frameRates[0]        # use only maximum frame rate is enough
        for mod in modelMethods:
            #'resolution', 'frameRate','modelMethod',
            
            configIndex += 1
            # each file is a config of  the video
            out_file = outDir + str(res)+ '_' + str(fr) + '_' + str(mod) + '_estimation_result.tsv'
                
            headStr = 'Configuration_C_index' + '\t' + 'Resolution' +'\t' + 'Frame_rate' +'\t' + 'Model' +   \
             '\t' + 'Image_path' +'\t' + str('Estimation_result') + '\t' + 'numberOfHumans' + '\t'+ 'Time_SPF' + '\n'  # '\t' + 'image_Id'+ '\t' + 'category_Id' +'\t' + 'detection_keyPoint' + '\t' + 'scores' + '\n'
            

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
                    #save into file
                    if output_lst is None or len(output_lst) == 0:
                        continue
                    
                    #print ("output_lst: ", type(output_lst))
                    #for outDic in output_lst:
                    #    print ("outDic:" , type(outDic), outDic)
                    preWriteStr += imgPath + '\t' + str(output_lst)  + '\t'  + str(len(output_lst)) + '\t' + str(elapsedTime) + '\n'      # str(outDic['image_Id']) + '\t' + str(outDic['category_Id']) + '\t' + ','.join(outDic['detection_keyPoint']) + '\t' + str(outDic['scores']) + "\n"

                    f.write(preWriteStr)

                    frmCnt += 1



    
def getEachFrameAPTime(inputDir, outDir):
    '''
    get each frames's avearge precision(AP) and detection time
    
    read each file's detection result of each config, compare with the most expensive configuration and
    calculate the AP and time

    outProfFile:
        ['streamingNo', 'imagePath', 'resolution', 'frameRate','modelMethod', 'accuracy', 'costFPS']

    '''

    gtDic = defaultdict(list)        # ground truth detection file for each frame
    
    print ("inputDir: ", inputDir)
    filePathLst = sorted(glob(inputDir + "*.tsv"))  # must read ground truth file(the most expensive config) first
    for fileCnt, filePath in enumerate(filePathLst):
       # read poste estimation detection result file
       
        df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)  # det-> detection
        print ("filePath: ", filePath, df_det.columns)
       

        fileName = filePath.split('/')[-1]
       
        outDirProfile = outDir
        if not os.path.exists(outDirProfile):
            os.mkdir(outDirProfile)
        outProfFile = outDirProfile + '_'.join(fileName.split('_')[:3]) + '_profiling_result.tsv'
        
        reso = fileName.split('_')[0]
        w = int(reso.split('x')[0])
        h = int(reso.split('x')[1])
        
        if '1120x832_25_cmu' in fileName:  # this file has the most expensive config result
            # get the ground truth for each image_path
            # pose estimation result is ground truth, accuracy is 1, add a new column "Acc"
            # get ground truth dictionary with estimation result
            gtDic = dict(zip(df_det.Image_path, df_det.Estimation_result))
            df_det['Acc'] = 1 
            
            #for k, v in gtDic.items():
            #    print ("k: ", type(k), k, type(v), v)
            #    break
        else:
            # calculate the accuracy
            def computeOKSAcc(img):
                # compute oks with ground truth and average precision
                # get the imagepath df[df['B']==3]['A'],
                ''''
                img_path = df_det[df_det['Estimation_result'] == est_result]['Image_path']      # df_det.loc[df_det.Estimation_result == est_result, 'Image_path']
                print ("img_path:" , type(img_path), img_path)
                # get gt of estimation result in string
                gt_result = gtDic[img_path]
                print ("gt_result:" , type(gt_result), gt_result)
                # computeOks
                '''
                est_result = df_det[df_det['Image_path'] == img]['Estimation_result'].item()
                #print ("img_path:" , type(img), type(est_result), est_result)
                if img not in gtDic:
                    acc = 0
                    print ("img not found in ground truth")
                    return acc
                else:
                    gt_result = gtDic[img]
                # computeOks
                acc = computeOKSAP(est_result, gt_result, img, w, h)
                
                return acc
            # acculate accuracy on the Estimation_result column
            df_det['Acc'] = df_det.iloc[:, 4].apply(computeOKSAcc)    # df_det.iloc[:, 4]

            
        #remove a column
        df_det.drop('Estimation_result', axis=1, inplace=True)
        df_det.drop('numberOfHumans', axis=1, inplace=True)
            
        df_det.to_csv(outProfFile, sep='\t', index=False)
               
        #if (fileCnt >= 1):
        #    break
    
             
def getEachConfigurationForSegment(inputDir, segment_time, outDir):   
    '''
    get each segment time's profiling window's accuarcy and speed
    '''    
        
    # frameRates = [25, 10, 5, 2, 1] 
    
    
    segment_frames = PLAYOUT_RATE*segment_time  # 25*4 = 100   
    profilingTime = segment_time//4       
    profiling_frames = int(PLAYOUT_RATE*profilingTime)    # frames in the profiling time
    
    #profileFrmIntervals = [25//fr for fr in frameRates]   # sampling rate to get accuracy
    
    filePathLst = sorted(glob(inputDir + "*.tsv"))  # must read ground truth file(the most expensive config) first
    
    out_file = outDir + 'profiling_segment_time' + str(segment_time)+ '.tsv'
    
    headStr = 'Resolution' +'\t' + 'Frame_rate' +'\t' + 'Model' +   \
             '\t' + 'Segment_no' +'\t' + 'Detection_speed_FPS'  + '\t' + str('Acc') +'\n'   # '\t' + 'image_Id'+ '\t' + 'category_Id' +'\t' + 'detection_keyPoint' + '\t' + 'scores' + '\n'
        
    with open(out_file, 'w') as f:
        f.write(headStr)
        
        fileCnt = 0
        while (fileCnt < len(filePathLst)):
            # get the average accuaracy
            
            segment_no = 1
            startFrmCnt = 0
    
            df_prof = pd.read_csv(filePathLst[fileCnt], delimiter='\t', index_col=False)  # det-> detection
    
            endFrmCnt = segment_frames
            
            # get config
            #Configuration_C_index	Resolution	Frame_rate	Model	Image_path	Time_SPF	Acc
            
            config_index = df_prof.iat[0, 0]
            resolution = df_prof.iat[0, 1]
            frame_rate = df_prof.iat[0, 2]
            model = df_prof.iat[0, 3]
            
            framesTotalNum = df_prof.shape[0]
            print ("framesTotalNum: ",filePathLst[fileCnt], segment_frames, framesTotalNum)
            
            while (startFrmCnt < framesTotalNum):
                
                #startFrmCnt = 0
                # profileFrmIntervals
                for frmRate in frameRates:
                    profileFrmInter = int(PLAYOUT_RATE//frmRate)
                    
                    #print ("profileFrmInter: ",profiling_frames, profileFrmInter, type( df_prof.iloc[startFrmCnt:profiling_frames]))
                    #get det_speed in the profiling time
                    det_speed = float(df_prof.iloc[startFrmCnt:startFrmCnt+profiling_frames].iloc[::profileFrmInter, 5].sum())/PLAYOUT_RATE
                    #for index, row in  df_prof.iloc[startFrmCnt:profiling_frames].iterrows():
                    #print ("profileFrmInter: ",  det_speed)
                    det_speed = round(1/det_speed, 2)
                    
                    #get acc in the profiling time
                    acc = round(df_prof.iloc[startFrmCnt:startFrmCnt+profiling_frames].iloc[::profileFrmInter, 6].mean(), 2)
                    
                    rowStr =  str(resolution) + '\t' + str(frmRate) + '\t' + str(model) +   \
                    '\t'  + str(segment_no) + '\t' + str(det_speed) + '\t' + str(acc) + '\n'
                    f.write(rowStr)

                startFrmCnt = endFrmCnt
                endFrmCnt += segment_frames
                if endFrmCnt > framesTotalNum:   # not enough
                    endFrmCnt = framesTotalNum
                    
                segment_no += 1
                
            fileCnt += 1
    
    
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
        
    #df_prof_segment.insert(0, column="Config_index",value = df_prof_segment.index)

    #write into different files
    
    
if __name__== "__main__":
    
    '''
    #videoPath01 = dataDir1 + 'videoBatch01/'
    #outDir = dataDir + 'output/' 
    #if not os.path.exists(outDir):
    #    os.mkdir(outDir)
    #profilingVideoWithMaxFrameRate(videoPath01, outDir)
    
    #outDir = dataDir + 'output/' 
    #getEachFrameAPTime(outDir)
    
    inputDir = dataDir + 'output/profiling_result/'
    segment_time = 4
    outDir = inputDir + 'segment_result/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
        
    getEachConfigurationForSegment(inputDir, segment_time, outDir)
    '''
    
    
    inputDir = dataDir2 + 'video_dancing_01_frames/'
    outDir = dataDir2 + 'output_video_dancing_01/' 
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    profilingOneVideoWithMaxFrameRate(inputDir, outDir)
    
    inputDir = dataDir2 + 'output_video_dancing_01/' 
    outDir = dataDir2 + 'output_video_dancing_01/profiling_result/' 
    getEachFrameAPTime(inputDir, outDir)    
    
    
    inputDir = dataDir2 + 'output_video_dancing_01/profiling_result/'
    segment_time = 4
    outDir = inputDir + 'segment_result/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
        
    getEachConfigurationForSegment(inputDir, segment_time, outDir)
    
    
