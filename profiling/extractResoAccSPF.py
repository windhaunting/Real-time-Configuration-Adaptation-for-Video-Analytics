#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:31:00 2019

@author: fubao
"""




#scp -P 5122 -r ~/workDir/video_analytics_pose_estimation/profiling/*.py fubao@ipanema.ecs.umass.edu:/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/videoAnalytics_poseEstimation/profiling/


# combine the functions of segmentProcess.py and writeIntoPickle.py two files together

import re
import os
import math
import pandas as pd
import numpy as np

import sys
import csv

import pickle

from blist import blist
from collections import defaultdict
from glob import glob

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur)

#sfrom common_prof import dataDir2
from common_prof import  dataDir3
from common_prof import frameRates
from common_prof import PLAYOUT_RATE
from common_prof import NUM_KEYPOINT

from common_prof import computeOKS_mat
from common_prof import computeOKSACC
from common_prof import resoStrLst_OpenPose
from common_prof import extraction_kp_to_numpy


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from tian.preprocess import record2mat


'''
Resolution	index
'320x240'	0
'480x352'	1
'640x480'	2
'960x720'	3
'1120x832'	4

Index	Keypoint
0	nose
1	leftEye
2	rightEye
3	leftEar
4	rightEar
5	leftShoulder
6	rightShoulder
7	leftElbow
8	rightElbow
9	leftWrist
10	rightWrist
11	leftHip
12	rightHip
13	leftKnee
14	rightKnee
15	leftAnkle
16	rightAnkle

'''

# extract each frame result of acc, spf 
# relative to resolution only, frame_rate is considered as 25; because every frame considered


# (0(?:\.\d*)?), (0(?:\.\d*)?), ([0123])



def write_reso_frm_poseEst_result_kp_speed(data_pose_keypoint_dir, data_pickle_dir):
    '''
    input: read all poest estimation kp files
    output the numpy array and write into pickle file
    the specific format is referred as in "record2mat" function
    '''
    
    #config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)
    
    reso_id_dict = {r:i for i, r in enumerate(resoStrLst_OpenPose)}
    
    id_reso_dict = {i:r for i, r in enumerate(resoStrLst_OpenPose)}
    print ("numy reso_id_dict: ", reso_id_dict, id_reso_dict)
    
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    reso_num = len(resoStrLst_OpenPose)  # len(config_id_dict)
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection
    frame_num = len(df_det)     #  maybe some frame_id is missing, only consider all frames that could be parsed from a video
    #create a numpy array
    reso_frm_est_arr = np.zeros((reso_num, frame_num, NUM_KEYPOINT, 3), dtype=object) # array of estimation result with config vs frame_Id
    reso_frm_conf_score_arr = np.zeros((reso_num, frame_num)) # array of time_spf with config vs frame_Id
    reso_frm_spf_arr = np.zeros((reso_num, frame_num)) # array of time_spf with config vs frame_Id
    
    
    for fileCnt, filePath in enumerate(filePathLst):
        
        
        if 'a_cpn' in filePath or 'mobilenet_v2_small' in filePath:
            continue
        df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)         # det-> detection
        
        #kps_arr, spf_arr, conf_sc_ar = record2mat(filePath) 
        for index, row in df_det.iterrows():  
            #print ("index, row: ", index, row)
            reso = row['Resolution']
            #frm_rate = row['Frame_rate']
            model = row['Model'].split('_')[0]
            #frm_id = int(row['Image_path'].split('/')[-1].split('.')[0])
            est_res = row['Estimation_result']
            time_spf = row['Time_SPF']
            
            num_humans = int(row['numberOfHumans'])
            #print ("reso: ", reso)
            #get the config index
            
            id_reso = reso_id_dict[reso]
            
            kps_arr, conf_scr = extraction_kp_to_numpy(est_res, num_humans)
            #print (" kps_arr, spf_arr, conf_sc_ar : ",  kps_arr.shape,  type(conf_scr), conf_scr)
            
            reso_frm_est_arr[id_reso, index] = kps_arr
            reso_frm_conf_score_arr[id_reso, index] = conf_scr
            
            reso_frm_spf_arr[id_reso, index] = time_spf
                    
        #break    # test only
        
        #break    # test only
        
        
    reso_frm_est_arr = reso_frm_est_arr[:, :-1]
    reso_frm_spf_arr  = reso_frm_spf_arr[:, :-1]
    reso_frm_conf_score_arr = reso_frm_conf_score_arr[:, :-1]

    
    print ("confg_frm_est_arr: ", id_reso, reso_frm_est_arr.shape, reso_frm_est_arr[4][0], reso_frm_spf_arr[4][0])            

    with open(data_pickle_dir + 'reso_estimation_frm.pkl','wb') as fs:
        pickle.dump(reso_frm_est_arr, fs)
  
    #out_frm_spf_pickle_file = pickle_dir + "spf_frame.pkl"      # spf for config vs each frame
    with open(data_pickle_dir + 'reso_conf_score_frm.pkl','wb') as fs:
        pickle.dump(reso_frm_conf_score_arr, fs)
        
    #out_frm_spf_pickle_file = pickle_dir + "spf_frame.pkl"      # spf for config vs each frame
    with open(data_pickle_dir + 'reso_spf_frm.pkl','wb') as fs:
        pickle.dump(reso_frm_spf_arr, fs)


    return


    
'''
def write_reso_frm_poseEst_result(data_pose_keypoint_dir, data_pickle_dir):
    
    #frame-by-frame consideration
    #get config's estimation result based on each config and frame
    #and the spf result 

    
    #config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)
    
    reso_id_dict = {r:i for i, r in enumerate(resoStrLst_OpenPose)}
    
    id_reso_dict = {i:r for i, r in enumerate(resoStrLst_OpenPose)}
    print ("numy reso_id_dict: ", reso_id_dict, id_reso_dict)
    
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    reso_num = len(resoStrLst_OpenPose)  # len(config_id_dict)
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection
    frame_num = len(df_det)     #  maybe some frame_id is missing, only consider all frames that could be parsed from a video
    #create a numpy array
    reso_frm_est_arr = np.zeros((reso_num, frame_num), dtype=object) # array of estimation result with config vs frame_Id
    
    reso_frm_spf_arr = np.zeros((reso_num, frame_num)) # array of time_spf with config vs frame_Id
    
    
    for fileCnt, filePath in enumerate(filePathLst):
        
        
        if 'a_cpn' in filePath or 'mobilenet_v2_small' in filePath:
            continue
        df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)         # det-> detection
        

        print ("numy shape: ", reso_frm_est_arr.shape, filePath)
        
        for index, row in df_det.iterrows():  
            #print ("index, row: ", index, row)
            reso = row['Resolution']
            #frm_rate = row['Frame_rate']
            model = row['Model'].split('_')[0]
            #frm_id = int(row['Image_path'].split('/')[-1].split('.')[0])
            est_res = row['Estimation_result']
            time_spf = row['Time_SPF']
            #print ("reso: ", reso)
            #get the config index
            
            id_reso = reso_id_dict[reso]
            reso_frm_est_arr[id_reso, index] = est_res
            
            reso_frm_spf_arr[id_reso, index] = time_spf

            #break    # test only
        
        #break    # test only
        
        
    reso_frm_est_arr = reso_frm_est_arr[:, :-1]
    reso_frm_spf_arr  = reso_frm_spf_arr[:, :-1]
    
    print ("confg_frm_est_arr: ", id_reso, reso_frm_est_arr.shape, reso_frm_est_arr[4][0], reso_frm_spf_arr[4][0])            

    with open(data_pickle_dir + 'reso_estimation_frm.pkl','wb') as fs:
        pickle.dump(reso_frm_est_arr, fs)
      
    #out_frm_spf_pickle_file = pickle_dir + "spf_frame.pkl"      # spf for config vs each frame
    with open(data_pickle_dir + 'reso_spf_frm.pkl','wb') as fs:
        pickle.dump(reso_frm_spf_arr, fs)

    return
'''



def readResoFrmEstFile(data_pickle_dir):
    '''
    read confg_frm_est_arr from the output of write_config_frm_poseEst_result
    '''
    
    #pickle_dir = dataDir2 + 'output_006-cardio_condition-20mins/' + 'pickle_files/'
    
    pickleFile = data_pickle_dir + 'reso_estimation_frm.pkl'
    reso_frm_est_arr = np.load(pickleFile)

    #print ("confg_frm_est_arr: ", (confg_frm_est_arr[:, 0]))
    #return
    return reso_frm_est_arr



'''
def apply_acc_fun(arrs):
    
    #print ("commmmmmm: ", type((arrs[0])), arrs[0])
    #print ("commmmmmm22: ", str(arrs[0]) == 'nan')
    #print ("commmmmmm33: ", np.isnan(arrs[0]))

    if str(arrs[0]) == 'nan' and str(arrs[1]) == 'nan':
        return 1.0
    elif str(arrs[0]) == 'nan' and str(arrs[1]) != 'nan':
        return 0.0
    elif str(arrs[0]) != 'nan' and str(arrs[1]) == 'nan':
        return 0.0
    
    #because only one person in the video; 
    arr0 = getOnePersonEstimation(arrs[0])
    arr1 = getOnePersonEstimation(arrs[1])
    acc = computeOKSAP(arr0, arr1, '')

    return acc



def getOnePersonEstimation(ests_arr):
    #use only one person select the confidence score higher
    strLst = re.findall(r'],\d.\d+', ests_arr)
    person_lst = [re.findall(r'\d.\d+', st) for st in strLst]
    
    ind = np.argmax(person_lst)
    #print ("ind: ", person_lst, ind)
    
    return ests_arr.split(';')[ind]
    

def calculate_config_frm_acc(ests_arr, gts_arr):
    #each input it's the array of all frames
    
    print ("ests_arr shape: ", ests_arr.shape, gts_arr.shape)
    #combine together
    combine_arr = np.vstack((ests_arr, gts_arr))
    
    #print ("combine_arr shape: ", combine_arr.shape)
    acc_arr = np.apply_along_axis(apply_acc_fun, 0, combine_arr)

    #print ("acc_arr: ", acc_arr.shape, acc_arr)
    
    return acc_arr



def write_reso_frm_acc_result(reso_frm_est_arr, data_pickle_out_dir):
    #frame-by-frame consideration
    #get each config acc for each frame, frame rate is actually 25.
    
    # select the ground truth used config id, here it's a fixed number
    # so id is 0, that is it's the first line
    # only use one person
    gts_arr = reso_frm_est_arr[0]  #ground truth for each frame    1120*83-25-cmu
    
    print ("write_reso_frm_acc_result reso_frm_est_arr shape: ", reso_frm_est_arr.shape, gts_arr.shape)
    reso_frm_acc_arr = np.apply_along_axis(calculate_config_frm_acc, [1,2,3], reso_frm_est_arr, gts_arr)
    
    print ("config_frm_acc_arr1 final: ", reso_frm_acc_arr.shape, reso_frm_acc_arr[0])
    print ("config_frm_acc_arr2 final: ", reso_frm_acc_arr)
    
    out_frm_acc_pickle_file = data_pickle_out_dir + "reso_acc_frm.pkl"      # acc for config vs each frame
    
    with open(out_frm_acc_pickle_file,'wb') as fs:
        pickle.dump(reso_frm_acc_arr, fs)   

def apply_oks_fun(arrs):
    
    #print ("commmmmmm: ", type((arrs[0])), arrs[0])
    #print ("commmmmmm22: ", str(arrs[0]) == 'nan')
    #print ("commmmmmm33: ", np.isnan(arrs[0]))
    
    if str(arrs[0]) == 'nan' and str(arrs[1]) == 'nan':
        return 1.0
    elif str(arrs[0]) == 'nan' and str(arrs[1]) != 'nan':
        return 0.0
    elif str(arrs[0]) != 'nan' and str(arrs[1]) == 'nan':
        return 0.0
    arr0 = getOnePersonEstimation(arrs[0])
    arr1 = getOnePersonEstimation(arrs[1])
    oks = computeOKSFromOrigin(arr0, arr1, '')
    
    return oks



def calculate_config_frm_oks(ests_arr, gts_arr):
    #each input it's the array of all frames
    
    #print ("ests_arr shape: ", ests_arr.shape, gts_arr.shape)
    #combine together
    
    combine_arr = np.vstack((ests_arr, gts_arr))
    
    #print ("combine_arr shape: ", combine_arr.shape)
    acc_arr = np.apply_along_axis(apply_oks_fun, 0, combine_arr)

    #print ("acc_arr: ", acc_arr.shape, acc_arr)
    
    return acc_arr

def write_reso_frm_oks_result(reso_frm_est_arr, data_pickle_out_dir):
    #frame-by-frame consideration
    #get each config acc for each frame, frame rate is actually 25.
    
    # select the ground truth used config id, here it's a fixed number
    # so id is 0, that is it's the first line
    
    gts_arr = reso_frm_est_arr[0]  #ground truth for each frame    1120*83-25-cmu
    
    reso_frm_oks_arr = np.apply_along_axis(calculate_config_frm_oks, 1, reso_frm_est_arr, gts_arr)  # each frame
    
    print ("config_frm_oks_arr1 final: ", reso_frm_oks_arr.shape, reso_frm_oks_arr[0])
    print ("config_frm_oks_arr2 final: ", reso_frm_oks_arr[:, 1])
    
    out_frm_acc_pickle_file = data_pickle_out_dir + "reso_oks_frm.pkl"      # acc for config vs each frame
    
    with open(out_frm_acc_pickle_file,'wb') as fs:
        pickle.dump(reso_frm_oks_arr, fs)   
'''        


def write_reso_frm_oks_result(reso_frm_est_arr, data_pickle_out_dir):
    #get oks,  frame-by-frame consideration
    #get each config acc for each frame, frame rate is actually 25.
    
    conf_nums = reso_frm_est_arr.shape[0]
    frm_nums = reso_frm_est_arr.shape[1]
    
    reso_frm_oks_arr = np.zeros((conf_nums, frm_nums))
    for row_ind in range(0, conf_nums):
        for col_ind in range(0, frm_nums):
            gt_arr = reso_frm_est_arr[0][col_ind]               # index 0 as the ground truth
            est_arr = reso_frm_est_arr[row_ind][col_ind]
            #print ("reso_frm_oks_arr: ", gt_arr.shape, est_arr.shape) 
            reso_frm_oks_arr[row_ind][col_ind] = computeOKS_mat(gt_arr, est_arr)
            
    print ("reso_frm_oks_arr: ", reso_frm_oks_arr.shape, reso_frm_oks_arr[:, 1]) 
            
    out_frm_oks_pickle_file = data_pickle_out_dir + "reso_oks_frm.pkl"      # acc for config vs each frame
    
    with open(out_frm_oks_pickle_file,'wb') as fs:
        pickle.dump(reso_frm_oks_arr, fs)   
        
        
def write_reso_frm_acc_result(reso_frm_est_arr, data_pickle_out_dir):
    #get acc, frame-by-frame consideration
    #get each config acc for each frame, frame rate is actually 25.
    
    conf_nums = reso_frm_est_arr.shape[0]
    frm_nums = reso_frm_est_arr.shape[1]
    
    reso_frm_acc_arr = np.zeros((conf_nums, frm_nums))
    for row_ind in range(0, conf_nums):
        for col_ind in range(0, frm_nums):
            gt_arr = reso_frm_est_arr[0][col_ind]               # index 0 as the ground truth
            est_arr = reso_frm_est_arr[row_ind][col_ind]
            #print ("reso_frm_oks_arr: ", gt_arr.shape, est_arr.shape) 
            reso_frm_acc_arr[row_ind][col_ind] = computeOKSACC(gt_arr, est_arr)
            
    print ("reso_frm_acc_arr: ", reso_frm_acc_arr.shape, reso_frm_acc_arr[:, 1]) 
            
    out_frm_acc_pickle_file = data_pickle_out_dir + "reso_acc_frm.pkl"      # acc for config vs each frame
    
    with open(out_frm_acc_pickle_file,'wb') as fs:
        pickle.dump(reso_frm_acc_arr, fs)   
    

def executeWriteIntoPickleOnePeron():
    '''
    #Resolution only as rows
    '''
    
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/']
    
    
    for vd_dir in video_dir_lst[0:10]:        # [3:4]:   # [0:1]:
        
        data_pickle_out_dir = dataDir3 +  vd_dir + 'frames_pickle_result_resolution_only/'
        if not os.path.exists(data_pickle_out_dir):
            os.mkdir(data_pickle_out_dir)
            
        #write_reso_frm_poseEst_result_kp_speed(dataDir3 +  vd_dir, data_pickle_out_dir)
        
        reso_frm_est_arr = readResoFrmEstFile(data_pickle_out_dir)
        write_reso_frm_acc_result(reso_frm_est_arr, data_pickle_out_dir)
        write_reso_frm_oks_result(reso_frm_est_arr, data_pickle_out_dir)
        
        
if __name__== "__main__":
    
    executeWriteIntoPickleOnePeron()