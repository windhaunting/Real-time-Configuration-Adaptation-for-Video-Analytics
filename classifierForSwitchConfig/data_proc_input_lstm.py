#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:33:14 2019

@author: fubao
"""

# input_data_transfer; read pose estimation result and transfer
# into the basic format of numpy


#each image is a 5 x 18 points array

import sys
import os
import pickle

import numpy as np
import pandas as pd

from glob import glob
from blist import blist

from common_classifier import read_config_name_from_file


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir2
from profiling.common_prof import frameRates


COCO_KP_NUM = 17      # total 17 keypoints
REPRESENT_DIM = 5           # representation dimensions.  mean, std so far



def calculateRepresentationForHumans(keypoints_arr):
    '''
    we use each keypoint's mean, std, detected or not as representation of all the human keypoints
    keypoints_arr dimensions:  no of people detected num x 51  
    output: dimension 5*17       ( or 5*17 later  2x51 (17 x 3)  for test ?)
    
    '''
    #repres_arr = np.zeros((2, COCO_KP_NUM*2))
    
    repres_arr = np.zeros((2, keypoints_arr.shape[1]))  # 2*51
    #detected_indxs = range(2, COCO_KP_NUM*3, 3)
    
    repres_arr[0, :] = np.mean(keypoints_arr, axis = 0)
    repres_arr[1, :] = np.std(keypoints_arr, axis = 0)

    #repres_arr[0, :]  = np.delete(repres_arr[0, :], detected_indxs)
    #repres_arr[1, :]  = np.delete(repres_arr[1, :], detected_indxs)
    
    #print ("repres_arr: ", detected_indxs, repres_arr.shape, repres_arr)

    # transfer to REPRESENT_DIM x COCO_KP_NUM
    repres_arr_new = np.zeros((REPRESENT_DIM, COCO_KP_NUM))
    
    repres_arr_new[0] = repres_arr[0, 0::3]     # mean.x
    repres_arr_new[1] = repres_arr[0, 1::3]     # mean.y
    repres_arr_new[2] = repres_arr[1, 0::3]     # std.x
    repres_arr_new[3] = repres_arr[1, 1::3]     # std.y
    repres_arr_new[4] = repres_arr[0, 2::3]     # visited_flag

    #print ("repres_arr: ", repres_arr_new.shape, repres_arr_new)
    
    return repres_arr_new


def transferInputData(reso, est_res):
    '''
    transfer pose estimation result into unified feature format
    '''
    #    [500, 220, 2, 514, 214, 2, 498, 210, 2, 538, 232, 2, 0, 0, 0, 562, 308, 2, 470, 304, 2, 614, 362, 2, 420, 362, 2,
    # 674, 398, 2, 372, 394, 2, 568, 468, 2, 506, 468, 2, 596, 594, 2, 438, 554, 2, 616, 696, 2, 472, 658, 2],1.4246317148208618;
    #[974, 168, 2, 988, 162, 2, 968, 158, 2, 1004, 180, 2, 0, 0, 0, 1026, 244, 2, 928, 250, 2, 1072, 310, 2, 882, 302, 2, 1112, 360, 2, 810, 346, 2, 1016, 398, 2, 948, 396, 2, 1064, 518, 2, 876, 482, 2, 1060, 630, 2, 900, 594, 2],1.4541109800338745;[6, 172, 2, 16, 164, 2, 6, 162, 2, 48, 182, 2, 0, 0, 0, 68, 256, 2, 10, 254, 2, 112, 312, 2, 0, 0, 0, 168, 328, 2, 0, 0, 0, 70, 412, 2, 28, 420, 2, 108, 600, 2, 0, 0, 0, 144, 692, 2, 0, 0, 0],1.283275842666626;[0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 162, 2, 0, 0, 0, 92, 208, 2, 42, 206, 2, 134, 250, 2, 0, 0, 0, 176, 284, 2, 0, 0, 0, 118, 334, 2, 86, 334, 2, 126, 396, 2, 80, 396, 2, 148, 548, 2, 0, 0, 0],0.9429177641868591
    
    width = int(reso.split('x')[0])
    height = int(reso.split('x')[1])
    
    pose_ests = est_res.split(";")
    num_humans = len(pose_ests)
    keypoints_arr = np.zeros((num_humans, COCO_KP_NUM*3))      # 17x3, 17 keypoints with axis x, y, visibiliy(detected or not)

    #print ("num_humans", reso, num_humans )
    
    for col, human in enumerate(pose_ests):
        pose = human.split(']')[0].replace('[', '')
        keypoints = pose.split(',')
        #print ("pose", type(pose), len(keypoints), keypoints)
        #keypoints = [keypoints[i] for i in range(0, len(keypoints)) if i%3 != 2]
        #keypoints = [keypoints[i] for i in range(0, len(keypoints)) if i%2 == 0 elif i%2 == 1]
        row = 0
        for indx, kp in enumerate(keypoints):
            #if indx % 3 == 0:
            #keypoints_arr[co, :] = keypoints[indx: ]
            if indx % 3 == 0:
                keypoints_arr[col, row] = round(float(kp)/width, 4)
            elif indx % 3 == 1:
                keypoints_arr[col, row] = round(float(kp)/height, 4)
            else:
                keypoints_arr[col, row] = round(float(kp), 4)
            row += 1
    #print ("keypoints", len(keypoints_arr), keypoints_arr)
        
    return keypoints_arr


def getConfigRepresent(reso, model, config_id_dict, frm_id, repres_arr, config_frm_represent_arr):
    '''
    for each frame rate, get the config's id and put  repres_arr into 
    '''
    for frmRt in frameRates:
                
        config = reso + '-' + str(frmRt) + '-' + model.split('_')[0]
        cfg_id = config_id_dict[config]
        
        config_frm_represent_arr[cfg_id, frm_id-1, :, :] = repres_arr
        
    return config_frm_represent_arr

    
def readPoseEstimationKeyPoint(data_pose_keypoint_dir):
    '''
    read into numpy 
    confg_frm_feature_arr 4d array
    each is te 5 x 18 dimension values calculated from key point estimation results of multi persons
              frame_id
    config_Id  ...
    
    5 is detected_flag, mean, std, 18 is the number of points
    
    '''
    config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, True)
    
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    config_num = len(config_id_dict)    # len(config_id_dict)
    
    # get frame_num, because maybe there are some frame errors while video processing, it is not processed
    frame_num = 0
    for filePath in filePathLst:
        df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)         # det-> detection
        frame_num_tmp = len(df_det)        # because maybe some frame_id is missing
        if frame_num_tmp > frame_num:
            frame_num = frame_num_tmp
    
    
    config_frm_represent_arr = np.zeros((config_num, frame_num, REPRESENT_DIM, COCO_KP_NUM)) #
    
    for fileCnt, filePath in enumerate(filePathLst):
        print ("fileCnt ",  fileCnt, filePath)
        #parse each images 
        df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)         # det-> detection
        

        print ("numy shape: ", config_frm_represent_arr.shape, filePath)
        
        if '1120x832' in filePath and 'cmu' in filePath:        # neglect the most expensive config as ground truth for caluclating accuracy and resource cost
            continue
        for index, row in df_det.iterrows():  
            #print ("index, row: ", index, row)
            reso = row['Resolution']
            #frm_rate = row['Frame_rate']
            model = row['Model']
            #num_humans = row['numberOfHumans']        # number of human detected
            
            frm_id = int(row['Image_path'].split('/')[-1].split('.')[0])
            
            est_res = row['Estimation_result']
            
            # 
            #print ("frm_id num_humans, ",frm_id, num_humans, est_res)
            
            keypoints_arr = transferInputData(reso, est_res)
            repres_arr = calculateRepresentationForHumans(keypoints_arr)
            
            config_frm_represent_arr = getConfigRepresent(reso, model, config_id_dict, frm_id, repres_arr, config_frm_represent_arr)
            # get the config's representation of this frame  5 frame_rate considered
            
            #config_frm_represent_arr[]
            #if index == 0:
            #    break   # debug only
                
        #if fileCnt == 0:
        #    break   # debug only
        
    print ("config_frm_represent_arr: ", config_frm_represent_arr.shape)        

    out_frm_repres_pickle_dir = data_pose_keypoint_dir + "representation_files_lstm/" 
    if not os.path.exists(out_frm_repres_pickle_dir):
            os.mkdir(out_frm_repres_pickle_dir)
            
    with open(out_frm_repres_pickle_dir + "config_frames_representation_pkl",'wb') as fs:
        pickle.dump(config_frm_represent_arr, fs)


     
def readPoseEstimationAccSPFPickle(data_pickle_dir):
    '''
    Each pickle
    x = 1
    '''
    
    file_lst= ['acc_frame.pkl', 'spf_frame.pkl']

    acc_frame_arr = np.load(data_pickle_dir + file_lst[0])
    spf_frame_arr = np.load(data_pickle_dir + file_lst[1])
    #acc_seg_arr = np.load(data_pickle_dir + file_lst[2])
    #spf_seg_arr = np.load(data_pickle_dir + file_lst[3])
    
    print ("acc_frame_arr ", type(acc_frame_arr), acc_frame_arr)
    
    return acc_frame_arr, spf_frame_arr
    

if __name__== "__main__": 
        
    data_pose_keypoint_dir =  dataDir2 + 'output_006-cardio_condition-20mins/'
    
    #read_config_name_from_file(data_pose_keypoint_dir, True)
    
    readPoseEstimationKeyPoint(data_pose_keypoint_dir)