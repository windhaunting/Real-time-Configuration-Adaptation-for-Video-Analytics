#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:54:42 2019

@author: fubao
"""
# calculate EMA feature Feature_One: calculate the EMA of current speed + relative speed. The current speed use only current frame and it previous one frame

# use the most expensive config's pose estimation result to calcualate the speed feature
#consider the switching based on the selected config for next frame

#swtiching config use bounded accuracy only

import re
import sys
import os
import csv
import pickle
import math
import cv2
import time

import numpy as np
import pandas as pd

from glob import glob
from blist import blist
from collections import defaultdict

from common_classifier import read_all_config_name_from_file
from common_classifier import read_poseEst_conf_frm
from common_classifier import readProfilingResultNumpy
from common_classifier import getParetoBoundary
from common_classifier import checkCorrelationPlot
from common_classifier import extract_specific_config_name_from_file

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir3
from profiling.common_prof import frameRates
from profiling.common_prof import PLAYOUT_RATE

from profiling.common_prof import computeOKSAP
from profiling.common_prof import computeOKSFromOrigin

COCO_KP_NUM = 17      # total 17 keypoints


'''
{{0, "Nose"}, //t
{1, "Neck"}, //if is not included in coco. How do you get the Neck keypoints
{2, "RShoulder"}, //t
{3, "RElbow"}, //t
{4, "RWrist"}, //t
{5, "LShoulder"}, //t
{6, "LElbow"}, //t
{7, "LWrist"}, //t
{8, "RHip"}, //t
{9, "RKnee"}, //t
{10, "RAnkle"}, //t
{11, "LHip"}, //t
{12, "LKnee"}, //t
{13, "LAnkle"}, //t
{14, "REye"}, //t
{15, "LEye"}, //t
{16, "REar"}, //t
{17, "LEar"}, //t
{18, "Bkg"}},
'''

# the keypoint printout in csv is actually tensorflow coco format

'''
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


# use EMA EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
# =  Price(current)  x Multiplier)   + (1-Multiplier) * EMA(prev) 


ALPHA_EMA = 0.8   # EMA

def getPersonEstimation(est_res, width, height):
    '''
    analyze the personNo's pose estimation result with highest confidence score
    input: 
        [500, 220, 2, 514, 214, 2, 498, 210, 2, 538, 232, 2, 0, 0, 0, 562, 308, 2, 470, 304, 2, 614, 362, 2, 420, 362, 2, 674, 398, 2, 372, 394, 2, 568, 468, 2, 506, 468, 2, 596, 594, 2, 438, 554, 2, 616, 696, 2, 472, 658, 2],1.4246317148208618;
        [974, 168, 2, 988, 162, 2, 968, 158, 2, 1004, 180, 2, 0, 0, 0, 1026, 244, 2, 928, 250, 2, 1072, 310, 2, 882, 302, 2, 1112, 360, 2, 810, 346, 2, 1016, 398, 2, 948, 396, 2, 1064, 518, 2, 876, 482, 2, 1060, 630, 2, 900, 594, 2],1.4541109800338745;
        [6, 172, 2, 16, 164, 2, 6, 162, 2, 48, 182, 2, 0, 0, 0, 68, 256, 2, 10, 254, 2, 112, 312, 2, 0, 0, 0, 168, 328, 2, 0, 0, 0, 70, 412, 2, 28, 420, 2, 108, 600, 2, 0, 0, 0, 144, 692, 2, 0, 0, 0],1.283275842666626;
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 162, 2, 0, 0, 0, 92, 208, 2, 42, 206, 2, 134, 250, 2, 0, 0, 0, 176, 284, 2, 0, 0, 0, 118, 334, 2, 86, 334, 2, 126, 396, 2, 80, 396, 2, 148, 548, 2, 0, 0, 0],0.9429177641868591
        
        
    output:
        a specific person's detection result arr (17x2)
    '''
    '''
    #print ("est_res: ", type(est_res), est_res)
    person_est = est_res.split(';')[personNo].split('],')[0].replace('[', '')
    
    kp_arr = np.zeros((COCO_KP_NUM, 2))    # 17x2 each kp is a two-dimensional value
    

    for ind, kp in enumerate(person_est.split(',')):
        #if ind%3 == 0:
        #    kp_arr[int(ind/3), ind%3] = float(kp)*1120
        #elif ind%3 == 1:
        #    kp_arr[int(ind/3), ind%3] = float(kp)*832
        if ind%3 != 2:
            kp_arr[int(ind/3), ind%3] = float(kp)
    '''
    
    strLst = re.findall(r'],\d.\d+', est_res)
    person_score = [re.findall(r'\d.\d+', st) for st in strLst]
    
    personNo = np.argmax(person_score)
    
    
    est_res = est_res.split(';')[personNo]
    
    lst_points = [[float(t[0]), float(t[1])] for t in re.findall(r'(0(?:\.\d*)?), (0(?:\.\d*)?), ([0123])', est_res)]

    kp_arr = np.array(lst_points)
    
    #print ("kp_arr: ", kp_arr.shape, kp_arr)
    return kp_arr
    
    

def getEuclideanDist(val, time_frm_interval):
    '''
    calculate the euclidean distance
    val: [x1, y1, x2, y2]
    '''
    #or vector distance
    
    #speed_angle = [val[0]-val[2], val[1]-val[3]]

    
    dist = math.sqrt((val[1] - val[3])**2 + (val[0] - val[2])**2)
               
    speed = dist/time_frm_interval
    
    if (math.sqrt(val[0]**2 + val[1]**2) *math.sqrt(val[2]**2 + val[3]**2)) == 0:
        cos_angle = 0
    else:
        cos_angle = (val[0]*val[2] + val[1]*val[3]) / (math.sqrt(val[0]**2 + val[1]**2) *math.sqrt(val[2]**2 + val[3]**2))
    
    speed_angle = [speed, cos_angle]
    #print ("speed: ", val, speed)

    
    return speed_angle
    


def getFeatureOnePersonMovingSpeed(history_pose_est_arr, current_frm_id, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_speed_arr):
    '''
    feature1: One person’s moving speed of all keypoints V i,k based on the
    euclidean d distance of current frame with the previous frame {f j−m , m =
                                                                   1, 2, ..., 24
    only get the previous history_frame_num, that is current_frm_id is the current frame id
    the previous 24 frames are the previous frames.
    
    output: feature1_speed_arr COCO_KP_NUM x 1
    '''
    cur_frm_est_arr = history_pose_est_arr[current_frm_id]
    #print ("aaaa, ", current_frm_id, cur_frm_est_arr, len(range(current_frm_id-1, current_frm_id-1-history_frame_num-1, -1)))
    
    #feature1_speed_arr = np.zeros((COCO_KP_NUM, 2))
    
    
    prev_frm_est_arr = history_pose_est_arr[current_frm_id-1]
    
    # get the current speed based on current frame and previous frame
    hstack_arr = np.hstack((cur_frm_est_arr, prev_frm_est_arr))
    #frmInter = math.ceil(PLAYOUT_RATE/curr_frm_rate)          # frame rate sampling frames in interval, +1 every other

    time_frm_interval = 1*(1.0/PLAYOUT_RATE)*(skipped_frm_cnt+1)
    #time_frm_interval = 1.0/PLAYOUT_RATE
    current_speed_angle_arr = np.apply_along_axis(getEuclideanDist, 1, hstack_arr, time_frm_interval)    
    
    feature1_speed_arr = current_speed_angle_arr * ALPHA_EMA  + (1-ALPHA_EMA) * prev_EMA_speed_arr
    
    #print ("feature1_speed_arr, ", time_frm_interval, feature1_speed_arr)
        
    return feature1_speed_arr

    
def relativeVectorDistance(arr):
    '''
    arr [[x1, y1],[x2,y2],...,[x5, y5]]
    
    get the distance of x5 to xi (i=1,2,3,4)
    
    output: 4x2 arr
    '''
    
    num_kp = arr.shape[0]
    #print ("val ", arr, num_kp)
    rel_val_arr = np.zeros((num_kp-1, 2))
    for i in range(0, num_kp-1):
        rel_val_arr[i] =  [abs(arr[i][1] -arr[num_kp-1][1]), abs(arr[i][0] -arr[num_kp-1][0])]
    
    #print ("rel_dist_arrrrrr: ", arr[i], arr[4], rel_val_arr)
    
    return rel_val_arr
    
def relativeSpeed(arr1, arr2, time_frm_interval):
    '''
    input arr1: 4x2 , arr2: 4x2
    output: 4x1
    '''
    rel_vec_1 = relativeVectorDistance(arr1)
    rel_vec_2 = relativeVectorDistance(arr2)
    
    #relative_speed_arr = rel_vec_1 - rel_vec_2         # vector of relatve  speed
    
    # or use direction and angle
    combine_arr = np.hstack((rel_vec_1, rel_vec_2))
    
    #print ("combine_arr: ", rel_vec_1.shape,relative_speed_arr.shape, combine_arr.shape, combine_arr)
    relative_speed_arr = np.apply_along_axis(getEuclideanDist, 1, combine_arr, time_frm_interval)
    
    #print ("relative_speed_arr: ", relative_speed_arr.shape, relative_speed_arr)
    #print ("relative_speed_arr: ", rel_dist1, rel_dist2, relative_speed_arr)
    return relative_speed_arr
    

def getFeatureOnePersonRelativeSpeed1(history_pose_est_arr, current_frm_id, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr):
    '''
    feature 2 One person arm/feet’s relative speed to the torso of all keypoints
    with the previous frames.
    In each frame, we select the relative distance of left wrist, right wrist,left ankle
    and right ankle to the left hip.
    
    output: feature2_relative_speed_arr 5 x history_frame_num-1

    '''
    # get left wrist, right wrist,left ankle, right ankle and the left hip
    # which corresponds to [7, 4, 13, 10, 11]
    selected_kp_index = [9, 10, 15, 16, 11]   #  [7, 4, 13, 10, 11]
    selected_kp_history_pose_est_arr = history_pose_est_arr[:, selected_kp_index, :]
    #print ("selected_kp_history_pose_est_arr aaa, ", selected_kp_history_pose_est_arr[0])
    
    curr_frm_rel_dist_arr = selected_kp_history_pose_est_arr[current_frm_id]   #np.apply_along_axis(relativeDistance, 2, selected_kp_history_pose_est_arr)
    
    previous_frm_arr = selected_kp_history_pose_est_arr[current_frm_id-1]

    #frmInter = math.ceil(PLAYOUT_RATE/curr_frm_rate)          # frame rate sampling frames in interval, +1 every other

    time_frm_interval = 1*(1.0/PLAYOUT_RATE)*(skipped_frm_cnt+1)
    
    current_speed_arr = relativeSpeed(curr_frm_rel_dist_arr, previous_frm_arr, time_frm_interval)
    
    feature2_relative_speed_arr = current_speed_arr * ALPHA_EMA  + (1-ALPHA_EMA) * prev_EMA_relative_speed_arr
    
    return feature2_relative_speed_arr


def getFeatureOnePersonRelativeSpeed2(history_pose_est_arr, current_frm_id, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr):
    '''
    feature 2 One person arm/feet’s relative speed to the torso of all keypoints
    with the previous frames.
    In each frame, we select the relative distance of left wrist, right wrist,left ankle
    and right ankle to the right hip.
    
    output: feature2_relative_speed_arr 5 x history_frame_num-1

    '''
    # get left wrist, right wrist,left ankle, right ankle and the left hip
    # which corresponds to [7, 4, 13, 10, 11]
    selected_kp_index = [9, 10, 15, 16, 12]        # [7, 4, 13, 10, 8]
    selected_kp_history_pose_est_arr = history_pose_est_arr[:, selected_kp_index, :]
    #print ("selected_kp_history_pose_est_arr aaa, ", selected_kp_history_pose_est_arr[0])
    
    curr_frm_rel_dist_arr = selected_kp_history_pose_est_arr[current_frm_id]   #np.apply_along_axis(relativeDistance, 2, selected_kp_history_pose_est_arr)
    
    previous_frm_arr = selected_kp_history_pose_est_arr[current_frm_id-1]

    #frmInter = math.ceil(PLAYOUT_RATE/curr_frm_rate)          # frame rate sampling frames in interval, +1 every other

    time_frm_interval = 1*(1.0/PLAYOUT_RATE)*(skipped_frm_cnt+1)
    
    current_speed_arr = relativeSpeed(curr_frm_rel_dist_arr, previous_frm_arr, time_frm_interval)
    
    feature2_relative_speed_arr = current_speed_arr * ALPHA_EMA  + (1-ALPHA_EMA) * prev_EMA_relative_speed_arr
    
    return feature2_relative_speed_arr


def getFeatureOnePersonRelativeSpeed3(history_pose_est_arr, current_frm_id, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr):
    '''
    feature 2 One person arm/feet’s relative speed to the torso of all keypoints
    with the previous frames.
    In each frame, we select the relative distance of left wrist, right wrist,left ankle
    and right ankle to the left shoulder.
    
    output: feature2_relative_speed_arr 5 x history_frame_num-1

    '''
    # get left wrist, right wrist,left ankle, right ankle and the left hip
    # which corresponds to [7, 4, 13, 10, 11]
    selected_kp_index = [9, 10, 15, 16, 5]    # [7, 4, 13, 10, 1]
    selected_kp_history_pose_est_arr = history_pose_est_arr[:, selected_kp_index, :]
    #print ("selected_kp_history_pose_est_arr aaa, ", selected_kp_history_pose_est_arr[0])
    
    curr_frm_rel_dist_arr = selected_kp_history_pose_est_arr[current_frm_id]   #np.apply_along_axis(relativeDistance, 2, selected_kp_history_pose_est_arr)
    
    previous_frm_arr = selected_kp_history_pose_est_arr[current_frm_id-1]

    #frmInter = math.ceil(PLAYOUT_RATE/curr_frm_rate)          # frame rate sampling frames in interval, +1 every other

    time_frm_interval = 1*(1.0/PLAYOUT_RATE)*(skipped_frm_cnt+1)
    
    current_speed_arr = relativeSpeed(curr_frm_rel_dist_arr, previous_frm_arr, time_frm_interval)
    
    feature2_relative_speed_arr = current_speed_arr * ALPHA_EMA  + (1-ALPHA_EMA) * prev_EMA_relative_speed_arr
    
    return feature2_relative_speed_arr

def getFeatureOnePersonRelativeSpeed4(history_pose_est_arr, current_frm_id, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr):
    '''
    feature 2 One person arm/feet’s relative speed to the torso of all keypoints
    with the previous frames.
    In each frame, we select the relative distance of left wrist, right wrist,left ankle
    and right ankle to the right shoulder.
    
    output: feature2_relative_speed_arr 5 x history_frame_num-1

    '''
    # get left wrist, right wrist,left ankle, right ankle and the left hip
    # which corresponds to [7, 4, 13, 10, 11]
    selected_kp_index = [9, 10, 15, 16, 6]    # [7, 4, 13, 10, 1]
    selected_kp_history_pose_est_arr = history_pose_est_arr[:, selected_kp_index, :]
    #print ("selected_kp_history_pose_est_arr aaa, ", selected_kp_history_pose_est_arr[0])
    
    curr_frm_rel_dist_arr = selected_kp_history_pose_est_arr[current_frm_id]   #np.apply_along_axis(relativeDistance, 2, selected_kp_history_pose_est_arr)
    
    previous_frm_arr = selected_kp_history_pose_est_arr[current_frm_id-1]

    #frmInter = math.ceil(PLAYOUT_RATE/curr_frm_rate)          # frame rate sampling frames in interval, +1 every other

    time_frm_interval = 1*(1.0/PLAYOUT_RATE)*(skipped_frm_cnt+1)
    
    current_speed_arr = relativeSpeed(curr_frm_rel_dist_arr, previous_frm_arr, time_frm_interval)
    
    feature2_relative_speed_arr = current_speed_arr * ALPHA_EMA  + (1-ALPHA_EMA) * prev_EMA_relative_speed_arr
    
    return feature2_relative_speed_arr


def getOnePersonFeatureInputOutput01(data_pose_keypoint_dir, data_pickle_dir, history_frame_num, max_frame_example_used, minAccuracy):
    '''
    get one person's all history keypoint
    One person’s moving speed of all keypoints V i,k based on the euclidean d distance of current frame with the previous frame {f j−m , m = 1, 2, ..., 24}
    
    current_frame_id is also included in the next frame's id
    
    the input feature here we use the most expensive features first
    
    1120x832_25_cmu_estimation_result
    
    start from history_frame_num;
    the first previous history frames are neglected
    
    based on EMA
    
    
    '''
    
    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)

    print ("getOnePersonFeatureInputOutput01 acc_frame_arr: ", acc_frame_arr[:, 0])


    # select one person, i.e. no 0
    personNo = 0
    
    #max_frame_example_used = 1000   # 8000
    #current_frame_id = 25
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)
    
    print ("config_id_dict: ", len(config_id_dict))
    # only read the most expensive config
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*1120x832_25_cmu_estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection

    print ("filePath: ", filePathLst[0], len(df_det))

    
    history_pose_est_arr = np.zeros((max_frame_example_used+1, COCO_KP_NUM, 2)) # np.zeros((len(df_det), COCO_KP_NUM, 2))        #  to make not shift when new frames comes, we store all values
    
    previous_frm_indx = 0
    
    
    input_x_arr = np.zeros((max_frame_example_used, 21, 2))       # 17 + 4
    y_out_arr = np.zeros((max_frame_example_used+1), dtype=int)
    
    prev_EMA_speed_arr = np.zeros((COCO_KP_NUM, 2))
    
    prev_EMA_relative_speed_arr = np.zeros((4, 2))        # only get 4 keypoint
        
    current_delay = 0.0

    select_frm_cnt = 0
    skipped_frm_cnt = 0

    switching_config_skipped_frm = -1
    switching_config_inter_skip_cnts = 0
    
    selected_configs_acc_lst = blist()      # check the selected config's for all frame's lst, in order to get the accuracy
    for index, row in df_det.iterrows():  
        #print ("index, row: ", index, row)
        #reso = row['Resolution']
        #frm_rate = row['Frame_rate']
        #model = row['Model']
        #num_humans = row['numberOfHumans']        # number of human detected

        if index >= acc_frame_arr.shape[1]:
            break            
        
        frm_id = int(row['Image_path'].split('/')[-1].split('.')[0])
        
        est_res = row['Estimation_result']
        
        if str(est_res) == 'nan':  # here select_frm_cnt does not increase
            skipped_frm_cnt += 1
            print ("est_res: ", est_res, index, select_frm_cnt)
            continue
        
        # skipping interval by frame_rate
        if switching_config_skipped_frm != - 1 and switching_config_skipped_frm < switching_config_inter_skip_cnts-skipped_frm_cnt:   # switching_config_inter_skip_cnts:
            switching_config_skipped_frm += 1
            continue
        
        
        #print ("frm_id num_humans, ", reso, model, frm_id)
            
        kp_arr = getPersonEstimation(est_res, personNo)
        #history_pose_est_dict[previous_frm_indx] = kp_arr
         
        history_pose_est_arr[previous_frm_indx] = kp_arr
        #print ("kp_arr, ", kp_arr)
        #break    # debug only
        if previous_frm_indx>= history_frame_num:
            #print ("previous_frm_indx, ", previous_frm_indx, index)
            
            current_cofig = id_config_dict[int(y_out_arr[select_frm_cnt])]
            curr_frm_rate = int(current_cofig.split('-')[1])
            #print ("xxxx current_cofig: ", current_cofig, curr_frm_rate)
            # calculate the human moving speed feature (1)
            feature1_speed_arr = getFeatureOnePersonMovingSpeed(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_speed_arr)
            
            prev_EMA_speed_arr = feature1_speed_arr
            #calculate the relative moving speed feature (2)
            feature2_relative_speed_arr = getFeatureOnePersonRelativeSpeed(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr)
            
            prev_EMA_relative_speed_arr = feature2_relative_speed_arr
            #print ("feature1_speed_arr feature2_relative_speed_arr, ", feature1_speed_arr.shape, feature2_relative_speed_arr.shape)
            
            total_features_arr = np.vstack((feature1_speed_arr, feature2_relative_speed_arr))
            #print ("total_features_arr total_features_arr, ", frm_id,  total_features_arr.shape)
            
            input_x_arr[select_frm_cnt]= total_features_arr  #  input_x_arr[frm_id-1] = total_features_arr
            #print ("total_features_arr: ", frm_id-1)
            #previous_frm_indx = 1
                
            y_out_arr[select_frm_cnt+1] = select_config(acc_frame_arr, spf_frame_arr,  select_frm_cnt+1+switching_config_inter_skip_cnts+skipped_frm_cnt, minAccuracy)
                        
            current_cofig = id_config_dict[int(y_out_arr[select_frm_cnt])]
            
            selected_config_acc = acc_frame_arr[y_out_arr[select_frm_cnt], index]
            
            selected_configs_acc_lst.append(selected_config_acc)
            #print ("current_cofig: ", current_cofig)
            
            current_config_frmRt = int(current_cofig.split('-')[1])
            
            if switching_config_skipped_frm == - 1:
                switching_config_inter_skip_cnts = PLAYOUT_RATE-1 #  math.ceil(PLAYOUT_RATE/current_config_frmRt)-2  #       #math.ceil(PLAYOUT_RATE/frmRt)-1
            else:
                switching_config_inter_skip_cnts = PLAYOUT_RATE  # math.ceil(PLAYOUT_RATE/current_config_frmRt)-1  # PLAYOUT_RATE
                
            skipped_frm_cnt = 0       
            select_frm_cnt += 1 
            switching_config_skipped_frm = 0

        previous_frm_indx += 1
        
        #how many are used for traing, validation, and test
        if index > (max_frame_example_used-1):
            break 
    
    
    input_x_arr = input_x_arr[history_frame_num:select_frm_cnt].reshape(input_x_arr[history_frame_num:select_frm_cnt].shape[0], -1)
       
    y_out_arr = y_out_arr[history_frame_num+1:select_frm_cnt+1]

    print ("input_x_arraaaxxx, ", select_frm_cnt, input_x_arr, input_x_arr.shape, feature1_speed_arr.shape, feature2_relative_speed_arr.shape, current_delay, len(selected_configs_acc_lst), sum(selected_configs_acc_lst)/len(selected_configs_acc_lst))

    return input_x_arr, y_out_arr


def getFrmRateFeature(history_frmRt_arr, prev_frmRt_aver):
    
    
    if history_frmRt_arr.shape[0] < 10:
        current_frmRt_aver = np.mean(history_frmRt_arr)
    else:
        current_frmRt_aver = np.mean(history_frmRt_arr[0:10])
    feature_frmRate = current_frmRt_aver * ALPHA_EMA  + (1-ALPHA_EMA) * prev_frmRt_aver

    return feature_frmRate

def getOnePersonFeatureInputOutput02(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy):
    '''
    get one person's all history keypoint, plus over a period of frmRate feature
    One person’s moving speed of all keypoints V i,k based on the euclidean d distance of current frame with the previous frame {f j−m , m = 1, 2, ..., 24}
    
    current_frame_id is also included in the next frame's id
    
    the input feature here we use the most expensive features first
    
    1120x832_25_cmu_estimation_result
    
    start from history_frame_num;
    the first previous history frames are neglected
    
    based on EMA
    '''
    
    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)

    print ("getOnePersonFeatureInputOutput01 acc_frame_arr: ", acc_frame_arr[:, 0])


    # select one person, i.e. no 0
    personNo = 0
    
    #max_frame_example_used = 1000   # 8000
    #current_frame_id = 25
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)
    
    print ("config_id_dict: ", len(config_id_dict))
    # only read the most expensive config
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*1120x832_25_cmu_estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection

    print ("filePath: ", filePathLst[0], len(df_det))

    
    history_pose_est_arr = np.zeros((max_frame_example_used+1, COCO_KP_NUM, 2)) # np.zeros((len(df_det), COCO_KP_NUM, 2))        #  to make not shift when new frames comes, we store all values
    
    previous_frm_indx = 0
    
    
    input_x_arr = np.zeros((max_frame_example_used, 21, 2))       # 17 + 4
    y_out_arr = np.zeros((max_frame_example_used+1), dtype=int)
    
    prev_EMA_speed_arr = np.zeros((COCO_KP_NUM, 2))
    
    prev_EMA_relative_speed_arr = np.zeros((4, 2))        # only get 4 keypoint
        
    frmRt_feature_arr = np.zeros(max_frame_example_used)
    
    history_frmRt_arr = np.zeros(max_frame_example_used)
    prev_frmRt_aver = 0.0
    

    select_frm_cnt = 0
    skipped_frm_cnt = 0

    switching_config_skipped_frm = -1
    switching_config_inter_skip_cnts = 0
    
    selected_configs_acc_lst = blist()      # check the selected config's for all frame's lst, in order to get the accuracy
    for index, row in df_det.iterrows():  
        #print ("index, row: ", index, row)
        #reso = row['Resolution']
        #frm_rate = row['Frame_rate']
        #model = row['Model']
        #num_humans = row['numberOfHumans']        # number of human detected

        if index >= acc_frame_arr.shape[1]:
            break            
        
        frm_id = int(row['Image_path'].split('/')[-1].split('.')[0])
        
        est_res = row['Estimation_result']
        
        if str(est_res) == 'nan':  # here select_frm_cnt does not increase
            skipped_frm_cnt += 1
            print ("est_res: ", est_res, index, select_frm_cnt)
            continue
        
        # skipping interval by frame_rate
        if switching_config_skipped_frm != - 1 and switching_config_skipped_frm < switching_config_inter_skip_cnts-skipped_frm_cnt:   # switching_config_inter_skip_cnts:
            switching_config_skipped_frm += 1
            continue
        
        
        #print ("frm_id num_humans, ", reso, model, frm_id)
            
        kp_arr = getPersonEstimation(est_res, personNo)
        #history_pose_est_dict[previous_frm_indx] = kp_arr
         
        history_pose_est_arr[previous_frm_indx] = kp_arr
        #print ("kp_arr, ", kp_arr)
        #break    # debug only
        if previous_frm_indx>= history_frame_num:
            #print ("previous_frm_indx, ", previous_frm_indx, index)
            
            current_cofig = id_config_dict[int(y_out_arr[select_frm_cnt])]
            curr_frm_rate = int(current_cofig.split('-')[1])
            #print ("xxxx current_cofig: ", current_cofig, curr_frm_rate)
            # calculate the human moving speed feature (1)
            feature1_speed_arr = getFeatureOnePersonMovingSpeed(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_speed_arr)
            
            prev_EMA_speed_arr = feature1_speed_arr
            #calculate the relative moving speed feature (2)
            feature2_relative_speed_arr = getFeatureOnePersonRelativeSpeed(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr)
            
            prev_EMA_relative_speed_arr = feature2_relative_speed_arr
            #print ("feature1_speed_arr feature2_relative_speed_arr, ", feature1_speed_arr.shape, feature2_relative_speed_arr.shape)
            
            total_features_arr = np.vstack((feature1_speed_arr, feature2_relative_speed_arr))
            #print ("total_features_arr total_features_arr, ", frm_id,  total_features_arr.shape)
            
            input_x_arr[select_frm_cnt]= total_features_arr  #  input_x_arr[frm_id-1] = total_features_arr
            #print ("total_features_arr: ", frm_id-1)
            #previous_frm_indx = 1
                
            y_out_arr[select_frm_cnt+1] = select_config(acc_frame_arr, spf_frame_arr,  select_frm_cnt+1+switching_config_inter_skip_cnts+skipped_frm_cnt, minAccuracy)
                        
            current_cofig = id_config_dict[int(y_out_arr[select_frm_cnt])]
            
            selected_config_acc = acc_frame_arr[y_out_arr[select_frm_cnt], index]
            
            selected_configs_acc_lst.append(selected_config_acc)
            #print ("current_cofig: ", current_cofig)
            
            current_config_frmRt = int(current_cofig.split('-')[1])
            
            if switching_config_skipped_frm == - 1:
                switching_config_inter_skip_cnts = PLAYOUT_RATE-1 #  math.ceil(PLAYOUT_RATE/current_config_frmRt)-2  #       #math.ceil(PLAYOUT_RATE/frmRt)-1
            else:
                switching_config_inter_skip_cnts = PLAYOUT_RATE  # math.ceil(PLAYOUT_RATE/current_config_frmRt)-1  # PLAYOUT_RATE
                
            skipped_frm_cnt = 0       
            select_frm_cnt += 1 
            switching_config_skipped_frm = 0

            frmRt = int(current_cofig.split('-')[1])

            history_frmRt_arr[select_frm_cnt] = frmRt
                
            curr_frmRt_aver= getFrmRateFeature(history_frmRt_arr, prev_frmRt_aver)
            prev_frmRt_aver = curr_frmRt_aver
            
            frmRt_feature_arr[select_frm_cnt] = curr_frmRt_aver
            

        previous_frm_indx += 1
        
        #how many are used for traing, validation, and test
        if index > (max_frame_example_used-1):
            break 
    
    
    input_x_arr = input_x_arr[history_frame_num:select_frm_cnt].reshape(input_x_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    
    frmRt_feature_arr = frmRt_feature_arr[history_frame_num:select_frm_cnt].reshape(frmRt_feature_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((input_x_arr, frmRt_feature_arr))
   
    y_out_arr = y_out_arr[history_frame_num+1:select_frm_cnt+1]
    print ("frmRt_feature_arr, ",frmRt_feature_arr.shape, frmRt_feature_arr)
    print ("feature1_speed_arr, ", input_x_arr, input_x_arr.shape, feature1_speed_arr.shape, feature2_relative_speed_arr.shape)

    return input_x_arr, y_out_arr


def getConfigFeature(history_config_arr, select_frm_cnt, prev_config_aver):
    
    
    if select_frm_cnt < 2:
        current_confg_aver = np.mean(history_config_arr[:select_frm_cnt+1])
    else:
        current_confg_aver = np.mean(history_config_arr[select_frm_cnt-1:select_frm_cnt])
    #print ("CCCCCCCC:" , history_config_arr, prev_config_aver)
    feature_confg = int(current_confg_aver * ALPHA_EMA  + (1-ALPHA_EMA) * prev_config_aver)

    return feature_confg


def getOnePersonFeatureInputOutput03(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy):
    '''
    get one person's all history keypoint,  plus over a period of config feature
    One person’s moving speed of all keypoints V i,k based on the euclidean d distance of current frame with the previous frame {f j−m , m = 1, 2, ..., 24}
    
    current_frame_id is also included in the next frame's id
    
    the input feature here we use the most expensive features first
    
    1120x832_25_cmu_estimation_result
    
    start from history_frame_num;
    the first previous history frames are neglected
    
    based on EMA
    add over a period of config a feature
    '''
    
    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)

    print ("getOnePersonFeatureInputOutput01 acc_frame_arr: ", acc_frame_arr[:, 0])


    # select one person, i.e. no 0
    personNo = 0
    
    #max_frame_example_used = 1000   # 8000
    #current_frame_id = 25
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)
    
    print ("config_id_dict: ", len(config_id_dict))
    # only read the most expensive config
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*1120x832_25_cmu_estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection

    print ("filePath: ", filePathLst[0], len(df_det))

    
    history_pose_est_arr = np.zeros((max_frame_example_used+1, COCO_KP_NUM, 2)) # np.zeros((len(df_det), COCO_KP_NUM, 2))        #  to make not shift when new frames comes, we store all values
    
    previous_frm_indx = 0
    
    
    input_x_arr = np.zeros((max_frame_example_used, 21, 2))       # 17 + 4
    y_out_arr = np.zeros((max_frame_example_used+1), dtype=int)
    
    prev_EMA_speed_arr = np.zeros((COCO_KP_NUM, 2))
    
    prev_EMA_relative_speed_arr = np.zeros((4, 2))        # only get 4 keypoint
        
    config_feature_arr = np.zeros(max_frame_example_used, dtype=int)
    
    history_config_arr = np.zeros(max_frame_example_used)
    prev_config_aver = 0.0
    
    
    select_frm_cnt = 0
    skipped_frm_cnt = 0

    switching_config_skipped_frm = -1
    switching_config_inter_skip_cnts = 0
    
    selected_configs_acc_lst = blist()      # check the selected config's for all frame's lst, in order to get the accuracy
    for index, row in df_det.iterrows():  
        #print ("index, row: ", index, row)
        #reso = row['Resolution']
        #frm_rate = row['Frame_rate']
        #model = row['Model']
        #num_humans = row['numberOfHumans']        # number of human detected

        if index >= acc_frame_arr.shape[1]:
            break            
        
        frm_id = int(row['Image_path'].split('/')[-1].split('.')[0])
        
        est_res = row['Estimation_result']
        
        if str(est_res) == 'nan':  # here select_frm_cnt does not increase
            skipped_frm_cnt += 1
            print ("est_res: ", est_res, index, select_frm_cnt)
            continue
        
        # skipping interval by frame_rate
        if switching_config_skipped_frm != - 1 and switching_config_skipped_frm < switching_config_inter_skip_cnts-skipped_frm_cnt:   # switching_config_inter_skip_cnts:
            switching_config_skipped_frm += 1
            continue
        
        #print ("frm_id num_humans, ", reso, model, frm_id)
            
        kp_arr = getPersonEstimation(est_res, personNo)
        #history_pose_est_dict[previous_frm_indx] = kp_arr
         
        history_pose_est_arr[previous_frm_indx] = kp_arr
        #print ("kp_arr, ", kp_arr)
        #break    # debug only
        if previous_frm_indx>= history_frame_num:
            #print ("previous_frm_indx, ", previous_frm_indx, index)
            
            current_cofig = id_config_dict[int(y_out_arr[select_frm_cnt])]
            curr_frm_rate = int(current_cofig.split('-')[1])
            #print ("xxxx current_cofig: ", current_cofig, curr_frm_rate)
            # calculate the human moving speed feature (1)
            feature1_speed_arr = getFeatureOnePersonMovingSpeed(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_speed_arr)
            
            prev_EMA_speed_arr = feature1_speed_arr
            #calculate the relative moving speed feature (2)
            feature2_relative_speed_arr = getFeatureOnePersonRelativeSpeed(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr)
            
            prev_EMA_relative_speed_arr = feature2_relative_speed_arr
            #print ("feature1_speed_arr feature2_relative_speed_arr, ", feature1_speed_arr.shape, feature2_relative_speed_arr.shape)
            
            total_features_arr = np.vstack((feature1_speed_arr, feature2_relative_speed_arr))
            #print ("total_features_arr total_features_arr, ", frm_id,  total_features_arr.shape)
            
            input_x_arr[select_frm_cnt]= total_features_arr  #  input_x_arr[frm_id-1] = total_features_arr
            #print ("total_features_arr: ", frm_id-1)
            #previous_frm_indx = 1
                
            y_out_arr[select_frm_cnt+1] = select_config(acc_frame_arr, spf_frame_arr,  select_frm_cnt+1+switching_config_inter_skip_cnts+skipped_frm_cnt, minAccuracy)
                        
            current_cofig = id_config_dict[int(y_out_arr[select_frm_cnt])]
            
            selected_config_acc = acc_frame_arr[y_out_arr[select_frm_cnt], index]
            
            selected_configs_acc_lst.append(selected_config_acc)
            #print ("current_cofig: ", current_cofig)
            
            current_config_frmRt = int(current_cofig.split('-')[1])
            
            if switching_config_skipped_frm == - 1:
                switching_config_inter_skip_cnts = PLAYOUT_RATE-1 #  math.ceil(PLAYOUT_RATE/current_config_frmRt)-2  #       #math.ceil(PLAYOUT_RATE/frmRt)-1
            else:
                switching_config_inter_skip_cnts = PLAYOUT_RATE  # math.ceil(PLAYOUT_RATE/current_config_frmRt)-1  # PLAYOUT_RATE
                
            skipped_frm_cnt = 0       
            select_frm_cnt += 1 
            switching_config_skipped_frm = 0

            history_config_arr[select_frm_cnt] = y_out_arr[select_frm_cnt]
                
            curr_config_aver= getConfigFeature(history_config_arr, select_frm_cnt, prev_config_aver)
            
            prev_config_aver = curr_config_aver
            
            config_feature_arr[select_frm_cnt] =  curr_config_aver  # y_out_arr[select_frm_cnt]  # y_out_arr[select_frm_cnt]  #   #
            
            
        previous_frm_indx += 1
        
        #how many are used for traing, validation, and test
        if index > (max_frame_example_used-1):
            break 
    
    
    input_x_arr = input_x_arr[history_frame_num:select_frm_cnt].reshape(input_x_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    
    config_feature_arr = config_feature_arr[history_frame_num:select_frm_cnt].reshape(config_feature_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((input_x_arr, config_feature_arr))
   
    y_out_arr = y_out_arr[history_frame_num+1:select_frm_cnt+1]
    print ("config_feature_arr, ",config_feature_arr.shape, config_feature_arr)
    print ("feature1_speed_arr, ", input_x_arr, input_x_arr.shape, feature1_speed_arr.shape, feature2_relative_speed_arr.shape)

    return input_x_arr, y_out_arr



def getResoFeature(history_reso_arr, prev_reso_aver):
    
    
    if history_reso_arr.shape[0] <= 10:
        current_reso_aver = np.mean(history_reso_arr)
    else:
        current_reso_aver = np.mean(history_reso_arr[0:10])
    feature_Reso = int(current_reso_aver * ALPHA_EMA  + (1-ALPHA_EMA) * prev_reso_aver)

    return feature_Reso

def getOnePersonFeatureInputOutput04(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy):
    '''
    get one person's all history keypoint,  plus over a period of resolution feature
    One person’s moving speed of all keypoints V i,k based on the euclidean d distance of current frame with the previous frame {f j−m , m = 1, 2, ..., 24}
    
    current_frame_id is also included in the next frame's id
    
    the input feature here we use the most expensive features first
    
    1120x832_25_cmu_estimation_result
    
    start from history_frame_num;
    the first previous history frames are neglected
    
    based on EMA
    add over a period of resolution a feature
    '''
    
    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)

    print ("getOnePersonFeatureInputOutput01 acc_frame_arr: ", acc_frame_arr[:, 0])


    # select one person, i.e. no 0
    personNo = 0
    
    #max_frame_example_used = 1000   # 8000
    #current_frame_id = 25
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)
    
    print ("config_id_dict: ", len(config_id_dict))
    # only read the most expensive config
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*1120x832_25_cmu_estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection

    print ("filePath: ", filePathLst[0], len(df_det))

    
    history_pose_est_arr = np.zeros((max_frame_example_used+1, COCO_KP_NUM, 2)) # np.zeros((len(df_det), COCO_KP_NUM, 2))        #  to make not shift when new frames comes, we store all values
    
    previous_frm_indx = 0
    
    
    input_x_arr = np.zeros((max_frame_example_used, 21, 2))       # 17 + 4
    y_out_arr = np.zeros((max_frame_example_used+1), dtype=int)
    
    prev_EMA_speed_arr = np.zeros((COCO_KP_NUM, 2))
    
    prev_EMA_relative_speed_arr = np.zeros((4, 2))        # only get 4 keypoint
        
    reso_feature_arr = np.zeros(max_frame_example_used, dtype=int)
    
    history_reso_arr = np.zeros(max_frame_example_used)
    prev_reso_aver = 0.0
    
    select_frm_cnt = 0
    skipped_frm_cnt = 0

    switching_config_skipped_frm = -1
    switching_config_inter_skip_cnts = 0
    
    selected_configs_acc_lst = blist()      # check the selected config's for all frame's lst, in order to get the accuracy
    for index, row in df_det.iterrows():  
        #print ("index, row: ", index, row)
        #reso = row['Resolution']
        #frm_rate = row['Frame_rate']
        #model = row['Model']
        #num_humans = row['numberOfHumans']        # number of human detected

        if index >= acc_frame_arr.shape[1]:
            break            
        
        frm_id = int(row['Image_path'].split('/')[-1].split('.')[0])
        
        est_res = row['Estimation_result']
        
        if str(est_res) == 'nan':  # here select_frm_cnt does not increase
            skipped_frm_cnt += 1
            print ("nan nan est_res: ", est_res, index, select_frm_cnt)
            continue
        
        # skipping interval by frame_rate
        if switching_config_skipped_frm != - 1 and switching_config_skipped_frm < switching_config_inter_skip_cnts-skipped_frm_cnt:   # switching_config_inter_skip_cnts:
            switching_config_skipped_frm += 1
            continue
        
        #print ("frm_id num_humans, ", reso, model, frm_id)
            
        kp_arr = getPersonEstimation(est_res, personNo)
        #history_pose_est_dict[previous_frm_indx] = kp_arr
         
        history_pose_est_arr[previous_frm_indx] = kp_arr
        #print ("kp_arr, ", kp_arr)
        #break    # debug only
        if previous_frm_indx>= history_frame_num:
            #print ("previous_frm_indx, ", previous_frm_indx, index)
            
            current_cofig = id_config_dict[int(y_out_arr[select_frm_cnt])]
            curr_frm_rate = int(current_cofig.split('-')[1])
            #print ("xxxx current_cofig: ", current_cofig, curr_frm_rate)
            # calculate the human moving speed feature (1)
            feature1_speed_arr = getFeatureOnePersonMovingSpeed(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_speed_arr)
            
            prev_EMA_speed_arr = feature1_speed_arr
            #calculate the relative moving speed feature (2)
            feature2_relative_speed_arr = getFeatureOnePersonRelativeSpeed(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr)
            
            prev_EMA_relative_speed_arr = feature2_relative_speed_arr
            #print ("feature1_speed_arr feature2_relative_speed_arr, ", feature1_speed_arr.shape, feature2_relative_speed_arr.shape)
            
            total_features_arr = np.vstack((feature1_speed_arr, feature2_relative_speed_arr))
            #print ("total_features_arr total_features_arr, ", frm_id,  total_features_arr.shape)
            
            input_x_arr[select_frm_cnt]= total_features_arr  #  input_x_arr[frm_id-1] = total_features_arr
            #print ("total_features_arr: ", frm_id-1)
            #previous_frm_indx = 1
                
            y_out_arr[select_frm_cnt+1] = select_config(acc_frame_arr, spf_frame_arr,  select_frm_cnt+1+switching_config_inter_skip_cnts+skipped_frm_cnt, minAccuracy)
                        
            current_cofig = id_config_dict[int(y_out_arr[select_frm_cnt])]
            
            selected_config_acc = acc_frame_arr[y_out_arr[select_frm_cnt], index]
            
            selected_configs_acc_lst.append(selected_config_acc)
            #print ("current_cofig: ", current_cofig)
            
            current_config_frmRt = int(current_cofig.split('-')[1])
            
            if switching_config_skipped_frm == - 1:
                switching_config_inter_skip_cnts = PLAYOUT_RATE-1 #  math.ceil(PLAYOUT_RATE/current_config_frmRt)-2  #       #math.ceil(PLAYOUT_RATE/frmRt)-1
            else:
                switching_config_inter_skip_cnts = PLAYOUT_RATE  # math.ceil(PLAYOUT_RATE/current_config_frmRt)-1  # PLAYOUT_RATE
                
            reso = int(current_cofig.split('-')[0].split('x')[1])

            history_reso_arr[select_frm_cnt] = reso
                
            curr_reso_aver= getConfigFeature(history_reso_arr, select_frm_cnt, prev_reso_aver)
            
            prev_reso_aver = curr_reso_aver
            
            reso_feature_arr[select_frm_cnt] =  curr_reso_aver # curr_reso_aver y_out_arr[index]    #curr_reso_aver
            #print ("current_cofig reso: ", current_cofig, reso, curr_reso_aver)
         
            skipped_frm_cnt = 0       
            select_frm_cnt += 1 
            switching_config_skipped_frm = 0


        previous_frm_indx += 1
        
        #how many are used for traing, validation, and test
        if index > (max_frame_example_used-1):
            break 
    
    
    input_x_arr = input_x_arr[history_frame_num:select_frm_cnt].reshape(input_x_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    
    reso_feature_arr = reso_feature_arr[history_frame_num:select_frm_cnt].reshape(reso_feature_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((input_x_arr, reso_feature_arr))
   
    y_out_arr = y_out_arr[history_frame_num+1:select_frm_cnt+1]
    #print ("reso_feature_arr, ",reso_feature_arr.shape, reso_feature_arr)
    print ("feature1_speed_arr, ", input_x_arr, input_x_arr.shape, feature1_speed_arr.shape, feature2_relative_speed_arr.shape)

    return input_x_arr, y_out_arr



def getFeatureRelativeDistanceClosenessKeyPoint(history_pose_est_arr, current_frm_id):
    '''
    closeness feature; relative vector distance
    '''
    
    cur_frm_est_arr = history_pose_est_arr[current_frm_id]
    #[9, 10, 15, 16, 11];  [9, 10, 15, 16, 12];   [9, 10, 15, 16, 5] [9, 10, 15, 16, 6] 
    
    closenessPoint_feature1 = np.zeros((4, 2))
    #print ("cur_frm_est_arr: ", cur_frm_est_arr.shape)
    lst_kp_indx1 = [9, 10, 15, 16, 11]
    for ind  in range(0, len(lst_kp_indx1[0:-1])):
        closenessPoint_feature1[ind] = [abs(cur_frm_est_arr[lst_kp_indx1[ind]][0]- cur_frm_est_arr[lst_kp_indx1[-1]][0]), abs(cur_frm_est_arr[lst_kp_indx1[ind]][1]- cur_frm_est_arr[lst_kp_indx1[-1]][1])]
 
    
    closenessPoint_feature2 = np.zeros((4, 2))
    lst_kp_indx2 = [9, 10, 15, 16, 12]

    for ind  in range(0, len(lst_kp_indx1[0:-1])):
        closenessPoint_feature2[ind] = [abs(cur_frm_est_arr[lst_kp_indx2[ind]][0]- cur_frm_est_arr[lst_kp_indx2[-1]][0]), abs(cur_frm_est_arr[lst_kp_indx2[ind]][1]- cur_frm_est_arr[lst_kp_indx2[-1]][1])]
    
    
    closenessPoint_feature3 = np.zeros((4, 2))
    lst_kp_indx3 = [9, 10, 15, 16, 5]
    for ind  in range(0, len(lst_kp_indx1[0:-1])):
        closenessPoint_feature3[ind] = [abs(cur_frm_est_arr[lst_kp_indx3[ind]][0]- cur_frm_est_arr[lst_kp_indx3[-1]][0]), abs(cur_frm_est_arr[lst_kp_indx3[ind]][1]- cur_frm_est_arr[lst_kp_indx3[-1]][1])]
        
    closenessPoint_feature4 = np.zeros((4, 2))
    lst_kp_indx4 = [9, 10, 15, 16, 6]
    for ind  in range(0, len(lst_kp_indx1[0:-1])):
        closenessPoint_feature4[ind] = [abs(cur_frm_est_arr[lst_kp_indx4[ind]][0]- cur_frm_est_arr[lst_kp_indx4[-1]][0]), abs(cur_frm_est_arr[lst_kp_indx4[ind]][1]- cur_frm_est_arr[lst_kp_indx4[-1]][1])]
    

    
    total_features_arr = np.vstack((closenessPoint_feature1, closenessPoint_feature2))
    total_features_arr = np.vstack((total_features_arr, closenessPoint_feature3))
    total_features_arr = np.vstack((total_features_arr, closenessPoint_feature4))
    #print ("total_features_arr :", total_features_arr.shape)
    
    return total_features_arr


def getFeatureDistanceToCamera(history_pose_est_arr, current_frm_id):
    '''
    person size feature
    '''
    
    cur_frm_est_arr = history_pose_est_arr[current_frm_id]
    #[9, 10, 15, 16, 11];  [9, 10, 15, 16, 12];   [9, 10, 15, 16, 5] [9, 10, 15, 16, 6] 
    
    cameraDistance_feature = np.zeros((len(cur_frm_est_arr), 2))   # np.zeros((5, 2))
    #5-leftShoulder
    lst_kp_indx = range(0, len(cur_frm_est_arr))  # [9, 10, 15, 16, 5]
    for i, kp in enumerate(lst_kp_indx):
        cameraDistance_feature[i] = cur_frm_est_arr[kp]           # relative to original point to measure the distance to camera
    #print ("total_features_arr :", total_features_arr.shape)
    
    return cameraDistance_feature


def getBlurrinessFeature(imagePath, current_iterative_frm_id):
    '''
    blurrinessScore can be calculated in 2 ways, 1th using the direct opencv Lapcian method to calculate the frame's blurriness
    2nd se the lowest resolution's acc to indirectly simulate the bllurriness
    '''
    
   # blurrinessScore_arr = acc_frame_arr[lowest_config_id, current_iterative_frm_id]
    
    '''
    gt_result = confg_est_frm_arr[0, current_iterative_frm_id]
    est_result = confg_est_frm_arr[lowest_config_id, current_iterative_frm_id]
    #print ("est_result: ", est_result, gt_result)
    
    if str(est_result) == 'nan' or str(gt_result)== 'nan':  # here select_frm_cnt does not increase
        return 0
    img_path = str(current_iterative_frm_id) + '.jpg'
    
    blurrinessScore = computeOKSAP(est_result, gt_result, img_path)      # computeOKSFromOrigin(est_result, gt_result, img_path)
    #print ("blurrinessScore shape ", blurrinessScore.shape)
    
    #print ("blurrinessScore: ", current_frm_id, blurrinessScore)
    '''
    image = cv2.imread(imagePath)
    if image is None:
        return 250
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurrinessScore_arr = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return blurrinessScore_arr

def getOnePersonFeatureInputOutputAll001(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy):
    '''
    get one person's all history keypoint,  plus over a period of resolution feature
    One person’s moving speed of all keypoints V i,k based on the euclidean d distance of current frame with the previous frame {f j−m , m = 1, 2, ..., 24}
    
    current_frame_id is also included in the next frame's id
    
    the input feature here we use the most expensive features first
    
    1120x832_25_cmu_estimation_result
    
    start from history_frame_num;
    the first previous history frames are neglected
    
    based on EMA
    add over a period of resolution a feature
    '''
    
    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)
    
    #confg_est_frm_arr = read_poseEst_conf_frm(data_pickle_dir)
    #old_acc_frm_arr = acc_frame_arr
    #print ("getOnePersonFeatureInputOutput01 acc_frame_arr: ", acc_frame_arr.shape)

    # save to file to have a look
    #outDir = data_pose_keypoint_dir + "classifier_result/"
    #np.savetxt(outDir + "accuracy_above_threshold" + str(minAccuracy) + ".tsv", acc_frame_arr[:, :5000], delimiter="\t")
    
    
    #config_ind_pareto = getParetoBoundary(acc_frame_arr[:, 0], spf_frame_arr[:, 0])
    
    resolution_set = ["1120x832"]  #, "960x720", "640x480",  "480x352", "320x240"]   # for openPose models [720, 600, 480, 360, 240]   # [240] #     # [240]       # [720, 600, 480, 360, 240]    #   [720]     # [720, 600, 480, 360, 240]  #  [720]    # [720, 600, 480, 360, 240]            #  16: 9
    frame_set = [25, 15, 10, 5, 2, 1]     #  [25, 10, 5, 2, 1]    # [30],  [30, 10, 5, 2, 1] 
    model_set = ['cmu']   #, 'mobilenet_v2_small']

    lst_id_subconfig, id_config_dict = extract_specific_config_name_from_file(data_pose_keypoint_dir, resolution_set, frame_set, model_set)

    print ("getOnePersonFeatureInputOutput01 config_ind_pareto: ", lst_id_subconfig)
    acc_frame_arr = acc_frame_arr[lst_id_subconfig, :]
    spf_frame_arr =  spf_frame_arr[lst_id_subconfig, :]
    
    print ("getOnePersonFeatureInputOutput01 acc_frame_arr: ", acc_frame_arr[:, 0], acc_frame_arr.shape)

    # select one person, i.e. no 0
    
    #max_frame_example_used = 1000   # 8000
    #current_frame_id = 25
    config_id_dict, id_config_dict = read_all_config_name_from_file(data_pose_keypoint_dir, False)

    
    # get new id map based on pareto boundary/'s result
    new_id_config_dict = defaultdict()
    for i, ind in enumerate(lst_id_subconfig):
        new_id_config_dict[i] = id_config_dict[ind]
    id_config_dict = new_id_config_dict
    
    print ("config_id_dict: ", len(config_id_dict), id_config_dict)
    
    
    # only read the most expensive config
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*1120x832_25_cmu_estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection

    print ("filePath: ", filePathLst[0], len(df_det))

    
    history_pose_est_arr = np.zeros((max_frame_example_used, COCO_KP_NUM, 2)) # np.zeros((len(df_det), COCO_KP_NUM, 2))        #  to make not shift when new frames comes, we store all values
    
    previous_frm_indx = 0
    
    
    input_x_arr = np.zeros((max_frame_example_used, 66, 2))       # 17 + 4*4 + 4*4 + 17
    y_out_arr = np.zeros((max_frame_example_used+1), dtype=int)
    
    #current_instance_start_video_path_arr = np.zeros(max_frame_example_used, dtype=int)
    current_instance_start_frm_path_arr = np.zeros(max_frame_example_used, dtype=object)
    
    prev_EMA_speed_arr = np.zeros((COCO_KP_NUM, 2))
    
    prev_EMA_relative_speed_arr2 = np.zeros((4, 2))        # only get 4 keypoints
    prev_EMA_relative_speed_arr3 = np.zeros((4, 2))        # only get 4 keypoints
    prev_EMA_relative_speed_arr4 = np.zeros((4, 2))        # only get 4 keypoints
    prev_EMA_relative_speed_arr5 = np.zeros((4, 2))        # only get 4 keypoints
    
    
    reso_feature_arr = np.zeros(max_frame_example_used, dtype=int)
    history_reso_arr = np.zeros(max_frame_example_used)
    prev_reso_aver = 0.0
    
    config_feature_arr = np.zeros(max_frame_example_used, dtype=int)
    history_config_arr = np.zeros(max_frame_example_used) # + config_ind_pareto[0]
    prev_config_aver = 0.0
    
    frmRt_feature_arr = np.zeros(max_frame_example_used)
    history_frmRt_arr = np.zeros(max_frame_example_used)
    prev_frmRt_aver = 0.0
    
    
    blurriness_feature_arr = np.zeros((max_frame_example_used, 1))
    
    frm_id_debug_only_arr = np.zeros(max_frame_example_used)

    select_frm_cnt = 0
    skipped_frm_cnt = 0

    switching_config_skipped_frm = -1
    switching_config_inter_skip_cnts = 0
    
    selected_configs_acc_lst = blist()      # check the selected config's for all frame's lst, in order to get the accuracy
    
    #video_id = int(data_pose_keypoint_dir.split('/')[-2].split('_')[1])
    
    #arr_feature_frameIndex = np.zeros(max_frame_example_used)        # defaultdict(int) each interval's starting point -- corresponding frame index
    
    for index, row in df_det.iterrows():  
        #print ("index, row: ", index, row)
        #reso = row['Resolution']
        #frm_rate = row['Frame_rate']
        #model = row['Model']
        #num_humans = row['numberOfHumans']        # number of human detected

        if index >= acc_frame_arr.shape[1]:
            break            
        
        imgPath = row['Image_path']
        current_iterative_frm_id = int(imgPath.split('/')[-1].split('.')[0])
        

        est_res = row['Estimation_result']
        reso = row['Resolution']
        width = int(reso.split('x')[0])
        height = int(reso.split('x')[1])
        if str(est_res) == 'nan':  # here select_frm_cnt does not increase
            skipped_frm_cnt += 1
            print ("nan nan est_res: ", est_res, index, select_frm_cnt)
            continue
        # skipping interval by frame_rate
        if ((switching_config_skipped_frm != - 1) and (switching_config_skipped_frm < (switching_config_inter_skip_cnts-skipped_frm_cnt))):   # switching_config_inter_skip_cnts:
            switching_config_skipped_frm += 1
            continue
        #print ("frm_id num_humans, ", reso, model, frm_id)
            
        kp_arr = getPersonEstimation(est_res, width, height)
        #history_pose_est_dict[previous_frm_indx] = kp_arr
         
        history_pose_est_arr[previous_frm_indx] = kp_arr
        #print ("kp_arr, ", kp_arr)
        #break    # debug only
        if previous_frm_indx >= history_frame_num:
            #print ("previous_frm_indx, ", previous_frm_indx, index)
            
            current_cofig = id_config_dict[int(y_out_arr[select_frm_cnt])]
            curr_frm_rate = int(current_cofig.split('-')[1])
            #print ("xxxx current_cofig: ", current_cofig, curr_frm_rate)
            # calculate the human moving speed feature (1)
            feature1_speed_arr = getFeatureOnePersonMovingSpeed(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_speed_arr)
            
            prev_EMA_speed_arr = feature1_speed_arr
            #calculate the relative moving speed feature (2)
            feature2_relative_speed_arr = getFeatureOnePersonRelativeSpeed1(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr2)
            prev_EMA_relative_speed_arr2 = feature2_relative_speed_arr
            #print ("feature1_speed_arr feature2_relative_speed_arr, ", feature1_speed_arr.shape, feature2_relative_speed_arr.shape)
            
            feature3_relative_speed_arr = getFeatureOnePersonRelativeSpeed2(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr3)
            prev_EMA_relative_speed_arr3 = feature3_relative_speed_arr

            feature4_relative_speed_arr = getFeatureOnePersonRelativeSpeed3(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr4)
            prev_EMA_relative_speed_arr4 = feature4_relative_speed_arr
            
            feature5_relative_speed_arr = getFeatureOnePersonRelativeSpeed4(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr5)
            prev_EMA_relative_speed_arr5 = feature5_relative_speed_arr
            
            
            current_frame_relative_distance_arr = getFeatureRelativeDistanceClosenessKeyPoint(history_pose_est_arr, select_frm_cnt)
            
            cameraDistance_feature = getFeatureDistanceToCamera(history_pose_est_arr, select_frm_cnt)
                
            total_features_arr = np.vstack((feature1_speed_arr, feature2_relative_speed_arr))
            #print ("total_features_arr total_features_arr, ", frm_id,  total_features_arr.shape)
            #print ("total_features_arr1: ", total_features_arr.shape, feature3_relative_speed_arr.shape)
            
            total_features_arr = np.vstack((total_features_arr, feature3_relative_speed_arr))
            #print ("total_features_arr2: ", total_features_arr.shape, input_x_arr.shape)

            total_features_arr = np.vstack((total_features_arr, feature4_relative_speed_arr))
            
            total_features_arr = np.vstack((total_features_arr, feature5_relative_speed_arr))

            total_features_arr = np.vstack((total_features_arr, current_frame_relative_distance_arr))
            
            total_features_arr = np.vstack((total_features_arr, cameraDistance_feature))
            #print ("total_features_arr2: ", total_features_arr.shape, input_x_arr.shape)
            
            input_x_arr[select_frm_cnt]= total_features_arr  #  input_x_arr[frm_id-1] = total_features_arr
            
            #arr_feature_frameIndex[select_frm_cnt] = index+1          # frame_index because this start from 1 ***.jpg
            #print ("total_features_arr: ", frm_id-1)
            #previous_frm_indx = 1
                
            #y_out_arr[select_frm_cnt+1] = select_config(acc_frame_arr, spf_frame_arr,  select_frm_cnt+1+switching_config_inter_skip_cnts+skipped_frm_cnt, minAccuracy)
            y_out_arr[select_frm_cnt+1] = select_config(acc_frame_arr, spf_frame_arr,  current_iterative_frm_id-1, minAccuracy)
            current_cofig = id_config_dict[int(y_out_arr[select_frm_cnt])]
            
            selected_config_acc = acc_frame_arr[y_out_arr[select_frm_cnt], index]
            
            selected_configs_acc_lst.append(selected_config_acc)
            #print ("current_cofig: ", current_cofig)
            
            current_config_frmRt = int(current_cofig.split('-')[1])
            
            if switching_config_skipped_frm == - 1:
                switching_config_inter_skip_cnts = PLAYOUT_RATE-1 #  math.ceil(PLAYOUT_RATE/current_config_frmRt)-2  #       #math.ceil(PLAYOUT_RATE/frmRt)-1
            else:
                switching_config_inter_skip_cnts = PLAYOUT_RATE  # math.ceil(PLAYOUT_RATE/current_config_frmRt)-1  # PLAYOUT_RATE
                
            reso = int(current_cofig.split('-')[0].split('x')[1])

            history_reso_arr[select_frm_cnt] = reso
                
            curr_reso_aver= getConfigFeature(history_reso_arr, select_frm_cnt, prev_reso_aver)
            
            prev_reso_aver = curr_reso_aver
            
            reso_feature_arr[select_frm_cnt] =  curr_reso_aver # curr_reso_aver y_out_arr[index]    #curr_reso_aver
            #print ("current_cofig reso: ", current_cofig, reso, curr_reso_aver)


            history_config_arr[select_frm_cnt] = y_out_arr[select_frm_cnt]
            curr_config_aver= getConfigFeature(history_config_arr, select_frm_cnt, prev_config_aver)
            
            prev_config_aver = curr_config_aver
            
            config_feature_arr[select_frm_cnt] =  curr_config_aver  # y_out_arr[select_frm_cnt]     # curr_config_aver  #   # y_out_arr[select_frm_cnt]  #   #
            
            
            frmRt = int(current_cofig.split('-')[1])

            history_frmRt_arr[select_frm_cnt] = frmRt
                
            curr_frmRt_aver= getFrmRateFeature(history_frmRt_arr, prev_frmRt_aver)
            prev_frmRt_aver = curr_frmRt_aver
            
            
            #frmRt_feature_arr[select_frm_cnt] = curr_frmRt_aver
            
            #current_iterative_frm_id = previous_frm_indx + skipped_frm_cnt + switching_config_skipped_frm
            #lowest_config_id = [19, 10, 6, 2] 
            #stTime = time.time()
            blurrinessScore_arr =  getBlurrinessFeature(imgPath, current_iterative_frm_id-1)
            #print ("end TimeTimeTime: ", time.time()-stTime)
            blurriness_feature_arr[select_frm_cnt] = blurrinessScore_arr
            
            
            frm_id_debug_only_arr[select_frm_cnt] = current_iterative_frm_id
            
            #current_instance_start_video_path_arr[select_frm_cnt] = video_id
            current_instance_start_frm_path_arr[select_frm_cnt] = row['Image_path']
                
            skipped_frm_cnt = 0       
            select_frm_cnt += 1 
            switching_config_skipped_frm = 1
            

        previous_frm_indx += 1
        
        #how many are used for traing, validation, and test
        if previous_frm_indx > (max_frame_example_used-1):
            break 
        
    
    #arr_feature_frameIndex = arr_feature_frameIndex[history_frame_num:select_frm_cnt]
    input_x_arr = input_x_arr[history_frame_num:select_frm_cnt].reshape(input_x_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    
    current_instance_start_frm_path_arr = current_instance_start_frm_path_arr[history_frame_num:select_frm_cnt].reshape(current_instance_start_frm_path_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((current_instance_start_frm_path_arr, input_x_arr))
    
    # add each instance's starting frame's path
    #current_instance_start_video_path_arr = current_instance_start_video_path_arr[history_frame_num:select_frm_cnt].reshape(current_instance_start_video_path_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    #input_x_arr = np.hstack((current_instance_start_video_path_arr, input_x_arr))
    
    frmRt_feature_arr = frmRt_feature_arr[history_frame_num:select_frm_cnt].reshape(frmRt_feature_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((input_x_arr, frmRt_feature_arr))
    
    reso_feature_arr = reso_feature_arr[history_frame_num:select_frm_cnt].reshape(reso_feature_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((input_x_arr, reso_feature_arr))
   
    config_feature_arr = config_feature_arr[history_frame_num:select_frm_cnt].reshape(config_feature_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((input_x_arr, config_feature_arr))
    
    blurriness_feature_arr = blurriness_feature_arr[history_frame_num:select_frm_cnt].reshape(blurriness_feature_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((input_x_arr, blurriness_feature_arr))
        
    #frm_id_debug_only_arr = frm_id_debug_only_arr[history_frame_num:select_frm_cnt].reshape(frm_id_debug_only_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    #input_x_arr = np.hstack((input_x_arr, frm_id_debug_only_arr))
            
    y_out_arr = y_out_arr[history_frame_num+1:select_frm_cnt+1]
    #print ("reso_feature_arr, ", reso_feature_arr.shape, reso_feature_arr)
    print ("feature1_speed_arr, ", input_x_arr.shape, y_out_arr.shape, feature1_speed_arr.shape, feature2_relative_speed_arr.shape)

    #checkCorrelationPlot(data_pose_keypoint_dir, input_x_arr, y_out_arr, id_config_dict)
    return input_x_arr, y_out_arr, id_config_dict


def select_config(acc_frame_arr, spf_frame_arr, index_id, minAccuracy):
    '''
    need to use frm_id-1, index start from 0
    '''   
    
    #print ("[:, frm_id-1]:", acc_frame_arr.shape, index_id)
    
    indx_config_above_minAcc = np.where(acc_frame_arr[:, index_id] >= minAccuracy)      # the index of the config above the threshold minAccuracy
    #print("indx_config_above_minAcc: ", indx_config_above_minAcc, len(indx_config_above_minAcc[0]))
        
    cpy_minAccuracy = minAccuracy
    # in case no profiling config found satisfying the minAcc
    while len(indx_config_above_minAcc[0]) == 0:
        cpy_minAccuracy = cpy_minAccuracy - 0.05 
        indx_config_above_minAcc = np.where(acc_frame_arr[:, index_id] >= cpy_minAccuracy)      # the index of the config above the threshold minAccuracy
            
    #print ("indx_config_above_minAcc:", indx_config_above_minAcc)
    tmp_config_indx = np.argmin(spf_frame_arr[indx_config_above_minAcc, index_id])   # selected the minimum spf, i.e. the fastest processing speed
    #print ("tmp_config_indx tmp_config_indx:", tmp_config_indx )
    selected_config_indx = indx_config_above_minAcc[0][tmp_config_indx]      # final selected indx from all config_indx
    #print ("final selected_config_indx:",selected_config_indx, spf_frame_arr[selected_config_indx, frm_id-1] )

    return selected_config_indx

'''
def getGroundTruthY(data_pickle_dir, max_frame_example_used, history_frame_num):
    #this dataset Y

    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)
    frm_id = 26
    minAccuracy = 0.85
    y_out_arr = np.zeros((max_frame_example_used+1), dtype=int)
    for frm_id in range(0, max_frame_example_used+1):
        y_out_arr[frm_id] = select_config(acc_frame_arr, spf_frame_arr, frm_id, minAccuracy)
    
    print ("y_out_arr original:", y_out_arr.shape)
    return y_out_arr
'''

def getDataExamples():
    
    video_dir_lst = ['output_001-dancing-10mins/', 'output_006-cardio_condition-20mins/', 'output_008-Marathon-20mins/'
                    ]   
    
    for video_dir in video_dir_lst[1:2]:    #   #[2:3]:  # [1:2]:  # [0:1]:   # [2:3]:   # [0:1]:  # [1:2]:    # [0:1]:        #[1:2]:
        
        data_pose_keypoint_dir =  dataDir2 + video_dir
        
        history_frame_num = 1  #1          # 
        max_frame_example_used =  8025 # 10000 #8025   # 8000
        data_pickle_dir = dataDir2 + video_dir + 'frames_pickle_result/'
        minAccuracy = 0.85

        x_input_arr, y_out_arr = getOnePersonFeatureInputOutput01(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
        #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput02(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
        #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput03(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
        #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput04(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
        
        #y_out_arr = getGroundTruthY(data_pickle_dir, max_frame_example_used, history_frame_num)
        x_input_arr = x_input_arr.reshape((x_input_arr.shape[0], -1))
        
        # add current config as a feature
        print ("combined before:",x_input_arr.shape, y_out_arr[history_frame_num:-1].shape)
        #current_config_arr = y_out_arr[history_frame_num:-1].reshape((y_out_arr[history_frame_num:-1].shape[0], -1))
        #x_input_arr = np.hstack((x_input_arr, current_config_arr))
        
        #y_out_arr = y_out_arr[history_frame_num+1:]
        
        print ("y_out_arr shape after:", x_input_arr.shape, y_out_arr.shape)
        
        #data_examples_arr = np.hstack((x_input_arr, y_out_arr))
        
        
        out_frm_examles_pickle_dir = data_pose_keypoint_dir + "data_examples_files/" 
        if not os.path.exists(out_frm_examles_pickle_dir):
                os.mkdir(out_frm_examles_pickle_dir)
                
        with open(out_frm_examles_pickle_dir + "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
            pickle.dump(x_input_arr, fs)
            
        
        with open(out_frm_examles_pickle_dir + "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
            pickle.dump(y_out_arr, fs)
    

if __name__== "__main__": 
            
    
    getDataExamples()
            




