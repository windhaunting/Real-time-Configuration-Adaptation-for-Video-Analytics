#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:58:13 2019

@author: fubao
"""


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
import math
import cv2

import numpy as np

from glob import glob
from blist import blist
from common_classifier import read_all_config_name_from_file
from common_classifier import read_poseEst_conf_frm
from common_classifier import readProfilingResultNumpy
from common_classifier import get_cmu_model_config_acc_spf
from common_classifier import paddingZeroToInter
from common_classifier import COCO_KP_NUM
from common_classifier import getPersonEstimation

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir3
from profiling.common_prof import frameRates
from profiling.common_prof import PLAYOUT_RATE



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
  
    

def getEuclideanDist(val, time_frm_interval):
    '''
    calculate the euclidean distance
    val: [x1, y1, v1 x2, y2, v2]
    '''
    #or vector distance    
    #speed_angle = [val[0]-val[2], val[1]-val[3]]
    x1= val[0]
    y1 = val[1]
    x2 = val[2]
    y2 = val[3]
    
    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
               
    speed = dist/time_frm_interval
    
    if (math.sqrt(x1**2 + y1*2) *math.sqrt(x2**2 + y2**2)) == 0:
        cos_angle = 0
    else:
        cos_angle = (x1*x2 + y1*y2) / (math.sqrt(x1**2 + y1**2) *math.sqrt(x2**2 + y2**2))
    
    speed_angle = [speed, cos_angle]       # 2 is visibility not used
    #print ("speed: ", val, speed)

    
    return speed_angle
    

def fillEstimation(j, prev_frm_est_arr, cur_frm_est_arr):
    
    num_kp = prev_frm_est_arr.shape[0]
    
    for i in range(0, num_kp):
        if cur_frm_est_arr[i][2] == 0:     # visibility is 0, not detected
            #print ("prev_frm_est_arr: ", j, i, prev_frm_est_arr[i], cur_frm_est_arr[i][2])
            cur_frm_est_arr[i] = prev_frm_est_arr[i]
            
    return cur_frm_est_arr


    
def getCurrentAverageSpeed(history_pose_est_arr, used_current_frm):
    
    
    used_prev_frm = used_current_frm - (PLAYOUT_RATE)   # 1sec
    
    j = used_current_frm
    
    current_speed_angle_arr = np.zeros(history_pose_est_arr.shape[1:])
    while (j > used_prev_frm):
        cur_frm_est_arr = history_pose_est_arr[j]    
        prev_frm_est_arr = history_pose_est_arr[j-1]
        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        hstack_arr = np.hstack((cur_frm_est_arr, prev_frm_est_arr))
        
        #print ("current_speed_angle_arr: ", current_speed_angle_arr.shape, hstack_arr.shape)
        time_frm_interval = 1.0       # 1 sec  #(1.0/PLAYOUT_RATE)/2            # interval 1 sec
        
        current_speed_angle_arr += np.apply_along_axis(getEuclideanDist, 1, hstack_arr, time_frm_interval)    
        j -= 1
    
    current_speed_angle_arr = current_speed_angle_arr/(used_current_frm - used_prev_frm)
    
    return current_speed_angle_arr


def getCurrentAverageRelativeSpeed(history_pose_est_arr, used_current_frm):
    
    
    used_prev_frm = used_current_frm-PLAYOUT_RATE   # previous 1 sec
    
    j = used_current_frm
    
    current_speed_angle_arr = np.zeros(history_pose_est_arr.shape[1:])
    while (j > used_prev_frm):
        cur_frm_est_arr = history_pose_est_arr[j]    
        prev_frm_est_arr = history_pose_est_arr[j-1]
        time_frm_interval =  1.0 # (1.0/PLAYOUT_RATE)          # interval 1 frame
        current_speed_angle_arr = relativeSpeed(cur_frm_est_arr, prev_frm_est_arr, time_frm_interval)
        j -= 1
    
    current_speed_angle_arr = current_speed_angle_arr/(used_current_frm - used_prev_frm)
    return current_speed_angle_arr

    
def getFeatureOnePersonMovingSpeed(history_pose_est_arr, used_current_frm, prev_EMA_speed_arr):
    '''
    feature1: One person’s moving speed of all keypoints V i,k based on the
    euclidean d distance of current frame with the previous frame {f j−m , m =
                                                                   1, 2, ..., 24
    only get the previous history_frame_num, that is current_frm_id is the current frame id
    the previous 24 frames are the previous frames.
    
    output: feature1_speed_arr COCO_KP_NUM x 1
    '''
    '''
    used_prev_frm = used_current_frm-1
    cur_frm_est_arr = history_pose_est_arr[used_current_frm]    
    #feature1_speed_arr = np.zeros((COCO_KP_NUM, 2))
        
    prev_frm_est_arr = history_pose_est_arr[used_prev_frm]

    #print ("getFeatureOnePersonMovingSpeed cur_frm_est_arr ", prev_frm_est_arr, cur_frm_est_arr)
    
    # get the current speed based on current frame and previous frame
    hstack_arr = np.hstack((cur_frm_est_arr, prev_frm_est_arr))
    #frmInter = math.ceil(PLAYOUT_RATE/curr_frm_rate)          # frame rate sampling frames in interval, +1 every other

    time_frm_interval = (1.0/PLAYOUT_RATE)*(used_current_frm - used_prev_frm)            # interval 1 second
    #time_frm_interval = 1.0/PLAYOUT_RATE
    current_speed_angle_arr = np.apply_along_axis(getEuclideanDist, 1, hstack_arr, time_frm_interval)    
    '''
    current_speed_angle_arr = getCurrentAverageSpeed(history_pose_est_arr, used_current_frm)
    #print ("current_speed_angle_arr: ", current_speed_angle_arr)
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
    

def getFeatureOnePersonRelativeSpeed1(history_pose_est_arr, used_current_frm, prev_EMA_relative_speed_arr):
    '''
    feature 2 One person arm/feet’s relative speed to the torso of all keypoints
    with the previous frames.
    In each frame, we select the relative distance of left wrist, right wrist,left ankle
    and right ankle to the left hip.
    
    output: feature2_relative_speed_arr 5 x history_frame_num-1

    '''

    # get left wrist, right wrist,left ankle, right ankle and the left hip
    # which corresponds to [7, 4, 13, 10, 11]
    selected_kp_index = [9, 10, 13, 14, 11]   #  [7, 4, 13, 10, 11]
    selected_kp_history_pose_est_arr = history_pose_est_arr[:, selected_kp_index, :]
    #print ("selected_kp_history_pose_est_arr aaa, ", selected_kp_history_pose_est_arr[0])
    
    '''
    used_prev_frm = used_current_frm-1
    curr_frm_rel_dist_arr = selected_kp_history_pose_est_arr[used_current_frm]   #np.apply_along_axis(relativeDistance, 2, selected_kp_history_pose_est_arr)
    
    previous_frm_arr = selected_kp_history_pose_est_arr[used_prev_frm]

    #frmInter = math.ceil(PLAYOUT_RATE/curr_frm_rate)          # frame rate sampling frames in interval, +1 every other

    time_frm_interval = (1.0/PLAYOUT_RATE)* (used_current_frm - used_prev_frm)
    
    current_speed_arr = relativeSpeed(curr_frm_rel_dist_arr, previous_frm_arr, time_frm_interval)
    '''
    
    current_speed_arr = getCurrentAverageRelativeSpeed(selected_kp_history_pose_est_arr, used_current_frm)

    feature1_relative_speed_arr = current_speed_arr * ALPHA_EMA  + (1-ALPHA_EMA) * prev_EMA_relative_speed_arr
    
    return feature1_relative_speed_arr


def getFeatureOnePersonRelativeSpeed2(history_pose_est_arr, used_current_frm, prev_EMA_relative_speed_arr):
    '''
    feature 2 One person arm/feet’s relative speed to the torso of all keypoints
    with the previous frames.
    In each frame, we select the relative distance of left wrist, right wrist,left ankle
    and right ankle to the right hip.
    
    output: feature2_relative_speed_arr 5 x history_frame_num-1

    '''

    # get left wrist, right wrist,left ankle, right ankle and the left hip
    # which corresponds to [7, 4, 13, 10, 11]
    selected_kp_index = [9, 10, 13, 14, 12]        # [7, 4, 13, 10, 8]
    selected_kp_history_pose_est_arr = history_pose_est_arr[:, selected_kp_index, :]
    #print ("selected_kp_history_pose_est_arr aaa, ", selected_kp_history_pose_est_arr[0])
    '''
    used_prev_frm = used_current_frm-1
    curr_frm_rel_dist_arr = selected_kp_history_pose_est_arr[used_current_frm]   #np.apply_along_axis(relativeDistance, 2, selected_kp_history_pose_est_arr)
    
    previous_frm_arr = selected_kp_history_pose_est_arr[used_prev_frm]

    #frmInter = math.ceil(PLAYOUT_RATE/curr_frm_rate)          # frame rate sampling frames in interval, +1 every other

    time_frm_interval = (1.0/PLAYOUT_RATE)* (used_current_frm - used_prev_frm)
    
    current_speed_arr = relativeSpeed(curr_frm_rel_dist_arr, previous_frm_arr, time_frm_interval)
    '''
    current_speed_arr = getCurrentAverageRelativeSpeed(selected_kp_history_pose_est_arr, used_current_frm)
    
    feature2_relative_speed_arr = current_speed_arr * ALPHA_EMA  + (1-ALPHA_EMA) * prev_EMA_relative_speed_arr
    
    return feature2_relative_speed_arr


def getFeatureOnePersonRelativeSpeed3(history_pose_est_arr, used_current_frm, prev_EMA_relative_speed_arr):
    '''
    feature 2 One person arm/feet’s relative speed to the torso of all keypoints
    with the previous frames.
    In each frame, we select the relative distance of left wrist, right wrist,left ankle
    and right ankle to the left shoulder.
    
    output: feature2_relative_speed_arr 5 x history_frame_num-1

    '''

    # get left wrist, right wrist,left ankle, right ankle and the left hip
    # which corresponds to [7, 4, 13, 10, 11]
    selected_kp_index = [9, 10, 13, 14, 5]    # [7, 4, 13, 10, 1]
    selected_kp_history_pose_est_arr = history_pose_est_arr[:, selected_kp_index, :]
    #print ("selected_kp_history_pose_est_arr aaa, ", selected_kp_history_pose_est_arr[0])
    '''
    used_prev_frm = used_current_frm-1

    curr_frm_rel_dist_arr = selected_kp_history_pose_est_arr[used_current_frm]   #np.apply_along_axis(relativeDistance, 2, selected_kp_history_pose_est_arr)
    
    previous_frm_arr = selected_kp_history_pose_est_arr[used_prev_frm]

    #frmInter = math.ceil(PLAYOUT_RATE/curr_frm_rate)          # frame rate sampling frames in interval, +1 every other

    time_frm_interval = (1.0/PLAYOUT_RATE)* (used_current_frm - used_prev_frm)
    
    current_speed_arr = relativeSpeed(curr_frm_rel_dist_arr, previous_frm_arr, time_frm_interval)
    '''
    
    current_speed_arr = getCurrentAverageRelativeSpeed(selected_kp_history_pose_est_arr, used_current_frm)

    feature3_relative_speed_arr = current_speed_arr * ALPHA_EMA  + (1-ALPHA_EMA) * prev_EMA_relative_speed_arr
    
    return feature3_relative_speed_arr

def getFeatureOnePersonRelativeSpeed4(history_pose_est_arr, used_current_frm, prev_EMA_relative_speed_arr):
    '''
    feature 2 One person arm/feet’s relative speed to the torso of all keypoints
    with the previous frames.
    In each frame, we select the relative distance of left wrist, right wrist,left ankle
    and right ankle to the right shoulder.
    
    output: feature2_relative_speed_arr 5 x history_frame_num-1

    '''

    # get left wrist, right wrist,left ankle, right ankle and the left hip
    # which corresponds to [7, 4, 13, 10, 11]
    selected_kp_index = [9, 10, 13, 14, 6]    # [7, 4, 13, 10, 1]
    selected_kp_history_pose_est_arr = history_pose_est_arr[:, selected_kp_index, :]
    #print ("selected_kp_history_pose_est_arr aaa, ", selected_kp_history_pose_est_arr[0])
    '''
    used_prev_frm = used_current_frm-1
    curr_frm_rel_dist_arr = selected_kp_history_pose_est_arr[used_current_frm]   #np.apply_along_axis(relativeDistance, 2, selected_kp_history_pose_est_arr)
    
    previous_frm_arr = selected_kp_history_pose_est_arr[used_prev_frm]

    #frmInter = math.ceil(PLAYOUT_RATE/curr_frm_rate)          # frame rate sampling frames in interval, +1 every other

    time_frm_interval = (1.0/PLAYOUT_RATE)* (used_current_frm - used_prev_frm)
    
    current_speed_arr = relativeSpeed(curr_frm_rel_dist_arr, previous_frm_arr, time_frm_interval)
    '''
    
    current_speed_arr = getCurrentAverageRelativeSpeed(selected_kp_history_pose_est_arr, used_current_frm)

    feature4_relative_speed_arr = current_speed_arr * ALPHA_EMA  + (1-ALPHA_EMA) * prev_EMA_relative_speed_arr
    
    return feature4_relative_speed_arr




def getFrmRateFeature(history_frmRt_arr, select_frm_cnt, prev_frmRt_aver):
    
    
    if select_frm_cnt < 2:
        current_frmRt_aver = np.mean(history_frmRt_arr[:select_frm_cnt+1])
    else:
        current_frmRt_aver = np.mean(history_frmRt_arr[select_frm_cnt-1:select_frm_cnt])
    feature_frmRate = current_frmRt_aver * ALPHA_EMA  + (1-ALPHA_EMA) * prev_frmRt_aver

    return feature_frmRate



def getConfigFeature(history_config_arr, select_frm_cnt, prev_config_aver):
    
    
    if select_frm_cnt < 2:
        current_confg_aver = np.mean(history_config_arr[:select_frm_cnt+1])
    else:
        current_confg_aver = np.mean(history_config_arr[select_frm_cnt-1:select_frm_cnt])
    #print ("CCCCCCCC:" , history_config_arr, prev_config_aver)
    feature_confg = int(current_confg_aver * ALPHA_EMA  + (1-ALPHA_EMA) * prev_config_aver)

    return feature_confg



def getResoFeature(history_reso_arr, select_frm_cnt, prev_reso_aver):
    
    
    if select_frm_cnt < 2:
        current_reso_aver = np.mean(history_reso_arr[:select_frm_cnt+1])
    else:
        current_reso_aver = np.mean(history_reso_arr[select_frm_cnt-1:select_frm_cnt])
    feature_Reso = int(current_reso_aver * ALPHA_EMA  + (1-ALPHA_EMA) * prev_reso_aver)

    return feature_Reso



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
    
    image = cv2.imread(imagePath)
    if image is None:
        return 250         # based on a video obervation, if image is not readable, assume it's not blurry
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurrinessScore_arr = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return blurrinessScore_arr


#def select_part_config_frmRateOnly():
    
def getOnePersonFeatureInputOutputAll001(data_pose_keypoint_dir, data_pickle_dir, data_frame_path_dir, history_frame_num, max_frame_example_used, minAccuracy, minDelayTreshold):
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
    
    interval_frms = 1*PLAYOUT_RATE       # interval 1 sec for switching
    intervalFlag = 'sec'
    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir, intervalFlag)
    
    confg_est_frm_arr = read_poseEst_conf_frm(data_pickle_dir)
    
    #old_acc_frm_arr = acc_frame_arr
    #print ("getOnePersonFeatureInputOutput01 acc_frame_arr: ", acc_frame_arr.shape)

    # save to file to have a look
    #outDir = data_pose_keypoint_dir + "classifier_result/"
    #np.savetxt(outDir + "accuracy_above_threshold" + str(minAccuracy) + ".tsv", acc_frame_arr[:, :5000], delimiter="\t")
    
    
    #config_ind_pareto = getParetoBoundary(acc_frame_arr[:, 0], spf_frame_arr[:, 0])
    #resolution_set = ["1120x832", "960x720", "640x480",  "480x352", "320x240"]   # for openPose models [720, 600, 480, 360, 240]   # [240] #     # [240]       # [720, 600, 480, 360, 240]    #   [720]     # [720, 600, 480, 360, 240]  #  [720]    # [720, 600, 480, 360, 240]            #  16: 9
    
    acc_frame_arr, spf_frame_arr, id_config_dict = get_cmu_model_config_acc_spf(data_pickle_dir, data_pose_keypoint_dir)
    
  
    print ("config_id_dict: ", len(id_config_dict), acc_frame_arr.shape, spf_frame_arr.shape)
    
    
    # only read the most expensive config
    #filePathLst = sorted(glob(data_pose_keypoint_dir + "*1120x832_25_cmu_estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    #df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection
    #print ("filePath: ", filePathLst[0], len(df_det))

    current_delay = 0.0

    previous_frm_indx = 0
    
    
    input_x_arr = np.zeros((max_frame_example_used, 33, 2))    # np.zeros((max_frame_example_used, 66, 2))       # 17 + 4*4 + 4*4 + 17
    y_out_arr = np.zeros((max_frame_example_used+1), dtype=int)
    
    #current_instance_start_video_path_arr = np.zeros(max_frame_example_used, dtype=int)
    current_instance_start_frm_path_arr = np.zeros(max_frame_example_used, dtype=object)
    
    prev_EMA_speed_arr = np.zeros((COCO_KP_NUM, 2))
    
    prev_EMA_relative_speed_arr1 = np.zeros((4, 2))        # only get 4 keypoints
    prev_EMA_relative_speed_arr2 = np.zeros((4, 2))        # only get 4 keypoints
    prev_EMA_relative_speed_arr3 = np.zeros((4, 2))        # only get 4 keypoints
    prev_EMA_relative_speed_arr4 = np.zeros((4, 2))        # only get 4 keypoints
    
    
    reso_feature_arr = np.zeros(max_frame_example_used, dtype=int)
    history_reso_arr = np.zeros(max_frame_example_used)
    prev_reso_aver = 0.0
    
    config_feature_arr = np.zeros(max_frame_example_used, dtype=int)
    history_config_arr = np.zeros(max_frame_example_used) # + config_ind_pareto[0]
    prev_config_aver = 0.0
    
    frmRt_feature_arr = np.zeros(max_frame_example_used)
    history_frmRt_arr = np.zeros(max_frame_example_used)
    prev_frmRt_aver = 0.0
    
    
    delay_feature_arr = np.zeros(max_frame_example_used)

    blurriness_feature_arr = np.zeros((max_frame_example_used, 1))
    
    frm_id_debug_only_arr = np.zeros(max_frame_example_used)

    select_frm_cnt = 0
    skipped_frm_cnt = 0

    switching_config_skipped_frm = -1
    switching_config_inter_skip_cnts = 0
    
    selected_configs_acc_lst = blist()      # check the selected config's for all frame's lst, in order to get the accuracy
    
    #video_id = int(data_pose_keypoint_dir.split('/')[-2].split('_')[1])
    
    #arr_feature_frameIndex = np.zeros(max_frame_example_used)        # defaultdict(int) each interval's starting point -- corresponding frame index
    if intervalFlag == 'sec':
        interval_jump = PLAYOUT_RATE        # one sec  25 frames
    elif intervalFlag == 'frame':
        interval_jump = 1                     # one frame
        
    num_frames = acc_frame_arr.shape[1]
    
    print ("num_frames :", num_frames)
    
    history_pose_est_arr = np.zeros((max_frame_example_used, COCO_KP_NUM, 3)) # np.zeros((len(df_det), COCO_KP_NUM, 2))        #  to make not shift when new frames comes, we store all values
    #first frame is not used?
    for j in range(1, min(max_frame_example_used, confg_est_frm_arr.shape[1])):
        #print ("est_res pre, ", confg_est_frm_arr[0][j])
        est_res = confg_est_frm_arr[0][j]              # only most expensive config to calculate the feature
        
        if j == 1 and str(confg_est_frm_arr[0][j-1]) == 'nan':
            confg_est_frm_arr[0][j] = np.zeros((COCO_KP_NUM, 3))
            
        elif str(est_res) == 'nan':
            #print("eeeeest_res: ", j, est_res)
            history_pose_est_arr[j] = history_pose_est_arr[j-1]
            confg_est_frm_arr[0][j] = confg_est_frm_arr[0][j-1]
        else:
       
    
            history_pose_est_arr[j] = getPersonEstimation(est_res)
            history_pose_est_arr[j] = fillEstimation(j, history_pose_est_arr[j-1], history_pose_est_arr[j])

            tmp_arr_est = history_pose_est_arr[j].reshape(1, -1)
            tmp_arr_est = str(tmp_arr_est.tolist()[0]) + ',' + str(confg_est_frm_arr[0][j].split('],')[-1])
            confg_est_frm_arr[0][j] = tmp_arr_est
        
        
        '''
        tmp_arr_est = history_pose_est_arr[j].reshape(1, -1)
        print ("confg_est_frm_arr[0][j], ", confg_est_frm_arr[0][j])
        if str(confg_est_frm_arr[0][j]) == 'nan':
            tmp_arr_est = np.zeros((COCO_KP_NUM, 3))
        else:
            tmp_arr_est = str(tmp_arr_est.tolist()[0]) + ',' + str(confg_est_frm_arr[0][j].split('],')[-1])
        '''
        #print ("est_res after, ", confg_est_frm_arr[0][j])
        
    history_pose_est_arr = history_pose_est_arr[:, :, :2]
    #history_pose_est_arr = confg_est_frm_arr
    
    for j in range(1, min(max_frame_example_used, confg_est_frm_arr.shape[1])):
        est_res = confg_est_frm_arr[0][j] 
        if str(est_res) == 'nan':
            print("nnnnnnneeeeest_res: ", j, est_res)
            
            
    index_frm = 0
    previous_frm_indx = 0
    while index_frm < num_frames:           # index_frm is 1 sec interval      
        
        #est_res = confg_est_frm_arr[0][previous_frm_indx]   # confg_est_frm_arr[0][index_frm]         # 0 corresponds to ground truth original big array without extract_specific_config_name_from_file function
        
        #kp_arr = getPersonEstimation(est_res)

        #if kp_arr == 'nan' or kp_arr == '' or kp_arr = '0':
        
        #print ("index index index: ", interval_jump,  index_frm)
        
        if index_frm >= history_frame_num:              # jump interval;  jump the first interval/frame, because calculating speed from second to previous interval/frame
                        
            used_current_frm = index_frm-1
            #used_prev_frm = used_current_frm - 1      # used_current_frm - PLAYOUT_RATE    # previous frame not previous sec
            #print ("kp_arr kp_arr, ", history_pose_est_arr[used_current_frm])
            
            feature1_speed_arr = getFeatureOnePersonMovingSpeed(history_pose_est_arr, used_current_frm, prev_EMA_speed_arr)
            prev_EMA_speed_arr = feature1_speed_arr

            
            feature1_relative_speed_arr = getFeatureOnePersonRelativeSpeed1(history_pose_est_arr, used_current_frm, prev_EMA_relative_speed_arr1)
            prev_EMA_relative_speed_arr1 = feature1_relative_speed_arr
            #print ("feature1_speed_arr feature2_relative_speed_arr, ", feature1_speed_arr.shape, feature2_relative_speed_arr.shape)
   
            feature2_relative_speed_arr = getFeatureOnePersonRelativeSpeed2(history_pose_est_arr, used_current_frm, prev_EMA_relative_speed_arr2)
            prev_EMA_relative_speed_arr2 = feature2_relative_speed_arr
            
            
            feature3_relative_speed_arr = getFeatureOnePersonRelativeSpeed3(history_pose_est_arr, used_current_frm, prev_EMA_relative_speed_arr3)
            prev_EMA_relative_speed_arr3 = feature1_relative_speed_arr
            #print ("feature1_speed_arr feature2_relative_speed_arr, ", feature1_speed_arr.shape, feature2_relative_speed_arr.shape)
   
            feature4_relative_speed_arr = getFeatureOnePersonRelativeSpeed2(history_pose_est_arr, used_current_frm, prev_EMA_relative_speed_arr4)
            prev_EMA_relative_speed_arr4 = feature4_relative_speed_arr
            
            
            total_features_arr = np.vstack((feature1_speed_arr, feature1_relative_speed_arr))            
            total_features_arr = np.vstack((total_features_arr, feature2_relative_speed_arr))            
            total_features_arr = np.vstack((total_features_arr, feature3_relative_speed_arr))            
            total_features_arr = np.vstack((total_features_arr, feature4_relative_speed_arr))
            
            
            input_x_arr[select_frm_cnt]=  total_features_arr  # feature1_speed_arr  # total_features_arr  #  input_x_arr[frm_id-1] = total_features_arr
            
            
            current_iterative_frm_id = index_frm #  + interval_jump               # the next interval for predicting is the class output
                        
            if minAccuracy == -1:
                delay_feature_arr[select_frm_cnt] = current_delay
                y_out_arr[select_frm_cnt], current_delay = select_config_boundedDelayOnly(acc_frame_arr, spf_frame_arr, current_iterative_frm_id, interval_frms, current_delay, minDelayTreshold)
            else:
                y_out_arr[select_frm_cnt] = select_config_boundedAccuracy(acc_frame_arr, spf_frame_arr, current_iterative_frm_id, minAccuracy)

            #print ("current_iterative_frm_id current_iterative_frm_id, ", acc_frame_arr[:, current_iterative_frm_id])
            current_cofig = id_config_dict[int(y_out_arr[select_frm_cnt-1])]
            
            #selected_configs_acc_lstselected_config_acc = acc_frame_arr[y_out_arr[select_frm_cnt], index_frm]
            
            #selected_configs_acc_lst.append(selected_config_acc)
            #print ("current_cofig: ", current_cofig)
            
            current_config_frmRt = int(current_cofig.split('-')[1])
               
            reso = int(current_cofig.split('-')[0].split('x')[1])

            history_reso_arr[select_frm_cnt] = reso
                
            curr_reso_aver= getConfigFeature(history_reso_arr, select_frm_cnt, prev_reso_aver)
            
            prev_reso_aver = curr_reso_aver
            
            reso_feature_arr[select_frm_cnt] =  reso # curr_reso_aver # y_out_arr[index]    #curr_reso_aver


            history_config_arr[select_frm_cnt-1] = y_out_arr[select_frm_cnt-1]
            curr_config_aver= getConfigFeature(history_config_arr, select_frm_cnt, prev_config_aver)
            
            prev_config_aver = curr_config_aver
            
            config_feature_arr[select_frm_cnt] =  y_out_arr[select_frm_cnt-1]  # curr_config_aver  #    # curr_config_aver  #   # y_out_arr[select_frm_cnt]  #   #
            
            
            frmRt = int(current_cofig.split('-')[1])
#
            history_frmRt_arr[select_frm_cnt-1] = frmRt
                
            curr_frmRt_aver= getFrmRateFeature(history_frmRt_arr, select_frm_cnt, prev_frmRt_aver)
            prev_frmRt_aver = frmRt  # curr_frmRt_aver
            
            
            #frmRt_feature_arr[select_frm_cnt] = curr_frmRt_aver
            
            #current_iterative_frm_id = previous_frm_indx + skipped_frm_cnt + switching_config_skipped_frm
            #lowest_config_id = [19, 10, 6, 2] 
            #stTime = time.time()
            
            #blurrinessScore_arr =  getBlurrinessFeature(imgPath, current_iterative_frm_id-1)
            #print ("end TimeTimeTime: ", time.time()-stTime)
            #blurriness_feature_arr[select_frm_cnt] = blurrinessScore_arr
            
            
            img_file_name = paddingZeroToInter(index_frm+1) + '.jpg'
            # '../input_output/one_person_diy_video_dataset/005_dance_frames/000026.jpg'
            current_instance_start_frm_path_arr[select_frm_cnt] = data_frame_path_dir + img_file_name

            select_frm_cnt += 1
            
            
        previous_frm_indx += 1
        
        index_frm +=  interval_jump   
        
        if index_frm > (max_frame_example_used-1):
            break 

    input_x_arr = input_x_arr[:select_frm_cnt].reshape(input_x_arr[:select_frm_cnt].shape[0], -1)
    
    current_instance_start_frm_path_arr = current_instance_start_frm_path_arr[:select_frm_cnt].reshape(current_instance_start_frm_path_arr[:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((current_instance_start_frm_path_arr, input_x_arr))
    
    if minAccuracy == -1:
        delay_feature_arr = delay_feature_arr[:select_frm_cnt].reshape(delay_feature_arr[:select_frm_cnt].shape[0], -1)
        input_x_arr = np.hstack((input_x_arr, delay_feature_arr))
      
    frmRt_feature_arr = frmRt_feature_arr[:select_frm_cnt].reshape(frmRt_feature_arr[:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((input_x_arr, frmRt_feature_arr))
    
    reso_feature_arr = reso_feature_arr[:select_frm_cnt].reshape(reso_feature_arr[:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((input_x_arr, reso_feature_arr))
   
    config_feature_arr = config_feature_arr[:select_frm_cnt].reshape(config_feature_arr[:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((input_x_arr, config_feature_arr))
    
    
    y_out_arr = y_out_arr[:select_frm_cnt]
    #print ("reso_feature_arr, ", reso_feature_arr.shape, reso_feature_arr)
    #print ("feature1_speed_arr, ", select_frm_cnt, input_x_arr.shape, y_out_arr.shape, feature1_speed_arr.shape, feature2_relative_speed_arr.shape)

    #print ("delay_feature_arr, ", delay_feature_arr)
    #checkCorrelationPlot(data_pose_keypoint_dir, input_x_arr, y_out_arr, id_config_dict)
    #print ("y_out_arr, ", y_out_arr)
    
    
    return input_x_arr, y_out_arr, id_config_dict, acc_frame_arr, spf_frame_arr, confg_est_frm_arr


def select_config_boundedAccuracy(acc_frame_arr, spf_frame_arr, index_id, minAccuracy):
    '''
    need to use frm_id-1, index start from 0
    '''   
    
    #print ("[:, frm_id-1]:", acc_frame_arr.shape, index_id)
    
    indx_config_above_minAcc = np.where(acc_frame_arr[:, index_id] >= minAccuracy)      # the index of the config above the threshold minAccuracy
    #print("indx_config_above_minAcc: ", indx_config_above_minAcc, len(indx_config_above_minAcc[0]))
            # in case no profiling config found satisfying the minAcc
    if len(indx_config_above_minAcc[0]) == 0:        # select a config with maximum accuracy, because not considering bounded delay
       
        return np.argmax(acc_frame_arr[:, index_id])   # selected the minimum spf, i.e. the fastest processing speed


    #print ("indx_config_above_minAcc:", indx_config_above_minAcc)
    tmp_config_indx = np.argmin(spf_frame_arr[indx_config_above_minAcc, index_id])   # selected the minimum spf, i.e. the fastest processing speed
    #print ("tmp_config_indx tmp_config_indx:", tmp_config_indx )
    selected_config_indx = indx_config_above_minAcc[0][tmp_config_indx]      # final selected indx from all config_indx
    #print ("final selected_config_indx:",selected_config_indx, spf_frame_arr[selected_config_indx, frm_id-1] )

    return selected_config_indx



def select_config_boundedDelayOnly(acc_frame_arr, spf_frame_arr, index_id, interval_frms, current_delay, minDelayTreshold):
    '''
    bounded delay, and then select the highest accuracy
    
    '''    
    #print ("[:, frm_id-1]:", acc_frame_arr.shape, acc_frame_arr[:, frm_id-1], spf_frame_arr[:, frm_id-1])
    '''
    indx_config_above_minAcc = np.where(acc_frame_arr[:, index_id] >= minAccuracy)      # the index of the config above the threshold minAccuracy
    #print("indx_config_above_minAcc: ", indx_config_above_minAcc, len(indx_config_above_minAcc[0]))
    
    
    cpy_minAccuracy = minAccuracy
    # in case no profiling config found satisfying the minAcc
    while len(indx_config_above_minAcc[0]) == 0:
        cpy_minAccuracy = cpy_minAccuracy - 0.05 
        indx_config_above_minAcc = np.where(acc_frame_arr[:, index_id] >= cpy_minAccuracy)      # the index of the config above the threshold minAccuracy
            
    indx_acc_selected_arr = np.argsort(acc_frame_arr[indx_config_above_minAcc, index_id], axis=1) 
    
    '''
    
    #print ("spf_selected_arr: ", indx_config_above_minAcc, indx_acc_selected_arr, np.sort(acc_frame_arr[indx_config_above_minAcc, index_id], axis=1)  )
    
    current_delay_cpy = current_delay
    #print ("gggg: ", current_delay_cpy)
    
    satisfied_indices = []
    for c in range(0, spf_frame_arr.shape[0]):           #
        
        current_delay_cpy += (spf_frame_arr[c][index_id])*(interval_frms)   # consumed time
        current_delay_cpy -= (interval_frms)*(1/PLAYOUT_RATE)       # streamed time only 1 frame by 1 frame
        #print ("mmmmmm: ", current_delay_cpy, spf_frame_arr[indx_config_above_minAcc[0][r]][index_id], (switching_frm_num+1)*(1/PLAYOUT_RATE))
        if current_delay_cpy <= 0:
            current_delay_cpy = 0.0
           
        if current_delay_cpy <= minDelayTreshold:
            #print ("rrrrrrrr: ", index_id, switching_frm_num, indx_config_above_minAcc[0][r], current_delay_cpy, tmp,  spf_frame_arr[indx_config_above_minAcc[0][r]][index_id])
            satisfied_indices.append(c)
        
        current_delay_cpy = current_delay
        #print ("ttttttttttttt: ", current_delay, current_delay_cpy)
    #if we can not find a config that satisfying the bounded accu and delay
    #print ("satisfied_indices satisfied_indices:", satisfied_indices)
    # selected the config with the highest accuracy
    tmp_config_indx = np.argmax(acc_frame_arr[satisfied_indices, index_id])   # selected the minimum spf, i.e. the fastest processing speed
    #print ("tmp_config_indx tmp_config_indx:", tmp_config_indx )
    
    selected_config_indx = satisfied_indices[tmp_config_indx]      # final selected indx from all config_indx
    #print ("final selected_config_indx:",selected_config_indx, (spf_frame_arr[selected_config_indx][index_id])*(interval_frms))
    current_delay_cpy += (spf_frame_arr[selected_config_indx][index_id])*(interval_frms)     # consumed time
    current_delay_cpy  -=  (interval_frms)*(1/PLAYOUT_RATE)        # streamed time
    
    #print ("current_delay_cpy current_delay_cpy:", current_delay_cpy, spf_frame_arr[selected_config_indx][index_id])
    #print ("current_delay selected_config_indx: ", current_delay_cpy, selected_config_indx)
    #if current_delay_cpy <= 0:
    #    current_delay_cpy = 0.0
        
    return selected_config_indx, current_delay_cpy

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


if __name__== "__main__": 
            
    
    x = 1            






