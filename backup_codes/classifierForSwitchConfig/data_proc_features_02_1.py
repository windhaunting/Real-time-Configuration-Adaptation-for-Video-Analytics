#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:03:05 2019

@author: fubao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:21:35 2019

@author: fubao
"""


import sys
import os
import csv
import pickle
import math

import numpy as np
import pandas as pd

from glob import glob
from blist import blist

from common_classifier import read_config_name_resolution_only


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir2
from profiling.common_prof import frameRates
from profiling.common_prof import PLAYOUT_RATE

COCO_KP_NUM = 17      # total 17 keypoints



def getPersonEstimation(est_res, personNo):
    '''
    analyze the personNo's pose estimation result"
    input: 
        [500, 220, 2, 514, 214, 2, 498, 210, 2, 538, 232, 2, 0, 0, 0, 562, 308, 2, 470, 304, 2, 614, 362, 2, 420, 362, 2, 674, 398, 2, 372, 394, 2, 568, 468, 2, 506, 468, 2, 596, 594, 2, 438, 554, 2, 616, 696, 2, 472, 658, 2],1.4246317148208618;
        [974, 168, 2, 988, 162, 2, 968, 158, 2, 1004, 180, 2, 0, 0, 0, 1026, 244, 2, 928, 250, 2, 1072, 310, 2, 882, 302, 2, 1112, 360, 2, 810, 346, 2, 1016, 398, 2, 948, 396, 2, 1064, 518, 2, 876, 482, 2, 1060, 630, 2, 900, 594, 2],1.4541109800338745;
        [6, 172, 2, 16, 164, 2, 6, 162, 2, 48, 182, 2, 0, 0, 0, 68, 256, 2, 10, 254, 2, 112, 312, 2, 0, 0, 0, 168, 328, 2, 0, 0, 0, 70, 412, 2, 28, 420, 2, 108, 600, 2, 0, 0, 0, 144, 692, 2, 0, 0, 0],1.283275842666626;
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 162, 2, 0, 0, 0, 92, 208, 2, 42, 206, 2, 134, 250, 2, 0, 0, 0, 176, 284, 2, 0, 0, 0, 118, 334, 2, 86, 334, 2, 126, 396, 2, 80, 396, 2, 148, 548, 2, 0, 0, 0],0.9429177641868591
        
        
    output:
        a specific person's detection result arr (17x2)
    '''
    person_est = est_res.split(';')[personNo].split('],')[0].replace('[', '')
    
    kp_arr = np.zeros((COCO_KP_NUM, 2))    # 17x2 each kp is a two-dimensional value
    

    for ind, kp in enumerate(person_est.split(',')):
        if ind%3 != 2:
            kp_arr[int(ind/3), ind%3] = kp

    return kp_arr
    


def getEuclideanDist(val, time_frm_interval):
    '''
    calculate the euclidean distance
    val: [x1, y1, x2, y2]
    '''
    dist = math.sqrt((val[1] - val[3])**2 + (val[0] - val[2])**2)
               
    speed = dist/time_frm_interval
    #print ("speed: ", val, speed)
    return speed
    


def getFeatureOnePersonMovingSpeed(history_pose_est_arr, current_frm_id, history_frame_num):
    '''
    feature1: One person’s moving speed of all keypoints V i,k based on the
    euclidean d distance of current frame with the previous frame {f j−m , m =
                                                                   1, 2, ..., 24
    only get the previous history_frame_num, that is current_frm_id is the current frame id
    the previous 24 frames are the previous frames.
    
    output: feature1_speed_arr COCO_KP_NUM x history_frame_num-1
    '''
    cur_frm_est_arr = history_pose_est_arr[current_frm_id]
    #print ("aaaa, ", current_frm_id, cur_frm_est_arr, len(range(current_frm_id-1, current_frm_id-1-history_frame_num-1, -1)))
    
    feature1_speed_arr = np.zeros((COCO_KP_NUM, history_frame_num-1))
    
    j = 0
    for i in range(current_frm_id-1, current_frm_id-history_frame_num, -1):

        #print ("aaaahistory_pose_est_arr[i], ", history_pose_est_arr[i])
        previous_arr = history_pose_est_arr[i]
        
        hstack_arr = np.hstack((cur_frm_est_arr, previous_arr))
        #print ("hstack_arr[i], ", hstack_arr)
        # print ("iiiiiiiiiii: ", i)
        time_frm_interval = (1.0/PLAYOUT_RATE)*(current_frm_id-i)
        feature1_speed_arr[:, j] = np.apply_along_axis(getEuclideanDist, 1, hstack_arr, time_frm_interval)
    
        j+= 1
    #print ("feature1_speed_arr[i], ", time_frm_interval, feature1_speed_arr.shape)
        
    return feature1_speed_arr

    
def relativeDistance(arr):
    '''
    arr [x1, y1],[x2,y2],...,[x5, y5]
    
    get the distance of x5 to xi (i=1,2,3,4)
    
    output: 4x1 arr
    '''
    
    num_kp = arr.shape[0]
    #print ("val ", arr, num_kp)
    rel_dist_arr = np.zeros(num_kp-1)
    for i in range(0, num_kp-1):
        rel_dist_arr[i] = math.sqrt((arr[i][1] -arr[num_kp-1][1])**2 + (arr[i][0] -arr[num_kp-1][0])**2)
    
        #print ("rel_dist_arrrrrr: ", arr[i], arr[4], rel_dist_arr)
    return rel_dist_arr
    
def relativeSpeed(arr1, arr2, time_frm_interval):
    '''
    input arr1: 5x2 , arr2: 5x2
    output: 4x1
    '''
    rel_dist1 = relativeDistance(arr1)
    rel_dist2 = relativeDistance(arr2)
    
    relative_speed_arr = abs(rel_dist1 - rel_dist2)
    #print ("relative_speed_arr: ", rel_dist1, rel_dist2, relative_speed_arr)
    return relative_speed_arr
    
def getFeatureOnePersonRelativeSpeed(history_pose_est_arr, current_frm_id, history_frame_num):
    '''
    feature 2 One person arm/feet’s relative speed to the torso of all keypoints
    with the previous frames.
    In each frame, we select the relative distance of left wrist, right wrist,left ankle
    and right ankle to the left hip.
    
    output: feature2_relative_speed_arr 5 x history_frame_num-1

    '''
    # get left wrist, right wrist,left ankle, right ankle and the left hip
    # which corresponds to [7, 4, 13, 10, 11]
    selected_kp_index = [7, 4, 13, 10, 11]
    selected_kp_history_pose_est_arr = history_pose_est_arr[:, selected_kp_index, :]
    #print ("selected_kp_history_pose_est_arr aaa, ", selected_kp_history_pose_est_arr[0])
    
    curr_frm_rel_dist_arr = selected_kp_history_pose_est_arr[current_frm_id]   #np.apply_along_axis(relativeDistance, 2, selected_kp_history_pose_est_arr)
        
    feature2_relative_speed_arr = np.zeros((len(selected_kp_index)-1, history_frame_num-1))
    
    #print ("selected_keypont_history_pose_est_arr[i], ", selected_kp_history_pose_est_arr.shape, selected_kp_history_pose_est_arr)

    j = 0
    for i in range(current_frm_id-1, current_frm_id-history_frame_num, -1):
        previous_arr = selected_kp_history_pose_est_arr[i]
        
        #print ("previous_arr aaa, ", previous_arr, curr_frm_rel_dist_arr)
        
        time_frm_interval = (1.0/PLAYOUT_RATE)*(current_frm_id-i)

        feature2_relative_speed_arr[:, j] = relativeSpeed( previous_arr, curr_frm_rel_dist_arr, time_frm_interval)

        j += 1
        
    #print ("feature2_relative_speed_arraaaaa, ", feature2_relative_speed_arr)

    return feature2_relative_speed_arr


def getOnePersonAllKPHistorySpeed(data_pose_keypoint_dir, history_frame_num, max_frame_example_used):
    '''
    get one person's all history keypoint
    One person’s moving speed of all keypoints V i,k based on the euclidean d distance of current frame with the previous frame {f j−m , m = 1, 2, ..., 24}
    
    current_frame_id is also included in the next frame's id
    
    the input feature here we use the most expensive features first
    
    1120x832_25_cmu_estimation_result
    
    start from history_frame_num;
    the first previous history frames are neglected
    
    '''
    
    # select one person, i.e. no 0
    
    personNo = 0
    
    #max_frame_example_used = 1000   # 8000
    #current_frame_id = 25
    
    
    # only read the most expensive config
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*1120x832_25_cmu_estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection

    print ("filePath: ", filePathLst[0], len(df_det))

    history_pose_est_arr = np.zeros((max_frame_example_used, COCO_KP_NUM, 2)) # np.zeros((len(df_det), COCO_KP_NUM, 2))        #  to make not shift when new frames comes, we store all values
    
    previous_frm_indx = 1
    
    
    input_x_arr = np.zeros((max_frame_example_used, 21, 24))
    
    for index, row in df_det.iterrows():  
        #print ("index, row: ", index, row)
        reso = row['Resolution']
        #frm_rate = row['Frame_rate']
        model = row['Model']
        #num_humans = row['numberOfHumans']        # number of human detected
        
        frm_id = int(row['Image_path'].split('/')[-1].split('.')[0])
        
        est_res = row['Estimation_result']
        
        
        #print ("frm_id num_humans, ", reso, model, frm_id)
            
        kp_arr = getPersonEstimation(est_res, personNo)
        #history_pose_est_dict[previous_frm_indx] = kp_arr
         
        history_pose_est_arr[index] = kp_arr
        #print ("kp_arr, ", kp_arr)
        #break    # debug only
        
        if previous_frm_indx> history_frame_num:
            #print ("previous_frm_indx, ", previous_frm_indx, index)
            # calculate the human moving speed feature (1)
            feature1_speed_arr = getFeatureOnePersonMovingSpeed(history_pose_est_arr, index, history_frame_num)
            
            #calculate the relative moving speed feature (2)
            feature2_relative_speed_arr = getFeatureOnePersonRelativeSpeed(history_pose_est_arr, index, history_frame_num)
            
            total_features_arr = np.vstack((feature1_speed_arr, feature2_relative_speed_arr))

            input_x_arr[frm_id-1] = total_features_arr
            #print ("total_features_arr: ", frm_id-1)
            #previous_frm_indx = 1
                
    
        previous_frm_indx += 1
        
        #how many are used for traing, validation, and test
        if index >= (max_frame_example_used-1):
            break 
    
    
    print ("feature1_speed_arr, ", input_x_arr, feature1_speed_arr.shape, feature2_relative_speed_arr.shape)

    return input_x_arr[history_frame_num:]


def readProfilingResultNumpy(data_pickle_dir):
    '''
    read profiling from pickle
    the pickle file is created from the file "writeIntoPickle.py"
    
    '''
    
    acc_frame_arr = np.load(data_pickle_dir + 'acc_frame.pkl')
    spf_frame_arr = np.load(data_pickle_dir + 'spf_frame.pkl')
    #acc_seg_arr = np.load(data_pickle_dir + file_lst[2])
    #spf_seg_arr = np.load(data_pickle_dir + file_lst[3])
    
    print ("acc_frame_arr ", type(acc_frame_arr), acc_frame_arr)
    
    return acc_frame_arr, spf_frame_arr

def select_config(acc_frame_arr, spf_frame_arr, frm_id, minAccuracy):
    '''
    need to use frm_id-1, index start from 0
    
    '''    
    #print ("[:, frm_id-1]:", acc_frame_arr.shape, acc_frame_arr[:, frm_id-1], spf_frame_arr[:, frm_id-1])
    
    indx_config_above_minAcc = np.where(acc_frame_arr[:, frm_id-1] >= minAccuracy)      # the index of the config above the threshold minAccuracy
    #print("indx_config_above_minAcc: ", indx_config_above_minAcc, len(indx_config_above_minAcc[0]))
        
    cpy_minAccuracy = minAccuracy
    # in case no profiling config found satisfying the minAcc
    while len(indx_config_above_minAcc[0]) == 0:
        cpy_minAccuracy = cpy_minAccuracy - 0.05 
        indx_config_above_minAcc = np.where(acc_frame_arr[:, frm_id-1] >= cpy_minAccuracy)      # the index of the config above the threshold minAccuracy
            
    #print ("indx_config_above_minAcc:", indx_config_above_minAcc)
    tmp_config_indx = np.argmin(spf_frame_arr[indx_config_above_minAcc, frm_id-1])   # selected the minimum spf, i.e. the fastest processing speed
    #print ("tmp_config_indx tmp_config_indx:", tmp_config_indx )
    selected_config_indx = indx_config_above_minAcc[0][tmp_config_indx]      # final selected indx from all config_indx
    #print ("final selected_config_indx:",selected_config_indx, spf_frame_arr[selected_config_indx, frm_id-1] )

    return selected_config_indx

def getGroundTruthY(max_frame_example_used, history_frame_num):
    '''
    this dataset Y
    '''
    data_pickle_dir = dataDir2 + 'output_006-cardio_condition-20mins/' + 'pickle_files_resolutionOnly/'

    data_pose_keypoint_dir = dataDir2 + 'output_006-cardio_condition-20mins/'
    config_id_dict, id_config_dict = read_config_name_resolution_only(data_pose_keypoint_dir, False)


    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)
    minAccuracy = 0.85
    frm_id = 26
    y_out_arr = np.zeros((max_frame_example_used), dtype=int)
    for frm_id in range(0, max_frame_example_used):
        
        selected_id = select_config(acc_frame_arr, spf_frame_arr, frm_id, minAccuracy)
        y_out_arr[frm_id] = id_config_dict[selected_id]
    print ("y_out_arr:", y_out_arr.shape)
    return y_out_arr[history_frame_num:]
    

def getDataExamples():
    
    data_pose_keypoint_dir =  dataDir2 + 'output_006-cardio_condition-20mins/'
    
    history_frame_num = 25          # 
    max_frame_example_used = 8025   # 8000
    x_input_arr = getOnePersonAllKPHistorySpeed(data_pose_keypoint_dir, history_frame_num, max_frame_example_used)
    
    y_out_arr = getGroundTruthY(max_frame_example_used, history_frame_num)
    
    print ("y_out_arr:",x_input_arr.shape, y_out_arr.shape)
    
    #data_examples_arr = np.hstack((x_input_arr, y_out_arr))
    
    
    out_frm_examles_pickle_dir = data_pose_keypoint_dir + "data_examples_files_resolutionOnly/" 
    if not os.path.exists(out_frm_examles_pickle_dir):
            os.mkdir(out_frm_examles_pickle_dir)
            
    with open(out_frm_examles_pickle_dir + "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(x_input_arr, fs)
        
    
    with open(out_frm_examles_pickle_dir + "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(y_out_arr, fs)


if __name__== "__main__": 
            
    
    getDataExamples()
            

