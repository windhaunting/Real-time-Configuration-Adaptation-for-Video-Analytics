#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:37:04 2019

@author: fubao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:09:06 2019

@author: fubao
"""

#Feature_two: calculate the EMA of current speed + relative speed. The current speed use current frame and it previous  N frame’s weighted average 

import sys
import os
import csv
import pickle
import math

import numpy as np
import pandas as pd

from glob import glob
from blist import blist

from common_classifier import read_config_name_from_file


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir2
from profiling.common_prof import frameRates
from profiling.common_prof import PLAYOUT_RATE

COCO_KP_NUM = 17      # total 17 keypoints


'''
{{0, "Nose"}, //t
{1, "Neck"}, //f is not included in coco. How do you get the Neck keypoints
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


# use EMA EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
# =  Price(current)  x Multiplier)   + (1-Multiplier) * EMA(prev) 
# Vi= ALPHA * (M_i) + (1-ALPHA)*V_{i-1}

ALPHA_EMA = 0.3       # 

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
    
    if (math.sqrt(val[0]**2 + val[1]**2) *math.sqrt(val[2]**2 + val[3]**2)) == 0:
        cos_angle = 0
    else:
        cos_angle = (val[0]*val[2] + val[1]*val[3]) / (math.sqrt(val[0]**2 + val[1]**2) *math.sqrt(val[2]**2 + val[3]**2))
    
    speed_angle = [speed, cos_angle]
    #print ("speed: ", val, speed)
    return speed_angle
    


def getFeatureOnePersonMovingSpeed(history_pose_est_arr, current_frm_id, previous_history_frms, weighted_hist_frm, history_frame_num, prev_EMA_speed_arr):
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
    
        
    current_speed_arr = np.zeros((history_frame_num, COCO_KP_NUM, 2))
    belta = 0.5
    j = 0
    
    for i in range(0, len(previous_history_frms)):
        #print ("aaaahistory_pose_est_arr[i], ", history_pose_est_arr[i])
        
        prev_frm_est_arr = history_pose_est_arr[current_frm_id-i]
        
        hstack_arr = np.hstack((cur_frm_est_arr, prev_frm_est_arr))
        #print ("hstack_arr[i], ", hstack_arr)
        #print ("iiiiiiiiiii: ", i)
        time_frm_interval = (1.0/PLAYOUT_RATE)*(current_frm_id-i)
        current_speed_arr[j,:] = weighted_hist_frm[i]*np.apply_along_axis(getEuclideanDist, 1, hstack_arr, time_frm_interval)
        j+= 1

    # get the current speed based on current frame and previous frame
    #hstack_arr = np.hstack((cur_frm_est_arr, prev_frm_est_arr))
    
    #time_frm_interval = 1.0/PLAYOUT_RATE
    #current_speed_arr = np.apply_along_axis(getEuclideanDist, 1, hstack_arr, time_frm_interval)    
    current_speed_arr = np.sum(current_speed_arr, axis=0)
    #print ("current_speed_arr, ", current_speed_arr.shape, feature1_speed_arr.shape)
    feature1_speed_arr = current_speed_arr * ALPHA_EMA  + (1-ALPHA_EMA) * prev_EMA_speed_arr
    
    #print ("feature1_speed_arr, ", current_speed_arr, time_frm_interval, feature1_speed_arr.shape)
     
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
        rel_val_arr[i] =  [arr[i][1] -arr[num_kp-1][1], arr[i][0] -arr[num_kp-1][0]]
    
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
    

def getFeatureOnePersonRelativeSpeed(history_pose_est_arr, current_frm_id, previous_history_frms, weighted_hist_frm, history_frame_num, prev_EMA_relative_speed_arr):
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

    current_speed_arr = np.zeros((history_frame_num, 4, 2))
    belta = 0.5

    j = 0
    for i in range(0, len(previous_history_frms)):
        previous_arr = selected_kp_history_pose_est_arr[current_frm_id-i]
        
        #print ("previous_arr aaa, ", previous_arr, curr_frm_rel_dist_arr)
        
        time_frm_interval = (1.0/PLAYOUT_RATE)*(current_frm_id-i)

        current_speed_arr[j, :] = weighted_hist_frm[i] * relativeSpeed( previous_arr, curr_frm_rel_dist_arr, time_frm_interval)

        j += 1
        
    current_speed_arr = np.sum(current_speed_arr, axis=0)
    feature2_relative_speed_arr = current_speed_arr * ALPHA_EMA  + (1-ALPHA_EMA) * prev_EMA_relative_speed_arr

    
    return feature2_relative_speed_arr


def getOnePersonAllKPHistorySpeed(data_pose_keypoint_dir, previous_history_frms, weighted_hist_frm, history_frame_num, max_frame_example_used):
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
    
    # select one person, i.e. no 0
    personNo = 0
    
    #max_frame_example_used = 1000   # 8000
    #current_frame_id = 25
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, True)
    
    print ("config_id_dict: ", len(config_id_dict))
    # only read the most expensive config
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*1120x832_25_cmu_estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection

    print ("filePath: ", filePathLst[0], len(df_det))


    history_pose_est_arr = np.zeros((max_frame_example_used, COCO_KP_NUM, 2)) # np.zeros((len(df_det), COCO_KP_NUM, 2))        #  to make not shift when new frames comes, we store all values
    
    previous_frm_indx = 1
    
    
    input_x_arr = np.zeros((max_frame_example_used, 21, 2))       # 17 + 4
    
    prev_EMA_speed_arr = np.zeros((COCO_KP_NUM, 2))
    
    prev_EMA_relative_speed_arr = np.zeros((4, 2))        # only get 4 keypoint to left hip
    
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
            feature1_speed_arr = getFeatureOnePersonMovingSpeed(history_pose_est_arr, index, previous_history_frms, weighted_hist_frm, history_frame_num, prev_EMA_speed_arr)
            
            prev_EMA_speed_arr = feature1_speed_arr
            #calculate the relative moving speed feature (2)
            feature2_relative_speed_arr = getFeatureOnePersonRelativeSpeed(history_pose_est_arr, index, previous_history_frms, weighted_hist_frm, history_frame_num, prev_EMA_relative_speed_arr)
            
            prev_EMA_relative_speed_arr = feature2_relative_speed_arr
            total_features_arr = np.vstack((feature1_speed_arr, feature2_relative_speed_arr))

            input_x_arr[frm_id-1] = total_features_arr
            
            #print ("total_features_arr: ", frm_id-1)
                
    
        previous_frm_indx += 1
        
        #how many are used for traing, validation, and test
        if index >= (max_frame_example_used-1):
            break 
    
    
    print ("feature1_speed_arr, ", input_x_arr, input_x_arr.shape, feature1_speed_arr.shape, feature2_relative_speed_arr.shape)

    return input_x_arr[history_frame_num:]




def select_config(acc_frame_arr, spf_frame_arr, frm_id, minAccuracy):
    '''
    #need to use frm_id-1, index start from 0
    
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
    #this dataset Y
    '''
    data_pickle_dir = dataDir2 + 'output_006-cardio_condition-20mins/' + 'pickle_files/'

    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)
    minAccuracy = 0.85
    frm_id = 26
    y_out_arr = np.zeros((max_frame_example_used), dtype=int)
    for frm_id in range(0, max_frame_example_used):
        y_out_arr[frm_id] = select_config(acc_frame_arr, spf_frame_arr, frm_id, minAccuracy)
    
    print ("y_out_arr:", y_out_arr.shape)
    return y_out_arr[history_frame_num:]
    

def getDataExamples():
    
    data_pose_keypoint_dir =  dataDir2 + 'output_006-cardio_condition-20mins/'
    
    previous_history_frms = [1, 5, 10]        # previous -1, -5, -10 frames
    weighted_hist_frm = [0.6, 0.3, 0.1]
    history_frame_num = max(previous_history_frms)
    max_frame_example_used = 8025   # 8000
    x_input_arr = getOnePersonAllKPHistorySpeed(data_pose_keypoint_dir, previous_history_frms, weighted_hist_frm, history_frame_num, max_frame_example_used)
    
    
    
    y_out_arr = getGroundTruthY(max_frame_example_used, history_frame_num)
    
    print ("y_out_arr:",x_input_arr.shape, y_out_arr.shape)
    
    #data_examples_arr = np.hstack((x_input_arr, y_out_arr))
    
    
    out_frm_examles_pickle_dir = data_pose_keypoint_dir + "data_examples_files/" 
    if not os.path.exists(out_frm_examles_pickle_dir):
            os.mkdir(out_frm_examles_pickle_dir)
            
    with open(out_frm_examles_pickle_dir + "X_data_features_config-weighted_interval-history-frms" + str(previous_history_frms[0]) + "-"+  str(previous_history_frms[1]) +"-" +  str(previous_history_frms[2]) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(x_input_arr, fs)
        
    
    with open(out_frm_examles_pickle_dir + "Y_data_features_config-weighted_interval-history-frms" + str(previous_history_frms[0]) + "-"+  str(previous_history_frms[1]) +"-" +  str(previous_history_frms[2]) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(y_out_arr, fs)
    

if __name__== "__main__": 
            
    
    getDataExamples()
            




