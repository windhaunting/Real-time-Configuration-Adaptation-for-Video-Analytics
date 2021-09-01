#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:11:49 2019

@author: fubao
"""

# consider which config to calculate the X features ? based on current selected config
# consider frm rate problem;       if the frame rate selected 5, then we only use previous -5 as the -1 frame in frame_rate 25

# calculate feature_ONE

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
from common_classifier import getNewconfig

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

ALPHA_EMA = 0.3       # EMA

#different frame rate corresponding the previous history_frm_num used for calculating the moving speed  [25, 15, 10, 5, 2, 1] 
frm_rate_to_previous_frm_id_dict = {1:25, 2:13, 5:5, 10:3, 15:2, 25:1}


def getEstimationEachConfig(data_pose_keypoint_dir, writeFlag):
    '''
    get config's estimation result
    '''
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)

    filePathLst = sorted(glob(data_pose_keypoint_dir + "*estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    config_num = len(config_id_dict)  # len(config_id_dict)
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection
    frame_num = len(df_det)+7000     # because maybe some frame_id is missing
    #create a numpy array
    confg_frm_est_arr = np.zeros((config_num, frame_num), dtype=object) # array of estimation result with config vs frame_Id
    
    print ("config_id_dict: ", config_id_dict,  len(config_id_dict), frame_num)
    
    for fileCnt, filePath in enumerate(filePathLst):
        df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)         # det-> detection
        

        print ("numy shape: ", confg_frm_est_arr.shape, filePath)
        
        for index, row in df_det.iterrows():  
            #print ("index, row: ", index, row)
            reso = row['Resolution']
            #frm_rate = row['Frame_rate']
            model = row['Model'].split('_')[0]
            frm_id = int(row['Image_path'].split('/')[-1].split('.')[0])
            est_res = row['Estimation_result']
            
            #print ("est_ressssss: ", est_res)
            #get the config index
            config_lst = getNewconfig(reso, model)     # add more configs
            for config in config_lst:
                id_conf = config_id_dict[config]  
                confg_frm_est_arr[id_conf, frm_id-1] = est_res    
            #print ("confg_frm_est_arr: ", str(confg_frm_est_arr[id_conf, frm_id-1]))
            
        #    break    # test only
        
         
    #print ("confg_frm_est_arr: ", confg_frm_est_arr.shape, confg_frm_est_arr[5][0])            

    if writeFlag:
        pickle_dir = dataDir2 + 'output_006-cardio_condition-20mins/' + 'pickle_files/'
        with open(pickle_dir + 'config_estimation_frm.pkl','wb') as fs:
            pickle.dump(confg_frm_est_arr, fs)
            
    return confg_frm_est_arr


def readConfigFrmEstFile(data_pickle_dir):
    '''
    read confg_frm_est_arr from the output of getEstimationEachConfig
    '''
    
    #pickle_dir = dataDir2 + 'output_006-cardio_condition-20mins/' + 'pickle_files/'
    
    pickleFile = data_pickle_dir + 'config_estimation_frm.pkl'
    confg_frm_est_arr = np.load(pickleFile)

    #print ("confg_frm_est_arr: ", str(confg_frm_est_arr[0][3]))
    #return
    return confg_frm_est_arr


def readProfilingResultNumpy(data_pickle_dir):
    '''
    read profiling from pickle
    the pickle file is created from the file "writeIntoPickle.py"
    
    '''
    
    acc_frame_arr = np.load(data_pickle_dir + 'acc_frame.pkl')
    spf_frame_arr = np.load(data_pickle_dir + 'spf_frame.pkl')
    #acc_seg_arr = np.load(data_pickle_dir + file_lst[2])
    #spf_seg_arr = np.load(data_pickle_dir + file_lst[3])
    
    #print ("acc_frame_arr ", type(acc_frame_arr), acc_frame_arr[:, 0])
    

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


def getFeatureOnePersonMovingSpeed(history_pose_est_arr, current_frm_id, lst_history_frame_num, prev_EMA_speed_arr):
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
        
    
    prev_frm_est_arr = history_pose_est_arr[current_frm_id-lst_history_frame_num]
    
    # get the current speed based on current frame and previous frame
    hstack_arr = np.hstack((cur_frm_est_arr, prev_frm_est_arr))
    
    time_frm_interval = (1.0/PLAYOUT_RATE)*lst_history_frame_num
    current_speed_arr = np.apply_along_axis(getEuclideanDist, 1, hstack_arr, time_frm_interval)    
    
    feature1_speed_arr = current_speed_arr * ALPHA_EMA  + (1-ALPHA_EMA) * prev_EMA_speed_arr
    
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

def getFeatureOnePersonRelativeSpeed(history_pose_est_arr, current_frm_id, lst_history_frame_num, prev_EMA_relative_speed_arr):
    '''
    feature 2 One person arm/feet’s relative speed to the torso of all keypoints
    with the previous frames.
    In each frame, we select the relative distance of left wrist, right wrist,left ankle
    and right ankle to the left hip.
    
    output: feature2_relative_speed_arr  dimen: 4

    '''
    # get left wrist, right wrist,left ankle, right ankle and the left hip
    # which corresponds to [7, 4, 13, 10, 11]
    selected_kp_index = [7, 4, 13, 10, 11]
    selected_kp_history_pose_est_arr = history_pose_est_arr[:, selected_kp_index, :]
    #print ("selected_kp_history_pose_est_arr aaa, ", selected_kp_history_pose_est_arr[0])
    
    curr_frm_rel_dist_arr = selected_kp_history_pose_est_arr[current_frm_id]   #np.apply_along_axis(relativeDistance, 2, selected_kp_history_pose_est_arr)
    
    previous_frm_arr = selected_kp_history_pose_est_arr[current_frm_id-lst_history_frame_num]

    time_frm_interval = 1.0/PLAYOUT_RATE
    
    current_speed_arr = relativeSpeed(curr_frm_rel_dist_arr, previous_frm_arr, time_frm_interval)
    
    feature2_relative_speed_arr = current_speed_arr * ALPHA_EMA  + (1-ALPHA_EMA) * prev_EMA_relative_speed_arr
    
    return feature2_relative_speed_arr


def getXFeatureOnePersonAllKPHistorySpeed(data_pose_keypoint_dir, data_pickle_dir, minAccuracy, max_frame_example_used):
    
    '''
    calculate the x features
    # read each frame's pose estimation result of all config 
    # read each frame
    # get the selected config
    # get the estimaition result according to the selected config
    #calculate the features
    
    '''

    #getEstimationEachConfig(data_pose_keypoint_dir, True)

    confg_frm_est_arr = readConfigFrmEstFile(data_pickle_dir)
    
    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)

    # select one person, i.e. no 0
    personNo = 0
    
    #max_frame_example_used = 1000      # 8000
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
    
    prev_EMA_speed_arr = np.zeros((COCO_KP_NUM,2))
    
    prev_EMA_relative_speed_arr = np.zeros((4,2))        # only get 4 keypoint
    
    
    lst_history_frame_num = 1
    selected_frm_id_for_training = blist()      # start from 0
    for index, row in df_det.iterrows():  
        #print ("index, row: ", index, row)
        
        if index == 0 or previous_frm_indx >= lst_history_frame_num:
            #reso = row['Resolution']
            #frm_rate = row['Frame_rate']
            #model = row['Model'].split('_')[0]
            #num_humans = row['numberOfHumans']        # number of human detected
            
            frm_id = int(row['Image_path'].split('/')[-1].split('.')[0])
            # select a config with bounded accuracy
            id_selected = select_config(acc_frame_arr, spf_frame_arr, frm_id, minAccuracy)
       
            config_selected = id_config_dict[id_selected]
            
            current_frm_rate = int(config_selected.split('-')[1])    # the frame rate of last config
            
            #print ("last_frm_rate: ", index, frm_id, config_selected, current_frm_rate)
            
            current_history_frame_num = frm_rate_to_previous_frm_id_dict[current_frm_rate]   # skip frame rate
            
            # get the estimation result from selected config
            est_res = str(confg_frm_est_arr[id_selected, frm_id])
            #print ("type est_res, ", type(est_res), est_res)
            
            kp_arr = getPersonEstimation(est_res, personNo)
            #history_pose_est_dict[previous_frm_indx] = kp_arr
             
            history_pose_est_arr[index] = kp_arr
            
            if index != 0:
                selected_frm_id_for_training.append(frm_id - 1)
                
            #print ("history_pose_est_arr: ", index, history_pose_est_arr[index] )
            
            if previous_frm_indx >= lst_history_frame_num:
                # calculate the speed
                feature1_speed_arr = getFeatureOnePersonMovingSpeed(history_pose_est_arr, index, lst_history_frame_num, prev_EMA_speed_arr)
                prev_EMA_speed_arr = feature1_speed_arr
                #print ("prev_EMA_speed_arr: ", index, prev_EMA_speed_arr )
                
                #calculate the relative moving speed feature (2)
                feature2_relative_speed_arr = getFeatureOnePersonRelativeSpeed(history_pose_est_arr, index, lst_history_frame_num, prev_EMA_relative_speed_arr)
                prev_EMA_relative_speed_arr = feature2_relative_speed_arr
                
                #print ("prev_EMA_relative_speed_arr aaa, ", prev_EMA_relative_speed_arr)
                total_features_arr = np.vstack((feature1_speed_arr, feature2_relative_speed_arr))                #input_x_arr[frm_id-1] = total_features_arr
            
                input_x_arr[frm_id-1] = total_features_arr

            lst_history_frame_num = current_history_frame_num
            
            previous_frm_indx = 1
                          
        previous_frm_indx += 1
    
    
        #how many are used for traing, validation, and test
        if index >= (max_frame_example_used-1):
            break 
    
    
    print ("feature1_speed_arr, ", len(selected_frm_id_for_training), input_x_arr.shape, feature1_speed_arr.shape, feature2_relative_speed_arr.shape)

    return input_x_arr[selected_frm_id_for_training], selected_frm_id_for_training
    



def getGroundTruthY(max_frame_example_used, selected_frm_id_for_training):
    '''
    this dataset Y
    '''
    data_pickle_dir = dataDir2 + 'output_006-cardio_condition-20mins/' + 'pickle_files/'

    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)
    minAccuracy = 0.85
    frm_id = 26
    y_out_arr = np.zeros((max_frame_example_used), dtype=int)
    for frm_id in range(0, max_frame_example_used):
        y_out_arr[frm_id] = select_config(acc_frame_arr, spf_frame_arr, frm_id, minAccuracy)
    
    print ("y_out_arr:", y_out_arr.shape)
    return y_out_arr[selected_frm_id_for_training]
    

def getDataExamples():
    
    data_pose_keypoint_dir =  dataDir2 + 'output_006-cardio_condition-20mins/'
    data_pickle_dir = dataDir2 + 'output_006-cardio_condition-20mins/' + 'pickle_files/'
    
    history_frame_num = 1          # 
    max_frame_example_used = 35765  # 8025   # 8000
    x_input_arr, selected_frm_id_for_training = getXFeatureOnePersonAllKPHistorySpeed(data_pose_keypoint_dir, data_pickle_dir, minAccuracy, max_frame_example_used)
    
    
    
    y_out_arr = getGroundTruthY(max_frame_example_used, selected_frm_id_for_training)
    
    print ("y_out_arr:",x_input_arr.shape, y_out_arr.shape)
    
    #data_examples_arr = np.hstack((x_input_arr, y_out_arr))
    
    
    out_frm_examles_pickle_dir = data_pose_keypoint_dir + "data_examples_files_feature_selected_config/" 
    if not os.path.exists(out_frm_examles_pickle_dir):
            os.mkdir(out_frm_examles_pickle_dir)
            
    with open(out_frm_examles_pickle_dir + "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(x_input_arr, fs)
        
    
    with open(out_frm_examles_pickle_dir + "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(y_out_arr, fs)
    
    
    
if __name__== "__main__": 
    data_pose_keypoint_dir = dataDir2 + "output_006-cardio_condition-20mins/"
    
    data_pickle_dir = dataDir2 + 'output_006-cardio_condition-20mins/' + 'pickle_files/'

    minAccuracy = 0.85
    max_frame_example_used = 35765 # 8000           # 8000
    #getXFeatureOnePersonAllKPHistorySpeed(data_pose_keypoint_dir, data_pickle_dir, minAccuracy, max_frame_example_used)
    
    getDataExamples()