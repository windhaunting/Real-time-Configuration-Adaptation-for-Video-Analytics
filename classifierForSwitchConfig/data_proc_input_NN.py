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
import math

import numpy as np
import pandas as pd

from glob import glob
from blist import blist

from common_classifier import getPersonEstimation
from common_classifier import read_poseEst_conf_frm
from common_classifier import fillEstimation
from common_classifier import readProfilingResultNumpy
from common_classifier import load_data_all_features

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir3
from profiling.common_prof import frameRates
from profiling.common_prof import PLAYOUT_RATE

COCO_KP_NUM = 17      # total 17 keypoints
REPRESENT_DIM = 5           # representation dimensions.  mean, std so far


def read_one_video_data_representition(data_pickle_dir, max_frame_example_used):
    '''
    # read each video's data, read the representation; 
    #input the config  30 *nums_frame estimated keypoint
    #output: num*34      # nums*51    each frame is 17*3 (51 dimension)
    '''
    
    confg_est_frm_arr = read_poseEst_conf_frm(data_pickle_dir)

    
    
    #get the most expensive config's keypoint
    
    config_num = confg_est_frm_arr.shape[0]
    frame_num = confg_est_frm_arr.shape[1]
    
    max_frame_example_used  = min(max_frame_example_used, confg_est_frm_arr.shape[1])        # update
    

    #print ("confg_est_frm_arr shape, ", confg_est_frm_arr[0][0], confg_est_frm_arr.shape)
    
    
    pose_est_frms_arr = np.zeros((max_frame_example_used, COCO_KP_NUM, 3)) # np.zeros((len(df_det), COCO_KP_NUM, 3))        #  to make not shift when new frames comes, we store all values
    #first frame is not used?
    pose_est_frms_arr[0] = getPersonEstimation(confg_est_frm_arr[0][0])   # use the most expensive config confg_est_frm_arr[0]
    for j in range(1, max_frame_example_used):
        #print ("est_res pre, ", confg_est_frm_arr[0][j])
        est_res = confg_est_frm_arr[0][j]              # only most expensive config to calculate the feature
        
        if j == 1 and str(confg_est_frm_arr[0][j-1]) == 'nan':
            confg_est_frm_arr[0][j] = np.zeros((COCO_KP_NUM, 3))
            
        elif str(est_res) == 'nan':
            #print("eeeeest_res: ", j, est_res)
            pose_est_frms_arr[j] = pose_est_frms_arr[j-1]
            confg_est_frm_arr[0][j] = confg_est_frm_arr[0][j-1]
        else:
       
    
            pose_est_frms_arr[j] = getPersonEstimation(est_res)
            pose_est_frms_arr[j] = fillEstimation(j, pose_est_frms_arr[j-1], pose_est_frms_arr[j])

            tmp_arr_est = pose_est_frms_arr[j].reshape(1, -1)
            tmp_arr_est = str(tmp_arr_est.tolist()[0]) + ',' + str(confg_est_frm_arr[0][j].split('],')[-1])
            confg_est_frm_arr[0][j] = tmp_arr_est
            
    
    
    #pose_est_frms_arr = pose_est_frms_arr.reshape((pose_est_frms_arr.shape[0], -1))
    #print ("pose_est_frms_arr shape, ", pose_est_frms_arr[1], pose_est_frms_arr.shape)
    
    # we only fetch the two dimensions
    pose_est_frms_arr = pose_est_frms_arr[:, :, :2]
    pose_est_frms_arr = pose_est_frms_arr.reshape((pose_est_frms_arr.shape[0], -1))
    
    return pose_est_frms_arr

def get_x_input_data_set(data_pickle_dir, max_frame_example_used, seg_frm=25):
    # default segment is 1 second
    # each instance of X: PLAYOUT_RATE*51      corresponding to one y_output
    #output: number_instance* X.shape
    pose_est_frms_arr = read_one_video_data_representition(data_pickle_dir, max_frame_example_used)
    
    #print ("pose_est_frms_arr shape, ", pose_est_frms_arr[1], pose_est_frms_arr.shape)

    offset = PLAYOUT_RATE         # every 1 sec to get input vs output pair seg_frm   segment*PLAYOUT_RATE
    num_frm = pose_est_frms_arr.shape[0]
    feat_dim = pose_est_frms_arr.shape[1]
    #inpt_x_arr = np.zeros((num_frm, feat_dim))
    seg_num = num_frm//(offset)
    input_x_arr = np.zeros((seg_num, seg_frm, feat_dim))
    
    
    j = 0
    
    start_sec = math.ceil(seg_frm/PLAYOUT_RATE)    # with the history frame of seg_frm, how many sec we have to shift
    
    i = 0 + start_sec*PLAYOUT_RATE
    #print ("seg_num:", seg_num)
    while (i < (num_frm-num_frm%offset)):
        #print ("jjjjjjjj:", j)
        #print ("pose_est_frms_arr[i:i+offset]: ", pose_est_frms_arr[i:i+offset].shape)
        input_x_arr[j] = pose_est_frms_arr[i-seg_frm:i]
        
        i += PLAYOUT_RATE
        j += 1
    #input_x_arr = np.asarray(input_x_arr)
    #print ("inpt_x_arr shape, ", input_x_arr[0].shape, input_x_arr.shape)

    return input_x_arr, start_sec

def get_y_out_data_set(data_pickle_dir, minAccuracy):
    
    
    '''
    need to use frm_id-1, index start from 0
    '''   
    intervalFlag = 'sec'
    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir, intervalFlag)
    
    #print ("acc_frame_arr shape, ", acc_frame_arr[0].shape, acc_frame_arr.shape)
    
    num_frm = acc_frame_arr.shape[1]          
    offset = PLAYOUT_RATE
    
    y_out_arr = blist()
    for index_id in range(0, num_frm, offset):
        
        #print ("[:, frm_id-1]:", acc_frame_arr.shape,  index_id)
        
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

        #print ("final selected_config_indx:",selected_config_indx, acc_frame_arr[selected_config_indx, index_id])
        y_out_arr.append(selected_config_indx)

    y_out_arr = np.asarray(y_out_arr)
    #print ("y_out_arr shape, ", y_out_arr[0].shape, y_out_arr.shape)
    
    return y_out_arr


def get_x_y_data(data_pickle_dir, max_frame_example_used=16000, seg_frm=25, minAccuracy=0.95):
    # get the traing and testing dataset
    
    #data_pickle_dir =  dataDir3 + 'output_005_dance/' + 'frames_pickle_result/'
            
    x_input_arr, start_sec = get_x_input_data_set(data_pickle_dir, max_frame_example_used, seg_frm=seg_frm)
    
    y_out_arr = get_y_out_data_set(data_pickle_dir, minAccuracy)
    
    minLen = min(x_input_arr.shape[0], y_out_arr.shape[0]-start_sec)       # because y_out_arr has last frame_lost
    
    x_input_arr = x_input_arr[0:minLen-1]   #
    y_out_arr = y_out_arr[start_sec:minLen-1+start_sec]     # predict next config
    #print ("get_x_y_data x_input_arr shape, ", x_input_arr.shape, y_out_arr.shape)
    
    return x_input_arr, y_out_arr
    



def get_all_video_x_y_data():
    # get the keypoint as feature
    
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/', \
                    'output_011_dance/', 'output_012_dance/', \
                    'output_013_dance/', 'output_014_dance/', \
                    'output_015_dance/', 'output_016_dance/', \
                    'output_017_dance/', 'output_018_dance/', \
                    'output_019_dance/', 'output_020_dance/', \
                    'output_021_dance/']
        

    input_video_frms_dir = ['001_dance_frames/', '002_dance_frames/', \
                        '003_dance_frames/', '004_dance_frames/',  \
                        '005_dance_frames/', '006_yoga_frames/', \
                        '007_yoga_frames/', '008_cardio_frames/',
                        '009_cardio_frames/', '010_cardio_frames/',
                        '011_dance_frames/', '012_dance_frames/',
                        '013_dance_frames/', '014_dance_frames/',
                        '015_dance_frames/', '016_dance_frames/',
                        '017_dance_frames/', '018_dance_frames/',
                        '019_dance_frames/', '020_dance_frames/',
                        '021_dance_frames/']
    
    # judge file exist or not

    max_frame_example_used =  16000 # 20000 #8025   # 10000
        
    minAccuracy = 0.95
    seg_frm = 50      # frame_num
    data_classification_dir = dataDir3  +'test_classification_result_NN/' + 'min_accuracy-' + str(minAccuracy) + '/'

    xfile = "X_data_features_config" + "-sampleNum" + str(max_frame_example_used) + "-seg_frm" + str(seg_frm) + "-minAcc" + str(minAccuracy) + ".pkl"
    yfile = "Y_data_features_config" + "-sampleNum" + str(max_frame_example_used) + "-seg_frm" + str(seg_frm) + "-minAcc" + str(minAccuracy) + ".pkl"
    
    if os.path.exists(data_classification_dir+xfile) and os.path.exists(data_classification_dir+yfile):
        
        total_X,total_y= load_data_all_features(data_classification_dir, xfile, yfile)
        
    else:
        X_lst = blist()
        y_lst = blist()
        
        for i, video_dir in enumerate(video_dir_lst):  # [2:3]:     # [2:3]:   #[1:2]:      #[0:1]:     #[ #[1:2]:  #[1:2]:         #[0:1]:
            
            #if i != 4:                    # check the 005_video only
            #    continue
            
            #data_pose_keypoint_dir =  dataDir3 + video_dir
            data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
            
            x_input_arr, y_out_arr = get_x_y_data(data_pickle_dir, max_frame_example_used=max_frame_example_used, seg_frm=seg_frm, minAccuracy=minAccuracy)
                
                
            #print ("x_input_arr shapeeeee: ", x_input_arr.shape)
            X_lst.append(x_input_arr)
            y_lst.append(y_out_arr.reshape(-1, 1))
            
            
        total_X = np.vstack(X_lst)
        total_y = np.vstack(y_lst)
        
        
        print ("total_X shapbbbb ", total_X.shape)
        data_classification_dir = dataDir3 + 'test_classification_result/'
        if not os.path.exists(data_classification_dir):
            os.mkdir(data_classification_dir)

        data_classification_dir = data_classification_dir + 'min_accuracy-' + str(minAccuracy)+ '/'
        if not os.path.exists(data_classification_dir):
            os.mkdir(data_classification_dir)
            
        with open(data_classification_dir + "X_data_features_config" + "-sampleNum" + str(max_frame_example_used) + "-seg_frm" + str(seg_frm) + "-minAcc" + str(minAccuracy) + ".pkl", 'wb') as fs:
            pickle.dump(total_X, fs)
                
        with open(data_classification_dir + "Y_data_features_config" + "-sampleNum" + str(max_frame_example_used) + "-seg_frm" + str(seg_frm) + "-minAcc" + str(minAccuracy) + ".pkl", 'wb') as fs:
            pickle.dump(total_y, fs) 
            
    return total_X, total_y


def get_all_augment_data_from_video_segment():
    
    
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/', \
                    'output_011_dance/', 'output_012_dance/', \
                    'output_013_dance/', 'output_014_dance/', \
                    'output_015_dance/', 'output_016_dance/', \
                    'output_017_dance/', 'output_018_dance/', \
                    'output_019_dance/', 'output_020_dance/', \
                    'output_021_dance/']
        

    input_video_frms_dir = ['001_dance_frames/', '002_dance_frames/', \
                        '003_dance_frames/', '004_dance_frames/',  \
                        '005_dance_frames/', '006_yoga_frames/', \
                        '007_yoga_frames/', '008_cardio_frames/',
                        '009_cardio_frames/', '010_cardio_frames/',
                        '011_dance_frames/', '012_dance_frames/',
                        '013_dance_frames/', '014_dance_frames/',
                        '015_dance_frames/', '016_dance_frames/',
                        '017_dance_frames/', '018_dance_frames/',
                        '019_dance_frames/', '020_dance_frames/',
                        '021_dance_frames/']
   
    
    # judge file exist or not
    max_frame_example_used =  17000 # 20000 #8025   # 10000
        
    minAccuracy = 0.95
    seg_frm = 50      # frame_num
    data_classification_dir = dataDir3  +'augmented_data_test_classification_result_NN/' + 'min_accuracy-' + str(minAccuracy) + '/'

    xfile = "X_data_features_config" + "-sampleNum" + str(max_frame_example_used) + "-seg_frm" + str(seg_frm) + "-minAcc" + str(minAccuracy) + ".pkl"
    yfile = "Y_data_features_config" + "-sampleNum" + str(max_frame_example_used) + "-seg_frm" + str(seg_frm) + "-minAcc" + str(minAccuracy) + ".pkl"
    
    if os.path.exists(data_classification_dir+xfile) and os.path.exists(data_classification_dir+yfile):
        
        total_X,total_y= load_data_all_features(data_classification_dir, xfile, yfile)
        
    else:
        
        X_lst = blist()
        y_lst = blist()
            
        for i, old_video_dir in enumerate(video_dir_lst[1:19]):  # [2:3]:     # [2:3]:   #[1:2]:      #[0:1]:     #[ #[1:2]:  #[1:2]:         #[0:1]:
                
            #if i != 4:                    # check the 005_video only
            #    continue
            #if i == 0:
            #    j = 10          # to 990
            #else:
            #    j = 280
            j = 10
            while (j < 50):
                if j == 0:
                    video_dir = old_video_dir
                else:
                    video_dir = '/'.join(old_video_dir.split('/')[:-1]) + "-start-" + str(j) + '/'
                
                print (" video_dir: ", video_dir)
                #data_pose_keypoint_dir =  dataDir3 + video_dir_lst[0]
                data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
                    
                x_input_arr, y_out_arr = get_x_y_data(data_pickle_dir, max_frame_example_used=max_frame_example_used, seg_frm=seg_frm, minAccuracy=minAccuracy)
                
             
                X_lst.append(x_input_arr)
                y_lst.append(y_out_arr.reshape(-1, 1))
                
                j += 10
                
        total_X = np.vstack(X_lst)
        total_y = np.vstack(y_lst)
        
        print("total_X: ", total_X.shape, total_y.shape)
            
        #data_pose_keypoint_dir =  dataDir3 + old_video_dir[0]
        
        data_classification_dir = dataDir3 + 'augmented_data_test_classification_result/' + 'min_accuracy-' + str(minAccuracy)+ '/'

        if not os.path.exists(data_classification_dir):
            os.mkdir(data_classification_dir)
            
        with open(data_classification_dir + "X_data_features_config" + "-sampleNum" + str(max_frame_example_used) + "-seg_frm" + str(seg_frm) + "-minAcc" + str(minAccuracy) + ".pkl", 'wb') as fs:
            pickle.dump(total_X, fs)
                
        with open(data_classification_dir + "Y_data_features_config" + "-sampleNum" + str(max_frame_example_used) + "-seg_frm" + str(seg_frm) + "-minAcc" + str(minAccuracy) + ".pkl", 'wb') as fs:
            pickle.dump(total_y, fs)   

    return total_X, total_y

        
if __name__== "__main__": 
        
    get_all_augment_data_from_video_segment()