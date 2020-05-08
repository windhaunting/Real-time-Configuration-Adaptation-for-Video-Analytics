#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:09:27 2020

@author: fubao
"""




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:20:12 2019

@author: fubao
"""


# get the data for modeling jumping frames

import sys
import os
import csv
import pickle
import math

import numpy as np
import pandas as pd

from glob import glob
from blist import blist

from data_file_process import write_pickle_data

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from classifierForSwitchConfig.common_classifier import read_poseEst_conf_frm_more_dim
from classifierForSwitchConfig.common_classifier import readProfilingResultNumpy

from profiling.writeIntoPickleConfigFrameAccSPFPoseEst import read_config_name_from_file
from profiling.common_prof import dataDir3
from profiling.common_prof import frameRates
from profiling.common_prof import PLAYOUT_RATE
from profiling.common_prof import resoStrLst_OpenPose
from profiling.common_prof import PLAYOUT_RATE
from profiling.common_prof import NUM_KEYPOINT   

from profiling.common_prof import computeOKS_1to1

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

#resolution_dict = {int(r.split('x')[1]): i for i, r in enumerate(resoStrLst_OpenPose)}

#print ("resolution_dict : ", resolution_dict)


#ALPHA_EMA = 0.8     # EMA

class DataGenerate(object):
    def __init__(self):
        pass


    def get_data_numpy_most_expensive(self, data_pose_keypoint_dir, data_pickle_dir):
        # read pickle data
        # output: most expensive configuration's  numpy  (13390, 17, 3) (13390,) (13365,)
        
        config_est_frm_arr = read_poseEst_conf_frm_more_dim(data_pickle_dir)
        
        intervalFlag = 'frame'
        acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir, intervalFlag)
    
        #print ('acc_frame_arr', config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        #config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)
        
        #print ("config_id_dict: ", len(config_id_dict), config_id_dict)
        
        # use most expensive configuration resolution to test now
        config_est_frm_arr = config_est_frm_arr[0]          # 
        acc_frame_arr = acc_frame_arr[0]
        spf_frame_arr = spf_frame_arr[0]
        print ('acc_frame_arr', config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        #estimat_frm_arr_more_dim = np.apply_along_axis(getPersonEstimation, 0, config_est_frm_arr)
        return config_est_frm_arr, acc_frame_arr, spf_frame_arr
    
    
    def get_detected_pos_frm_arr(self, start_frm_pos_arr, jumping_frame_num):
        # input: 17*3, one frame pos result
        # ouput: the array frame pos result with all frame detection number; jumping frame no need to detect actually
        detected_pos_frm_list = blist()
        for i in range(jumping_frame_num):
            detected_pos_frm_list.append(start_frm_pos_arr)
        #np.vstack([start_frm_pos_arr] * jumping_frame_num)
        #detected_pos_frm_arr = np.asarray(detected_pos_frm_arr)
        #print("detected_pos_frm_arr shape: ", start_frm_pos_arr, jumping_frame_num, detected_pos_frm_arr.shape)
        return detected_pos_frm_list
        
    def estimate_jumping_number(self, config_est_frm_arr, start_frame_index,  min_acc_threshold):
        #input: from a starting frame index, we use the current start frame to replace the several later frame
        # and calculate the accuracy, until we get an accuracy smaller than min_acc_threshold
        # ouput: the no. of the several frames jumped, and the replaced detection pose 
        
        # get start frame's pose estimation result
        #print("config_est_frm_arr: ", config_est_frm_arr.shape)
        frame_length = config_est_frm_arr.shape[0]
        
        curr_frame_index = start_frame_index + 1
        ref_pose = config_est_frm_arr[start_frame_index]     # reference pose
        #print("ref_pose shape: ", ref_pose.shape)
        
        average_acc = 0.0
        accum_acc = 0.0
        cnt_frames = 1      # include the first frame detected by DNN itself
        
        while(curr_frame_index < frame_length):
            # get the oks similarity acc
            curr_pose = config_est_frm_arr[curr_frame_index]
        
            oks = computeOKS_1to1(ref_pose, curr_pose, sigmas = None)
            accum_acc += oks
            #print ("oks: ", oks)
            average_acc = accum_acc/cnt_frames
            
            if average_acc < min_acc_threshold:
                break
                #return cnt_frames - 1
            curr_frame_index += 1
            cnt_frames += 1
        
        seg_detected_pos_frm_list = self.get_detected_pos_frm_arr(ref_pose, cnt_frames)
        return cnt_frames, seg_detected_pos_frm_list
    
    def get_data_instances(self, config_est_frm_arr, acc_frame_arr, spf_frame_arr, min_acc_threshold, speed_type = 'ema', all_kp_flag = 1, interval_frm = 1):
        # get estimated speed    
        # interval_frm for calculating the current estimated speed
        # all_kp_flag: use all points or 8 critical keypoints  (arm, wrist, knee, ankle) 
        
        
        detected_est_frm_arr = blist()       # detected pose result, each is 17x3,  because of jumping number, each segment has several frame replaced as the detection
        
        # add the first one frame for the first time speed
        detected_est_frm_arr.append(config_est_frm_arr[0]) 
        
        list_estimated_speed_2_jump_number = blist()          # data x => y
        
        FRM_NO =  spf_frame_arr.shape[0]          # total frame no for the whole video
        
        #print("FRM_NO: ", FRM_NO)
        
        ALPHA = 0.8
        time_interval = 1.0/PLAYOUT_RATE*interval_frm
        
        
        if all_kp_flag == 1:        # all keypoints
            arr_ema_speed = np.zeros((NUM_KEYPOINT, 3))

        if speed_type == 'ema':
            # Price(current)  x Multiplier)   + (1-Multiplier) * EMA(prev) 
            # Vi= ALPHA * (M_i) + (1-ALPHA)* V_{i-1}
            # current speed use previous current frame - previous frame as estimation now
            # start from 2nd frame
            current_indx = 1          # starting index = 0  left frame no = 1

            while(current_indx < FRM_NO):
                #get current speed, use current frame estimation point -  previous estimation point, and divide by 
                # frame time interval
                # vector speed
                current_frm_est = config_est_frm_arr[current_indx]  # detected_est_frm_arr[current_indx] #  current frame is detected, so we use
                
                prev_used_indx = max(current_indx - interval_frm, 0)
                
                #print("detected_est_frm_arr: ", len(detected_est_frm_arr), current_indx, prev_used_indx)
                prev_frm_est = detected_est_frm_arr[prev_used_indx]
                
                #print("current_frm_est: ", current_frm_est)
                #print("prev_frm_est: ", prev_frm_est)
                
                # calculate current speed vector
                if all_kp_flag == 1:        # use all the 17 keypoints  # absolute
                    arr_vec_diff = current_frm_est - prev_frm_est
                    #print("vec_diff: ", vec_diff)
                    
                    arr_abs_vec_diff = np.absolute(arr_vec_diff)
                    #print("abs_vec_diff: ", arr_abs_vec_diff)
                    arr_speed_vec = arr_abs_vec_diff/time_interval      #current speed
                    
                    
                    arr_ema_speed = arr_speed_vec * ALPHA + (1.0-ALPHA) * arr_ema_speed
                    
                    feature_x = self.get_feature_x(arr_ema_speed)

                    
                    start_frame_index = current_indx
                    count_jumping_frames, seg_detected_pos_frm_list = self.estimate_jumping_number(config_est_frm_arr, start_frame_index,  min_acc_threshold)
                    #print("end_frame_index: ", end_frame_index)
                    
                    data_one_instance = np.append(feature_x, count_jumping_frames)
                    # list_estimated_speed_2_jump_number.append([feature_x, end_frame_index])
                    #print("data_one_instance: ", data_one_instance)
                    list_estimated_speed_2_jump_number.append(data_one_instance)
        
                    # update the detected result of pose 
                    detected_est_frm_arr += seg_detected_pos_frm_list
                    
                    current_indx += count_jumping_frames  # update next segment start index
                    #print("ddddddddetected_est_frm_arr: ", len(detected_est_frm_arr), len(seg_detected_pos_frm_list), count_jumping_frames, current_indx)

                if current_indx > 200:  # debug only
                    break
            
        #print("list_estimated_speed_2_jump_number: ", len(list_estimated_speed_2_jump_number), len(detected_est_frm_arr))
        arr_estimated_speed_2_jump_number = np.asarray(list_estimated_speed_2_jump_number)
        detected_est_frm_arr = np.asarray(detected_est_frm_arr)
        return arr_estimated_speed_2_jump_number, detected_est_frm_arr
            
    
    def get_feature_x(self, arr_ema_speed):
        # from estimated speed
        feature_vect_speed = arr_ema_speed[:, :2]
        #print("arr_ema_speed: ", feature_vect_speed)
        
        # get mean of the keypoints 
        feature_vect_mean = np.mean(feature_vect_speed, axis = 0)
        feature_vect_var = np.var(feature_vect_speed, axis = 0)
        print("arr_ema_speed: ", feature_vect_mean, feature_vect_var, np.asarray([feature_vect_mean, feature_vect_var]))
        # get the jumping number y
        
        feature_x = np.hstack((feature_vect_speed.flatten(), feature_vect_mean, feature_vect_var))
                    #print("feature_x: ", feature_x)
        return feature_x
    
    def getDataExamples(self):
        # input video 
        #video_dir_lst = ['output_001-dancing-10mins/', 'output_006-cardio_condition-20mins/', 'output_008-Marathon-20mins/']   
        
        video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                        'output_003_dance/', 'output_004_dance/',  \
                        'output_005_dance/', 'output_006_yoga/', \
                        'output_007_yoga/', 'output_008_cardio/', \
                        'output_009_cardio/', 'output_010_cardio/']
            
        for video_dir in video_dir_lst[4:5]:    # [2:3]:   #[1:2]:  # [1:2]:  #[0:1]:        #[1:2]:
            data_pose_keypoint_dir =  dataDir3 + video_dir
            
            data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
            config_est_frm_arr, acc_frame_arr, spf_frame_arr = self.get_data_numpy_most_expensive(data_pose_keypoint_dir, data_pickle_dir)
            
            speed_type = 'ema'
            interval_frm = 1
            all_kp_flag = 1
            min_acc_threshold = 0.90
            list_estimated_speed_2_jump_number, detected_est_frm_arr = self.get_data_instances(config_est_frm_arr, acc_frame_arr, spf_frame_arr, min_acc_threshold, speed_type, all_kp_flag, interval_frm)

            out_pickle_dir =  dataDir3 + video_dir + "/jumping_number_result/"
            if not os.path.exists(out_pickle_dir):
                os.mkdir(out_pickle_dir)
            
            out_pickle_file = out_pickle_dir + "data_instance_xy.pkl"
            write_pickle_data(list_estimated_speed_2_jump_number, out_pickle_file)
            
if __name__== "__main__": 
    
    data_obj = DataGenerate()
    data_obj.getDataExamples()









