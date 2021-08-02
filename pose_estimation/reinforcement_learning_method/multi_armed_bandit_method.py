#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:40:53 2021

@author: fubao
"""

# based on multi-armed bandit


import sys
import os
import math
import numpy as np
import random as rand

from blist import blist

from fitting_accuracy_func import fit_function
from fitting_accuracy_func import expo_func

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from data_file_process import write_pickle_data

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, current_file_cur + '/../..')

from classifierForSwitchConfig.common_classifier import read_poseEst_conf_frm_more_dim
from classifierForSwitchConfig.common_classifier import readProfilingResultNumpy

from profiling.common_prof import dataDir3
from profiling.common_prof import PLAYOUT_RATE
from profiling.common_prof import resoStrLst_OpenPose
from profiling.common_prof import NUM_KEYPOINT   

from profiling.common_prof import computeOKS_1to1

from profiling.writeIntoPickleConfigFrameAccSPFPoseEst import read_config_name_from_file



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
                    'output_021_dance/', 'output_022_dance/', \
                    'output_023_dance/', 'output_024_dance/', \
                    'output_025_dance/', 'output_026_dance/', \
                    'output_027_dance/', 'output_028_dance/', \
                    'output_029_dance/', 'output_030_dance/', \
                    'output_031_dance/', 'output_032_dance/', \
                    'output_033_dance/', 'output_034_dance/', \
                    'output_035_dance/']


class MultiArmedConfig(object):
    
    def __init__(self):
        
        self.alpha = 0.8 # accuracy to reward ; the weight for accuracy coefficient
        
    
    
    def get_data_numpy(self, data_pose_keypoint_dir, data_pickle_dir, intervalFlag, all_config_flag):
        # read pickle data (30, 13390, 17, 3) (30, 13390,) (30, 13365,)
        # output: output all or  most expensive configuration's  numpy  (5, 13390, 17, 3) (5, 13390,) (5, 13365,)
        
        config_est_frm_arr = read_poseEst_conf_frm_more_dim(data_pickle_dir)
        
        acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir, intervalFlag)
    
        #print ('acc_frame_arr', config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        if all_config_flag:      # output all configuration 5x6 30 configs
            return config_est_frm_arr, acc_frame_arr, spf_frame_arr
        
        config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)
        
        print ("id_config_dict: ", len(id_config_dict), id_config_dict)
      
        #print ("config_id_dict: ", len(config_id_dict), config_id_dict)
        
        # use most expensive configuration resolution to test now
        #config_est_frm_arr = config_est_frm_arr[0]          # 
        #acc_frame_arr = acc_frame_arr[0]
        #spf_frame_arr = spf_frame_arr[0]
        #print ('acc_frame_arr', config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        # get only 25 frame that is each frame's detection result
        # {'1120x832-25-cmu': 0,  '960x720-25-cmu': 1, '640x480-25-cmu': 3, '480x352-25-cmu': 5, '320x240-25-cmu': 9}
        
        resolution_id_max_frame_rate = [] # each frame detect
        for reso in resoStrLst_OpenPose:          # 25 frame
            key = reso + "-25-cmu"
            if key in config_id_dict:
                resolution_id_max_frame_rate.append(config_id_dict[key])
        
        #print ("resolution_id_max_frame_rate: ", resolution_id_max_frame_rate)
        
        config_est_frm_arr = config_est_frm_arr[resolution_id_max_frame_rate]          #  (x, 5)
        acc_frame_arr = acc_frame_arr[resolution_id_max_frame_rate]
        spf_frame_arr = spf_frame_arr[resolution_id_max_frame_rate]
        #print ('222acc_frame_arr', config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        #estimat_frm_arr_more_dim = np.apply_along_axis(getPersonEstimation, 0, config_est_frm_arr)
        return config_est_frm_arr, acc_frame_arr, spf_frame_arr
    
        
    def fit_accuracy_frame_rate(self, fr_acc_data_point):
        # fit the accuracy  with frame rate
        # return the association
        
        popt_acc_fr = fit_function(fr_acc_data_point)          # for accuracy with frame rate
        
        return popt_acc_fr

    def fit_accuracy_resolution(self, reso_acc_data_point):
        # fit the accuracy  with frame rate
        # return the association
        
        popt_acc_reso = fit_function(reso_acc_data_point)          # for accuracy with frame rate
        
        return popt_acc_reso


    def fitting_accuracy(self):
        x = 1   
        
        

        
    def instance_reward(self,instance_acc, instance_processing_time):
        # get the instance reward from the accuracy and processing time
        # r = alpha*instance_acc + (1-alpha)*instance_processing_time
        
        inst_reward = self.alpha = 0.8*instance_acc + (1-self.alpha)*instance_processing_time
        
        return inst_reward
    
        
    def original_Epoch_Greedy(self, c,d,T,K,r):
        accumulated_reward = np.zeros(K, dtype=np.int)
    
        for t in range(T):
            if t == 0:
                # first trail
                ind = rand.randrange(K)
                accumulated_reward[ind] = r  # r: reward in first trail
            else:
                # more than one trail
                epoch = min(1,(c*K)/(d*d*(t+1)))
                val =   rand.random()
                if val < epoch:
                    ind = rand.randrange(K)
                    accumulated_reward[ind] = accumulated_reward[ind] + r # r: reward for ind machine in trail t
                else:
                    ind = accumulated_reward.index(max(accumulated_reward))
                    accumulated_reward[ind] = accumulated_reward[ind] + r # r: reward for ind machine in trail t
                
    
    def get_accuracy_interval_time(self, config_est_frm_arr, acc_frame_arr, spf_frame_arr, update_time):
        # get the accuracy in the interval time similar to profiling method, but it is not profiling
        # so we have to run the expensive configuration.
        # get the average accuracy and the configuration;  for three configurations that are enough
        
        print("config_est_frm_arr: ", config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        
        # read 
                

        
    def online_learning_RL(self):
        
        global dataDir3
        dataDir3 = "../" + dataDir3 
        
        update_time = 1 #    # one second
        #all_arr_estimated_speed_2_jump_number = blist()  # all video
        for i, video_dir in enumerate(video_dir_lst[0:1]): # [3:4]):    # [2:3]:   #[1:2]:  # [1:2]:  #[0:1]:        #[1:2]:
            data_pose_keypoint_dir = dataDir3 + video_dir
            
            data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
            #data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result_each_frm/'
            intervalFlag = 'frame'
            all_config_flag = False
            config_est_frm_arr, acc_frame_arr, spf_frame_arr = self.get_data_numpy(data_pose_keypoint_dir, data_pickle_dir, intervalFlag, all_config_flag)
            
            
            self.get_accuracy_interval_time(config_est_frm_arr, acc_frame_arr, spf_frame_arr, update_time)
            
            
            xx 
 
            
if __name__== "__main__": 
    
    MultiArmedConfig_obj = MultiArmedConfig()
    MultiArmedConfig_obj.online_learning_RL()
    
            
        