#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:56:11 2020

@author: fubao
"""


# periodic update profiling update experiments



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:50:29 2020

@author: fubao
"""

# -*- coding: utf-8 -*-


# online detection

import sys
import os
import cv2

import numpy as np
from glob import glob
import time


from get_data_jumpingNumber_resolution import samplingResoDataGenerate
from get_data_jumpingNumber_resolution import video_dir_lst


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/../..')


from profiling.common_prof import dataDir3
from profiling.common_prof import PLAYOUT_RATE
from classifierForSwitchConfig.common_classifier import getParetoBoundary_boundeAcc

dataDir3 = "../" + dataDir3

# video analytics based on offline once time profiling 

class PeriodicOnlineProfiling(object):
    def __init__(self):
        pass
    
    def read_video_frm(self, predicted_video_frm_dir):
        
        imagePathLst = sorted(glob(predicted_video_frm_dir + "*.jpg"))  # , key=lambda filePath: int(filePath.split('/')[-1][filePath.split('/')[-1].find(start)+len(start):filePath.split('/')[-1].rfind(end)]))          # [:75]   5 minutes = 75 segments

        #print ("imagePathLst: ", len(imagePathLst), imagePathLst[0])
        
        return imagePathLst
    
    
    
    def select_config(self, acc_frame_arr, spf_frame_arr,frm_start_indx, profile_frame_length, min_acc_thres): 
        # select a config that can achieve min accuracy threshold with min spf
        
        min_spf = float('inf')
        ans_indx = 0
        
        average_profile_acc_arr = np.mean(acc_frame_arr[:, frm_start_indx:frm_start_indx+profile_frame_length], axis=1, keepdims=True)
        average_profile_spf_arr = np.mean(spf_frame_arr[:, frm_start_indx:frm_start_indx+profile_frame_length], axis=1, keepdims=True)
        #print("aaaaaaverage_profile_spf_arr: ", average_profile_spf_arr.shape)
        for idx, acc in np.ndenumerate(average_profile_acc_arr): 
            spf = average_profile_spf_arr[idx[0]]
            
            if acc >= min_acc_thres and spf < min_spf:
                min_spf = spf
                ans_indx = idx[0]
        
        return ans_indx      # index
                
    def select_optimal_configuration_above_accuracy(self, min_acc_thres, config_est_frm_arr, acc_frame_arr, spf_frame_arr, frm_start_indx, segment_len_time, pareto_bound_flag):
        # select the best configuration with the minimum efficiency above the accuracy threshold
        profile_frame_length = segment_len_time * PLAYOUT_RATE
        
        
        #average_profile_acc_arr = np.mean(acc_frame_arr[:, 0:profile_frame_length], axis=1, keepdims=True)
        
        #print("average_profile_acc_arr: ", average_profile_acc_arr.shape)
        
        if pareto_bound_flag == 0:   # all the configurations
            ans_indx = self.select_config(acc_frame_arr,spf_frame_arr,frm_start_indx, profile_frame_length, min_acc_thres)
            
        elif pareto_bound_flag == 1:    # pareto boundary with minimum accuracy threshold
            
            first_segment_indx = min(profile_frame_length, 25)
            acc_arr = np.mean(acc_frame_arr[:, frm_start_indx:first_segment_indx+frm_start_indx], axis=1)
            spf_arr = np.mean(spf_frame_arr[:, frm_start_indx:first_segment_indx+frm_start_indx], axis=1)
            
            #print("acc_arr, spf_arr: ", acc_arr.shape, spf_arr.shape)
            lst_idx_selected_config = getParetoBoundary_boundeAcc(acc_arr, spf_arr, min_acc_thres)
            
            #lst_idx_selected_config.remove(0)       # if 0 in lst_idx_selected_config
    
            #print ("lst_idx_selected_config: ", len(lst_idx_selected_config), lst_idx_selected_config)
            
            acc_frame_arr = acc_frame_arr[lst_idx_selected_config, :]
            spf_frame_arr = spf_frame_arr[lst_idx_selected_config, :]
            
            ans_indx = self.select_config(acc_frame_arr, spf_frame_arr,frm_start_indx, profile_frame_length, min_acc_thres)
         
        #print("ans_indx: ", ans_indx)
        profiling_time = np.sum(spf_frame_arr[:, frm_start_indx:frm_start_indx+profile_frame_length])
            
        #print("profiling_time: ", profiling_time)
        return ans_indx, profiling_time
    
        
    def execute_video_periodic_time_profiling(self, predicted_video_dir, min_acc_thres, interval_len_time, segment_len_time):
            

        data_pose_keypoint_dir = dataDir3 + predicted_video_dir
    
        data_pickle_dir = dataDir3 + predicted_video_dir + 'frames_pickle_result/'
        ResoDataGenerateObj = samplingResoDataGenerate()
        intervalFlag = 'frame'
        all_config_flag = True
        config_est_frm_arr, acc_frame_arr, spf_frame_arr = ResoDataGenerateObj.get_data_numpy(data_pose_keypoint_dir, data_pickle_dir,intervalFlag, all_config_flag)
        
        print ("get_prediction_acc_delay config_est_frm_arr: ", config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        
        FRM_LEN = config_est_frm_arr.shape[1]
        
        frm_start_indx = 0
        
        acc_accumulate = 0.0
        profiling_time_accumulate = 0.0
        
        pareto_bound_flag = 0
        while (frm_start_indx < FRM_LEN-(interval_len_time*PLAYOUT_RATE)):  # neglect the last interval for convenience of simulation
            
            # profile to get ocnfiguration
            reso_indx, profiling_time = self.select_optimal_configuration_above_accuracy(min_acc_thres, config_est_frm_arr, acc_frame_arr, spf_frame_arr, frm_start_indx, segment_len_time, pareto_bound_flag)
            
            acc_accumulate += 1.0 * segment_len_time * PLAYOUT_RATE  # profiling accuracy is 1.0
            
            profiling_time_accumulate += profiling_time
            frm_start_indx += (segment_len_time * PLAYOUT_RATE )
            rest_segment_frm_len = (interval_len_time - segment_len_time) * PLAYOUT_RATE
            # rest of segment time
            acc_accumulate += np.sum(acc_frame_arr[reso_indx, frm_start_indx: frm_start_indx+rest_segment_frm_len])
            
            profiling_time_accumulate += np.sum(spf_frame_arr[reso_indx, frm_start_indx: frm_start_indx+rest_segment_frm_len])
         
            frm_start_indx += rest_segment_frm_len
            # use predicted result to apply to this new video and get delay and accuracy
            # get the average acc with this frame index   # get the average acc with this frame index
        
        acc = acc_accumulate/frm_start_indx
        
        spent_time = profiling_time_accumulate - frm_start_indx * (1/PLAYOUT_RATE)  # processed time - playout time
        spent_time /= frm_start_indx
        print("acc_accumulate: ", min_acc_thres, acc, spent_time)
        
        
        return acc, spent_time  


    def execute_video_analytics_simulation(self):
        segment_len_time = 1  # 1 sec
        interval_len_time = 4 # 4 sec
        min_acc_threshold_lst = [0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
        
        acc_lst = []
        SPF_spent_lst = []
        
        for min_acc_thres in min_acc_threshold_lst:
            
            acc_average = 0.0
            spf_average = 0.0
            analyzed_video_lst = video_dir_lst[0:10]
            for predicted_video_dir in analyzed_video_lst:
                predicted_video_frm_dir = dataDir3 + "_".join(predicted_video_dir[:-1].split("_")[1:]) + "_frames/"
                
                print ("predicted_video_frm_dir: ", predicted_video_frm_dir)  # ../input_output/speaker_video_dataset/sample_03_frames/
            
                acc, spf = self.execute_video_periodic_time_profiling(predicted_video_dir, min_acc_thres, interval_len_time, segment_len_time)
                acc_average += acc
                spf_average += spf
            
            acc_lst.append(acc_average/len(analyzed_video_lst))
            
            SPF_spent_lst.append(spf_average/len(analyzed_video_lst))
            
        print("acc_lst, SPF_spent_lst: ", acc_lst, SPF_spent_lst)
        
        
if __name__== "__main__": 
    
    PeriodicOnlineProfilingObj = PeriodicOnlineProfiling()
    PeriodicOnlineProfilingObj.execute_video_analytics_simulation()
    
    
            
            
    