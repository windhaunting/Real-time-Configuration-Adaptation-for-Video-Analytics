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


from data_file_process import write_pickle_data


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/../..')


from profiling.common_prof import dataDir3
from profiling.common_prof import PLAYOUT_RATE
from profiling.common_prof import frameRates
from profiling.common_prof import computeOKSACC

from classifierForSwitchConfig.common_classifier import getParetoBoundary_boundeAcc

dataDir3 = "../" + dataDir3

# video analytics based on offline once time profiling 

class OfflineOnceTimeProfiling(object):
    def __init__(self):
        pass
    
    def read_video_frm(self, predicted_video_frm_dir):
        
        imagePathLst = sorted(glob(predicted_video_frm_dir + "*.jpg"))  # , key=lambda filePath: int(filePath.split('/')[-1][filePath.split('/')[-1].find(start)+len(start):filePath.split('/')[-1].rfind(end)]))          # [:75]   5 minutes = 75 segments

        #print ("imagePathLst: ", len(imagePathLst), imagePathLst[0])
        
        return imagePathLst       
    
    def calculate_config_frm_acc(self, ests_arr, gts_arr):
        #each input it's the array of all frames
    
        
        FRM_LEN = ests_arr.shape[0]
        acc_one_config_arr = np.zeros(FRM_LEN)
        for i,(ets, gts) in enumerate(zip(ests_arr,gts_arr)):
            acc_one_config_arr[i] = computeOKSACC(ets,gts)
            #print ("ets, gts: ", ets.shape, gts.shape, acc_one_config_arr[i])
            
     
        return acc_one_config_arr

    """
    def get_all_configuration_acc_spf(self, config_est_frm_arr, acc_frame_arr, spf_frame_arr):
        # input: config_est_frm_arr: (5, frame_num, 17, 3) acc_frame_arr: (5, frame_num),  spf_frame_arr: (5, frame_num)
        # output: acc_config_arr: (5, 6, frame_num), spf_frame_arr: ((5, 6, frame_num)
        
        # frameRates = [25, 15, 10, 5, 2, 1] 
        reso_num = config_est_frm_arr.shape[0]
        rate_num = len(frameRates)
        
        acc_config_arr = np.zeros((reso_num, rate_num))
        
        for i, fr in enumerate(frameRates):
            
            interval = int(PLAYOUT_RATE/fr)
            
            print ("config_est_frm_arr[0, ::interval, :, :] shape: ", config_est_frm_arr[i, ::interval, :, :].shape)
            
            acc_one_config_arr = self.calculate_config_frm_acc(config_est_frm_arr[0, ::interval, :, :], config_est_frm_arr[i, ::interval, :, :])
            
            print ("acc_one_config_arr: ", acc_one_config_arr.shape, acc_one_config_arr)
    """
    
    def select_config(self, acc_frame_arr, spf_frame_arr, profile_frame_length, min_acc_thres): 
        # select a config that can achieve min accuracy threshold with min spf
        
        min_spf = float('inf')
        ans_indx = 0
        
        average_profile_acc_arr = np.mean(acc_frame_arr[:, 0:profile_frame_length], axis=1, keepdims=True)
        average_profile_spf_arr = np.mean(spf_frame_arr[:, 0:profile_frame_length], axis=1, keepdims=True)
        print("aaaaaaverage_profile_spf_arr: ", average_profile_spf_arr.shape)
        for idx, acc in np.ndenumerate(average_profile_acc_arr): 
            spf = average_profile_spf_arr[idx[0]]
            
            if acc >= min_acc_thres and spf < min_spf:
                min_spf = spf
                ans_indx = idx[0]
        
        return ans_indx      # index
                
        
    def select_optimal_configuration_above_accuracy(self, min_acc_thres, config_est_frm_arr, acc_frame_arr, spf_frame_arr, interval_len_time, pareto_bound_flag):
        # select the best configuration with the minimum efficiency above the accuracy threshold
        profile_frame_length = interval_len_time * PLAYOUT_RATE
        
        
        #average_profile_acc_arr = np.mean(acc_frame_arr[:, 0:profile_frame_length], axis=1, keepdims=True)
        
        #print("average_profile_acc_arr: ", average_profile_acc_arr.shape)
        
        if pareto_bound_flag == 0:   # all the configurations
            ans_indx = self.select_config(acc_frame_arr,spf_frame_arr, profile_frame_length, min_acc_thres)
            
        elif pareto_bound_flag == 1:    # pareto boundary with minimum accuracy threshold
            
            first_segment_indx = 25
            acc_arr = np.mean(acc_frame_arr[:, 0:first_segment_indx], axis=1)
            spf_arr = np.mean(spf_frame_arr[:, 0:first_segment_indx], axis=1)
            
            print("acc_arr, spf_arr: ", acc_arr.shape, spf_arr.shape)
            lst_idx_selected_config = getParetoBoundary_boundeAcc(acc_arr, spf_arr, min_acc_thres)
            
            #lst_idx_selected_config.remove(0)       # if 0 in lst_idx_selected_config
    
            print ("lst_idx_selected_config: ", len(lst_idx_selected_config), lst_idx_selected_config)
            
            acc_frame_arr = acc_frame_arr[lst_idx_selected_config, :]
            spf_frame_arr = spf_frame_arr[lst_idx_selected_config, :]
            
            ans_indx = self.select_config(acc_frame_arr, spf_frame_arr, profile_frame_length, min_acc_thres)
         
        print("ans_indx: ", ans_indx)
        profiling_time = np.sum(spf_frame_arr[:, 0:profile_frame_length])
            
        #print("profiling_time: ", profiling_time)
        return ans_indx, profiling_time
    

        
    def execute_video_once_time_offline_profiling(self, predicted_video_dir, min_acc_thres, interval_len_time):
        
        data_pose_keypoint_dir = dataDir3 + predicted_video_dir

        data_pickle_dir = dataDir3 + predicted_video_dir + 'frames_pickle_result/'
        ResoDataGenerateObj = samplingResoDataGenerate()
        intervalFlag = 'frame'
        all_config_flag = True
        config_est_frm_arr, acc_frame_arr, spf_frame_arr = ResoDataGenerateObj.get_data_numpy(data_pose_keypoint_dir, data_pickle_dir, intervalFlag, all_config_flag)
     
        print ("get_prediction_acc_delay config_est_frm_arr: ",data_pose_keypoint_dir, data_pickle_dir, config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        #self.get_all_configuration_acc_spf(config_est_frm_arr, acc_frame_arr, spf_frame_arr)
        
        # profile to get ocnfiguration
        pareto_bound_flag = 0
        reso_indx, profiling_time = self.select_optimal_configuration_above_accuracy(min_acc_thres, config_est_frm_arr, acc_frame_arr, spf_frame_arr, interval_len_time, pareto_bound_flag)
    
        # use predicted result to apply to this new video and get delay and accuracy
        # get the average acc with this frame index
        
        print("aaaaaaaacc_frame_arr shape: ", acc_frame_arr[reso_indx].shape, reso_indx,  acc_frame_arr.shape)
        
        
        average_acc = np.mean(acc_frame_arr[reso_indx])
        
        
        frm_len = acc_frame_arr.shape[1]
        
        spent_time = np.sum(spf_frame_arr[reso_indx]) - frm_len * (1/PLAYOUT_RATE)    # delay;  processed time - playout time
            
        SPF_spent = spent_time/frm_len
        print("average_acc: ", average_acc, SPF_spent)
        
        return average_acc, SPF_spent



    def execute_video_once_time_no_adaptation_arr_spf(self, predicted_video_dir, min_acc_thres, interval_len_time):
                
            
        output_pickle_dir = dataDir3 + predicted_video_dir + "jumping_number_result/jumpingNumber_resolution_selection/intervalFrm-10_speedType-ema_minAcc-" + str(min_acc_thres) + "/"    
      
        
        data_pose_keypoint_dir = dataDir3 + predicted_video_dir

        data_pickle_dir = dataDir3 + predicted_video_dir + 'frames_pickle_result/'
        ResoDataGenerateObj = samplingResoDataGenerate()
        intervalFlag = 'frame'
        all_config_flag = True
        config_est_frm_arr, acc_frame_arr, spf_frame_arr = ResoDataGenerateObj.get_data_numpy(data_pose_keypoint_dir, data_pickle_dir, intervalFlag, all_config_flag)
     
        print ("get_prediction_acc_delay config_est_frm_arr: ",data_pose_keypoint_dir, data_pickle_dir, config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        #self.get_all_configuration_acc_spf(config_est_frm_arr, acc_frame_arr, spf_frame_arr)
        
        # profile to get ocnfiguration
        pareto_bound_flag = 0
        reso_indx, profiling_time = self.select_optimal_configuration_above_accuracy(min_acc_thres, config_est_frm_arr, acc_frame_arr, spf_frame_arr, interval_len_time, pareto_bound_flag)
    
        # use predicted result to apply to this new video and get delay and accuracy
        # get the average acc with this frame index
        
        print("aaaaaaaacc_frame_arr shape: ", acc_frame_arr[reso_indx].shape, reso_indx,  acc_frame_arr.shape)
        
        
        # use predicted result to apply to this new video and get delay and accuracy
        #get the 
        # get the average acc with this frame index
        frm_len = acc_frame_arr.shape[1]
        
        
        arr_acc = acc_frame_arr[reso_indx]
        
        arr_delay = spf_frame_arr[reso_indx] - (1.0/PLAYOUT_RATE)
            
        arr_delay[arr_delay<0] = 0
        
        print("arr_acc: ", reso_indx, arr_acc, arr_delay)
        
        
        detect_out_result_dir = output_pickle_dir + "video_applied_detection_result/"
        if not os.path.exists(detect_out_result_dir):
            os.mkdir(detect_out_result_dir)

        arr_acc_segment_file = detect_out_result_dir + "no_adaptation_arr_acc_segment_.pkl"
        arr_delay_up_to_segment_file = detect_out_result_dir + "no_adaptation_arr_delay_up_to_segment_.pkl"
        write_pickle_data(arr_acc, arr_acc_segment_file)
        write_pickle_data(arr_delay, arr_delay_up_to_segment_file)
        
        
        
        return arr_acc, arr_delay
        
        
    def execute_video_analytics_simulation(self):
        interval_len_time = 10
        min_acc_threshold_lst = [0.9, 0.92, 0.94, 0.96, 0.98, 1.0]

        acc_lst = []
        SPF_spent_lst = []
        
        for min_acc_thres in min_acc_threshold_lst[3:4]:
            
            acc_average = 0.0
            spf_average = 0.0
            analyzed_video_lst = video_dir_lst[0:1]
            for predicted_video_dir in analyzed_video_lst:
                predicted_video_frm_dir = dataDir3 + "_".join(predicted_video_dir[:-1].split("_")[1:]) + "_frames/"
                
                print ("predicted_video_frm_dir: ", predicted_video_frm_dir)  # ../input_output/speaker_video_dataset/sample_03_frames/
            
                #acc, spf = self.execute_video_once_time_offline_profiling(predicted_video_dir, min_acc_thres, interval_len_time)

                acc, spf = self.execute_video_once_time_no_adaptation_arr_spf(predicted_video_dir, min_acc_thres, interval_len_time)
                
                xx
                acc_average += acc
                spf_average += spf
            
            acc_lst.append(acc_average/len(analyzed_video_lst))
            
            SPF_spent_lst.append(spf_average/len(analyzed_video_lst))
            
        print("acc_lst, SPF_spent_lst: ", acc_lst, SPF_spent_lst)



if __name__== "__main__": 
    OfflineOnceTimeProfilingObj = OfflineOnceTimeProfiling()
    OfflineOnceTimeProfilingObj.execute_video_analytics_simulation()
    
    
            
            