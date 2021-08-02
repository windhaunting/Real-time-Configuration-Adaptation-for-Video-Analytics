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


from data_preproc import input_dir
from data_preproc import bb_intersection_over_union
from data_preproc import write_numpy_into_file
from data_preproc import write_pickle_data

from read_feature_speaker import get_data_numpy
from read_feature_speaker import file_dir_lst


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from classifierForSwitchConfig.common_classifier import getParetoBoundary_boundeAcc

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/../..')


from profiling.common_prof import dataDir3
from profiling.common_prof import PLAYOUT_RATE


dataDir3 = "../" + dataDir3

# speaker detection video analytics based on offline once time profiling  





class OfflineOnceTimeProfiling(object):
    def __init__(self):
        pass
    
    def read_video_frm(self, predicted_video_frm_dir):
        
        imagePathLst = sorted(glob(predicted_video_frm_dir + "*.jpg"))  # , key=lambda filePath: int(filePath.split('/')[-1][filePath.split('/')[-1].find(start)+len(start):filePath.split('/')[-1].rfind(end)]))          # [:75]   5 minutes = 75 segments

        #print ("imagePathLst: ", len(imagePathLst), imagePathLst[0])
        
        return imagePathLst
    
    def get_config_id_dictionary(self):
        dict_all_configName_to_id = {0: '1120x832-25-RCNN', 1: '960x720-25-RCNN', 2: '1120x832-15-RCNN', 3: '640x480-25-RCNN', 
                         4: '960x720-15-RCNN', 5: '480x352-25-RCNN', 6: '1120x832-10-RCNN', 7: '640x480-15-RCNN', 
                         8: '960x720-10-RCNN', 9: '320x240-25-RCNN', 10: '480x352-15-RCNN', 11: '640x480-10-RCNN',
                         12: '1120x832-5-RCNN', 13: '320x240-15-RCNN', 14: '960x720-5-RCNN', 15: '480x352-10-RCNN', 
                         16: '320x240-10-RCNN', 17: '640x480-5-RCNN', 18: '480x352-5-RCNN', 19: '1120x832-2-RCNN', 
                         20: '960x720-2-RCNN', 21: '320x240-5-RCNN', 22: '640x480-2-RCNN', 23: '1120x832-1-RCNN', 
                         24: '960x720-1-RCNN', 25: '480x352-2-RCNN', 26: '320x240-2-RCNN', 27: '640x480-1-RCNN', 
                         28: '480x352-1-RCNN', 29: '320x240-1-RCNN'}
        
        dict_all_id_to_configName = {val:key for key, val in dict_all_configName_to_id.items()}
        
        
        resolution_to_original_idx = {'1120x832': 0, '960x720': 1, '640x480': 2, '480x352': 3, '320x240': 4}
        print("dict_all_id_to_configName:", dict_all_id_to_configName)
        
        return dict_all_configName_to_id, resolution_to_original_idx
        
    def apply_func_compute_bboverunion(self, combine_stk):
        
        bbox_a = combine_stk[0:4]
        bbox_b = combine_stk[4:]
        
        #print ("combine_stk: ", combine_stk.shape, bbox_a, bbox_b)
        iou = bb_intersection_over_union(bbox_a, bbox_b)
        
        return iou
    
    def extend_expensive_config_to_all_configs(self, speaker_box_arr, acc_frame_arr, spf_frame_arr, output_dir):
        #input speaker_box_arr: (5, 37977, 4)
        # output shape: speaker_box_arr (30, 37977, 4)    # 30 configurations
        
        print ("extend_expensive_config_to_all_configs configs: ", speaker_box_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        dict_all_configName_to_id, resolution_to_original_idx = self.get_config_id_dictionary()
        
        
        FRM_NUM = speaker_box_arr.shape[1]
        
        new_speaker_box_arr = np.zeros((30, FRM_NUM, 4)) # [[]]*30 # np.zeros((30, FRM_NUM, 4)) #  [[]]*30   # np.zeros((30, FRM_NUM))
        for i in range(0, 30):
            new_speaker_box_arr[i] = speaker_box_arr[0]
            
        new_spf_frame_arr = np.zeros((30, FRM_NUM))  # (30, FRM_NUM))  # [[]]*30
        for i in range(0, 30):
            new_spf_frame_arr[i] = spf_frame_arr[0]//5
            
        new_acc_frame_arr = np.zeros((30, FRM_NUM))
        
        for i in range(0, 30):
            new_acc_frame_arr[i] = acc_frame_arr[0]
            
        for idx, config_name in dict_all_configName_to_id.items():
            
            reso = config_name.split('-')[0]
            
            frm_sampling_rate = int(config_name.split('-')[1])
            
            origin_idx = resolution_to_original_idx[reso]
            
            print("origin_idx:", origin_idx, frm_sampling_rate)
            
            seconds = FRM_NUM//25
            frm_start = 0
            array_box_frm = np.empty((1, 4))
            array_spf_frm = np.empty((1,1))
            for sec in range(0, seconds):
                
                # intervals
                intervals = 25//frm_sampling_rate
                
                for inter in range(frm_sampling_rate):
                    frm_box_array = np.tile(speaker_box_arr[origin_idx][frm_start], (intervals, 1))   #   [list(speaker_box_arr[origin_idx][frm_start])]*intervals
                    #print("frm_array:", frm_array.shape, array_box_frm.shape, frm_array, array_box_frm) 
                    array_box_frm = np.concatenate((array_box_frm, frm_box_array), axis=0)
                    
                    frm_spf_array = np.tile(np.asarray([spf_frame_arr[origin_idx][frm_start]/intervals]),(intervals, 1))
                    #print("frm_spf_array:", array_spf_frm.shape, frm_spf_array.shape, array_spf_frm, frm_spf_array) 
                    array_spf_frm = np.concatenate((array_spf_frm, frm_spf_array), axis=0)
                    
                    frm_start += intervals
                    
                    if frm_start >= FRM_NUM:
                        break
            
            #print("array_box_frm shape: ", new_speaker_box_arr.shape, array_box_frm.shape)
            max_frm_len = min(array_box_frm.shape[0], array_spf_frm.shape[0], FRM_NUM)
            new_speaker_box_arr[idx][0:max_frm_len] = array_box_frm[0:max_frm_len]
            array_spf_frm = array_spf_frm.ravel()
            new_spf_frame_arr[idx][0:max_frm_len] = array_spf_frm[0:max_frm_len]
            
            #print("speaker_box_arr[idx]: ",idx, new_speaker_box_arr[idx].shape,new_speaker_box_arr[idx].shape, speaker_box_arr[0].shape)  # np.asarray(new_speaker_box_arr[idx]).shape)
            
            combine_stk = np.concatenate((new_speaker_box_arr[idx], speaker_box_arr[0]), axis=1)  # with ground truth
            #print("combine_stk[idx]: ",idx, combine_stk.shape)  # np.asarray(new_speaker_box_arr[idx]).shape)
            new_acc_frame_arr[idx] = np.apply_along_axis(self.apply_func_compute_bboverunion, 1, combine_stk)
            #print("new_acc_frame_arr[idx]: ",idx, new_acc_frame_arr[idx])  # np.asarray(new_speaker_box_arr[idx]).shape)
            
        print("new_speaker_box_arr: ",idx, new_speaker_box_arr.shape, new_spf_frame_arr.shape, new_acc_frame_arr.shape)
        
        out_pickle_file_box = output_dir + "single_speaker_box.npy"
        write_numpy_into_file(new_speaker_box_arr, out_pickle_file_box)

        out_pickle_file_acc = output_dir + "single_acc.npy"
        write_numpy_into_file(new_acc_frame_arr, out_pickle_file_acc)
        
        out_pickle_file_spf = output_dir + "single_spf.npy"
        write_numpy_into_file(new_spf_frame_arr, out_pickle_file_spf)
        
        return new_speaker_box_arr, new_acc_frame_arr, new_spf_frame_arr
        

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


    def select_optimal_configuration_above_accuracy(self, min_acc_thres, acc_frame_arr, spf_frame_arr, frm_start_indx, segment_len_time, pareto_bound_flag):
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
        
        
    def execute_video_peridoic_update_profiling(self, predicted_video_dir, min_acc_thres, interval_len_time, segment_len_time):
                
            
        speaker_box_arr, acc_frame_arr, spf_frame_arr = get_data_numpy(input_dir + predicted_video_dir)

        print ("get_prediction_acc_delay input_dir: ", speaker_box_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        out_dir_new_config_arr =  input_dir + predicted_video_dir + "out_dir_new_config_arr/"
        if not os.path.exists(out_dir_new_config_arr):
            os.mkdir(out_dir_new_config_arr)
            speaker_box_arr, acc_frame_arr, spf_frame_arr = self.extend_expensive_config_to_all_configs(speaker_box_arr, acc_frame_arr, spf_frame_arr, out_dir_new_config_arr)
        else:
            speaker_box_arr, acc_frame_arr, spf_frame_arr = get_data_numpy(out_dir_new_config_arr)
        
        FRM_LEN = speaker_box_arr.shape[1]
        
        #periodic update profiling
        
        frm_start_indx = 0
        
        acc_accumulate = 0.0
        profiling_time_accumulate = 0.0
        
        pareto_bound_flag = 0
        while (frm_start_indx < FRM_LEN-(interval_len_time*PLAYOUT_RATE)):  # neglect the last interval for convenience of simulation
            
            # profile to get ocnfiguration
            reso_indx, profiling_time = self.select_optimal_configuration_above_accuracy(min_acc_thres, acc_frame_arr, spf_frame_arr, frm_start_indx, segment_len_time, pareto_bound_flag)
            
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


    def execute_video_peridoic_update_profiling_one_second(self, predicted_video_dir, min_acc_thres, interval_len_time, segment_len_time):
                
        
        output_pickle_dir = input_dir + predicted_video_dir + "data_instance_xy/minAcc_" + str(min_acc_thres) + "/"

        speaker_box_arr, acc_frame_arr, spf_frame_arr = get_data_numpy(input_dir + predicted_video_dir)

        print ("get_prediction_acc_delay input_dir: ", speaker_box_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        out_dir_new_config_arr =  input_dir + predicted_video_dir + "out_dir_new_config_arr/"
        if not os.path.exists(out_dir_new_config_arr):
            os.mkdir(out_dir_new_config_arr)
            speaker_box_arr, acc_frame_arr, spf_frame_arr = self.extend_expensive_config_to_all_configs(speaker_box_arr, acc_frame_arr, spf_frame_arr, out_dir_new_config_arr)
        else:
            speaker_box_arr, acc_frame_arr, spf_frame_arr = get_data_numpy(out_dir_new_config_arr)
        
        FRM_LEN = speaker_box_arr.shape[1]
        
        #periodic update profiling
        
        frm_start_indx = 0
        
        acc_lst = []
        time_spf_lst = []
        profiling_time_accumulate = 0.0
        
        pareto_bound_flag = 0
        while (frm_start_indx < FRM_LEN-(interval_len_time*PLAYOUT_RATE)):  # neglect the last interval for convenience of simulation
            
            # profile to get ocnfiguration
            reso_indx, profiling_time = self.select_optimal_configuration_above_accuracy(min_acc_thres, acc_frame_arr, spf_frame_arr, frm_start_indx, segment_len_time, pareto_bound_flag)
            
            acc_lst  += [1.0] * segment_len_time * PLAYOUT_RATE  # profiling accuracy is 1.0
            
            rest_segment_frm_len = (interval_len_time - segment_len_time) * PLAYOUT_RATE
            acc_lst += list(acc_frame_arr[reso_indx, frm_start_indx: frm_start_indx+rest_segment_frm_len])
            
            profiling_time_accumulate += profiling_time
            
            time_spf_lst += [profiling_time/(segment_len_time * PLAYOUT_RATE)] * (segment_len_time * PLAYOUT_RATE)
            
            
            time_spf_lst += list(spf_frame_arr[reso_indx, frm_start_indx: frm_start_indx+rest_segment_frm_len])
            
            frm_start_indx += (segment_len_time * PLAYOUT_RATE )
            # rest of segment time
                     
            frm_start_indx += rest_segment_frm_len
            # use predicted result to apply to this new video and get delay and accuracy
            # get the average acc with this frame index   # get the average acc with this frame index
        
        tmp_acc_lst = []
        i = 0
        interval_sec_frm = 25    # 1 sec 25 frame
        while (i < len(acc_lst)):
            if i+ interval_sec_frm < len(acc_lst):
                
                tmp_acc_lst.append(sum(acc_lst[i:i+interval_sec_frm])/interval_sec_frm)
            
            i += interval_sec_frm
                
        tmp_spf_lst = []
        i = 0
        interval_sec_frm = 25    # 1 sec 25 frame
        while (i < len(time_spf_lst)):
            if i+ interval_sec_frm < len(time_spf_lst):
                
                tmp_spf_lst.append(sum(time_spf_lst[i:i+interval_sec_frm]))     # interval total processing time
            
            i += interval_sec_frm
        
        
        arr_acc = np.asarray(tmp_acc_lst)
        arr_spf = np.asarray(tmp_spf_lst)
        print("arr_acc: ", arr_acc, arr_spf)
            
        detect_out_result_dir = output_pickle_dir + "video_applied_detection_result/"
        if not os.path.exists(detect_out_result_dir):
            os.mkdir(detect_out_result_dir)

        arr_acc_segment_file = detect_out_result_dir + "periodic_adaptation_arr_acc_segment_.pkl"
        arr_spf_segment_file = detect_out_result_dir + "periodic_adaptation_arr_spf_segment_.pkl"
        write_pickle_data(arr_acc, arr_acc_segment_file)
        write_pickle_data(arr_spf, arr_spf_segment_file)
        
        return arr_acc, arr_spf  
    
    
    
    def execute_video_analytics_simulation(self):
        segment_len_time = 1  # 1 sec
        interval_len_time = 4 # 4 sec
        min_acc_threshold_lst = [0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
        
        acc_lst = []
        SPF_spent_lst = []
        
        for min_acc_thres in min_acc_threshold_lst[1:2]:
            
            acc_average = 0.0
            spf_average = 0.0
            analyzed_video_lst = file_dir_lst[0:1]
            for predicted_video_dir in analyzed_video_lst:
                predicted_video_frm_dir = dataDir3 + "_".join(predicted_video_dir[:-1].split("_")[1:]) + "_frames/"
                
                print ("predicted_video_frm_dir: ", predicted_video_frm_dir)  # ../input_output/speaker_video_dataset/sample_03_frames/
                            
                #acc, spf = self.execute_video_peridoic_update_profiling(predicted_video_dir, min_acc_thres, interval_len_time, segment_len_time)
                
                acc, spf = self.execute_video_peridoic_update_profiling_one_second(predicted_video_dir, min_acc_thres, interval_len_time, segment_len_time)
                
                xx
                acc_average += acc
                spf_average += spf
            
            acc_lst.append(acc_average/len(analyzed_video_lst))
            
            SPF_spent_lst.append(spf_average/len(analyzed_video_lst))
            
        print("acc_lst, SPF_spent_lst: ", acc_lst, SPF_spent_lst)
        
        
if __name__== "__main__": 
    OfflineOnceTimeProfilingObj = OfflineOnceTimeProfiling()
    OfflineOnceTimeProfilingObj.execute_video_analytics_simulation()
    
    
            
            