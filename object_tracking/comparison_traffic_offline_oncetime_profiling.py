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


from tracking_data_preprocess import bb_intersection_over_union
from tracking_data_preprocess import write_numpy_into_file
from tracking_data_preprocess import read_json_dir
from tracking_data_preprocess import write_pickle_data

from detection_applied_result import VideoApply

# training and test the adaptive configuration

from tracking_get_training_data import *
#from tracking_get_training_data import data_dir
#from tracking_get_training_data import object_name

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from classifierForSwitchConfig.common_classifier import getParetoBoundary_boundeAcc

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/../..')


from profiling.common_prof import dataDir3
from profiling.common_prof import PLAYOUT_RATE


# car traffic video analytics based on offline once time profiling  



class OfflineOnceTimeProfiling(object):
    def __init__(self):
        pass
    
    def read_video_frm(self, predicted_video_frm_dir):
        
        imagePathLst = sorted(glob(predicted_video_frm_dir + "*.jpg"))  # , key=lambda filePath: int(filePath.split('/')[-1][filePath.split('/')[-1].find(start)+len(start):filePath.split('/')[-1].rfind(end)]))          # [:75]   5 minutes = 75 segments

        #print ("imagePathLst: ", len(imagePathLst), imagePathLst[0])
        
        return imagePathLst

        
    def execute_video_once_time_offline_profiling(self, data_dir, predicted_video_dir, min_acc_thres, interval_len_time):
                        
            dict_detection_reso_video = read_json_dir(predicted_video_dir)

            data_obj = DataGenerate(data_dir)


            print ("execute_video_once_time_offline_profiling input_dir: ", len(dict_detection_reso_video))
           
            
            output_pickle_dir = predicted_video_dir +  "data_instance_xy/"  + "minAcc_" + str(min_acc_thres) + "/"


            #use 10 second to decide a configuration
            frm_no = 1000  # test only 10000 for paper ploting experiment len(dict_detection_reso_video[reso_list[0]])  #   min(len(imagePathLst), spf_arr.shape[1])
            
            ans_jfr, ans_reso_indx, aver_acc = data_obj.predict_next_configuration_jumping_frm_reso(dict_detection_reso_video, min_acc_thres, 1, frm_no)
        
                
            # profile to get ocnfiguration
            pareto_bound_flag = 0
            #ans_indx, profiling_time = self.select_optimal_configuration_above_accuracy(min_acc_thres, acc_frame_arr, spf_frame_arr, interval_len_time, pareto_bound_flag)
            
            
            # use predicted result to apply to this new video and get delay and accuracy
            #get the 
            # get the average acc with this frame index
            
            highest_reso = reso_list[0]
            curre_reso = reso_list[ans_reso_indx]

            
            current_frm_indx = 1                   # 1  starting index = 0  left frame no = 1
            arr_acc_lst = []
            arr_time_lst = []
            
            videoApplyObj = VideoApply()            

            while current_frm_indx < frm_no:
                
                #dict_cars_each_jumped_frm_higest_reso =  dict_detection_reso_video[highest_reso][str(current_frm_indx)][object_name]

                
                acc_accumulate, time_accumualate, calculated_frm_num = videoApplyObj.get_online_video_analytics_accuracy_spf(data_dir, dict_detection_reso_video, current_frm_indx, ans_jfr, highest_reso, curre_reso)
        
                    
                #aver_acc = get_numpy_arr_all_jumpingFrameInterval_resolution(dict_detection_reso_video, current_frm_indx, ans_jfr, highest_reso, curre_reso)
        
                arr_acc_lst.append(acc_accumulate/(ans_jfr+1))
                latency = max(0, time_accumualate)
                arr_time_lst.append(latency)
                
                #time_accumualate =  dict_detection_reso_video[highest_reso][str(current_frm_indx)]['spf']

                current_frm_indx += ans_jfr
            
            arr_acc = np.asarray(arr_acc_lst)
            arr_delay = np.asarray(arr_time_lst)
            print("arr_acc: ", arr_acc, arr_delay)
            

            detect_out_result_dir = output_pickle_dir + "video_applied_detection_result/"
            if not os.path.exists(detect_out_result_dir):
                os.mkdir(detect_out_result_dir)
    
            arr_acc_segment_file = detect_out_result_dir + "no_adaptation_arr_acc_segment_.pkl"
            arr_delay_up_to_segment_file = detect_out_result_dir + "no_adaptation_arr_spf_segment_.pkl"
            write_pickle_data(arr_acc, arr_acc_segment_file)
            write_pickle_data(arr_delay, arr_delay_up_to_segment_file)
            
            return arr_acc, arr_delay



    def execute_video_once_time_offline_profiling_each_second(self, data_dir, predicted_video_dir, min_acc_thres, interval_len_time):
                
            # get each frame's accuracy and processing speed
        
            dict_detection_reso_video = read_json_dir(predicted_video_dir)

            data_obj = DataGenerate(data_dir)


            print ("execute_video_once_time_offline_profiling_each_frame input_dir: ", len(dict_detection_reso_video))
           
            
            output_pickle_dir = predicted_video_dir +  "data_instance_xy/"  + "minAcc_" + str(min_acc_thres) + "/"


            #use 10 second to decide a configuration
            frm_no = 3000  # test only 10000 for paper ploting experiment len(dict_detection_reso_video[reso_list[0]])  #   min(len(imagePathLst), spf_arr.shape[1])
            
            ans_jfr, ans_reso_indx, aver_acc = data_obj.predict_next_configuration_jumping_frm_reso(dict_detection_reso_video, min_acc_thres, 1, frm_no)
        
                
            # profile to get ocnfiguration
            pareto_bound_flag = 0
            #ans_indx, profiling_time = self.select_optimal_configuration_above_accuracy(min_acc_thres, acc_frame_arr, spf_frame_arr, interval_len_time, pareto_bound_flag)
            
            
            # use predicted result to apply to this new video and get delay and accuracy
            #get the 
            # get the average acc with this frame index
            
            highest_reso = reso_list[0]
            curre_reso = reso_list[ans_reso_indx]

            
            current_frm_indx = 1                   # 1  starting index = 0  left frame no = 1
            arr_acc_lst = []
            arr_time_lst = []
            
            videoApplyObj = VideoApply()            

            while current_frm_indx < frm_no:
                
                #dict_cars_each_jumped_frm_higest_reso =  dict_detection_reso_video[highest_reso][str(current_frm_indx)][object_name]

                acc_lst, time_lst, calculated_frm_num = videoApplyObj.get_online_video_analytics_accuracy_spf_each_frame(data_dir, dict_detection_reso_video, current_frm_indx, ans_jfr, highest_reso, curre_reso)
        
                    
                #aver_acc = get_numpy_arr_all_jumpingFrameInterval_resolution(dict_detection_reso_video, current_frm_indx, ans_jfr, highest_reso, curre_reso)
                
                arr_acc_lst += acc_lst
                arr_time_lst += time_lst
                
                #time_accumualate =  dict_detection_reso_video[highest_reso][str(current_frm_indx)]['spf']

                current_frm_indx += ans_jfr
            
            
            tmp_acc_lst = []
            i = 0
            interval_sec_frm = 25    # 1 sec 25 frame
            while (i < len(arr_acc_lst)):
                if i+ interval_sec_frm < len(arr_acc_lst):
                    
                    tmp_acc_lst.append(sum(arr_acc_lst[i:i+interval_sec_frm])/interval_sec_frm)
                
                i += interval_sec_frm
                
                
            tmp_spf_lst = []
            i = 0
            interval_sec_frm = 25    # 1 sec 25 frame
            while (i < len(arr_time_lst)):
                if i+ interval_sec_frm < len(arr_time_lst):
                    
                    tmp_spf_lst.append(sum(arr_time_lst[i:i+interval_sec_frm]))
                
                i += interval_sec_frm
                
                
            arr_acc = np.asarray(tmp_acc_lst)
            arr_spf = np.asarray(tmp_spf_lst)
            print("arr_acc: ", arr_acc, arr_spf)
            

            detect_out_result_dir = output_pickle_dir + "video_applied_detection_result/"
            if not os.path.exists(detect_out_result_dir):
                os.mkdir(detect_out_result_dir)
    
            arr_acc_segment_file = detect_out_result_dir + "no_adaptation_arr_acc_segment_.pkl"
            arr_spf_segment_file = detect_out_result_dir + "no_adaptation_arr_spf_segment_.pkl"
            write_pickle_data(arr_acc, arr_acc_segment_file)
            write_pickle_data(arr_spf, arr_spf_segment_file)
            
            return arr_acc, arr_spf


    def execute_video_analytics_simulation(self, data_dir, video_dir_lst):
        interval_len_time = 10
        min_acc_threshold_lst = [0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
        
        acc_lst = []
        SPF_spent_lst = []
        
        for min_acc_thres in min_acc_threshold_lst[1:2]:
            
            acc_average = 0.0
            spf_average = 0.0
            analyzed_video_lst = video_dir_lst[0:1]    #[0:10]
            for predicted_video_dir in analyzed_video_lst:
                predicted_video_dir = predicted_video_dir + '/'
                predicted_video_frm_dir = dataDir3 + "_".join(predicted_video_dir.split('/')[-2].split("_")[1:]) + "_frames/"
                
                print ("predicted_video_frm_dir: ", predicted_video_frm_dir)  # ../input_output/speaker_video_dataset/sample_03_frames/
                            
                acc, spf = self.execute_video_once_time_offline_profiling_each_second(data_dir, predicted_video_dir, min_acc_thres, interval_len_time)
                
                
                acc_average += acc
                spf_average += spf
            
            acc_lst.append(acc_average/len(analyzed_video_lst))
            
            SPF_spent_lst.append(spf_average/len(analyzed_video_lst))
            
        print("acc_lst, SPF_spent_lst: ", acc_lst, SPF_spent_lst)




if __name__== "__main__": 
    
    data_obj = DataGenerate(data_dir)
    video_dir_lst = data_obj.video_dir_lst    # [5:6]   # [5, 15]

    OfflineOnceTimeProfilingObj = OfflineOnceTimeProfiling()
    OfflineOnceTimeProfilingObj.execute_video_analytics_simulation(data_dir, video_dir_lst)
    
    
            
            