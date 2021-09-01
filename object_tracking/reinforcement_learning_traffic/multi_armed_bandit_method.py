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
import itertools

from blist import blist
from collections import defaultdict

from fitting_accuracy_func import fit_function
from fitting_accuracy_func import expo_func
from common_plot import plotTwoLinesOneFigure
from common_plot import plotScatterLineOneFig
from common_plot import plotMultipleLinesOneFigure

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


# training and test the adaptive configuration
from tracking_get_training_data import data_dir
from tracking_get_training_data import min_acc_threshold
from tracking_get_training_data import max_jump_number
from tracking_get_training_data import interval_frm
from tracking_get_training_data import DataGenerate
from tracking_data_preprocess import read_json_dir
from tracking_data_preprocess import write_lst_to_csv
#from tracking_get_training_data import object_name


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, current_file_cur + '/../..') 

from profiling.common_prof import dataDir3
from profiling.common_prof import PLAYOUT_RATE
from profiling.common_prof import resoStrLst_OpenPose
from profiling.common_prof import NUM_KEYPOINT   
from profiling.common_prof import computeOKS_1to1
from profiling.writeIntoPickleConfigFrameAccSPFPoseEst import read_config_name_from_file


# here frame rate is the jumping frame interval

class MultiArmedConfig(object):
    
    def __init__(self):
        
        self.alpha = 0.8 # accuracy to reward ; the weight for accuracy coefficient
        self.beta = 0.02
        self.c = 0.002*200
        self.d = 0.5
        
        self.Total_CONFIG_NUM =  25*5             # K which is the K arms
        
    """
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
    """


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


    def fitting_accuracy(self, fr_acc_data_point, reso_acc_data_point):
        # with fitting line points to get the fitting function's coefficients
                
        popt_acc_fr = self.fit_accuracy_frame_rate(fr_acc_data_point)
        popt_acc_reso = self.fit_accuracy_resolution(reso_acc_data_point)
        
        return popt_acc_fr, popt_acc_reso
    
    """
    def get_instance_acc_with_fitting_function(self, popt_acc_fr, popt_acc_reso, curr_frm_rate, curr_reso):
        
        # get the instance accuracy with an already fitted function.  the input is a configuration  
        
        acc_fr = expo_func(curr_frm_rate, *popt_acc_fr)
        
        acc_reso = expo_func(curr_reso, *popt_acc_reso)
        
        instance_acc = acc_fr * acc_reso
        return instance_acc
    """


    def instance_reward(self,instance_acc, instance_processing_time):
        # get the instance reward from the accuracy and processing time
        # r = alpha*instance_acc + (1-alpha)*instance_processing_time
        
        inst_reward = self.alpha*instance_acc - self.beta*instance_processing_time
        
        return inst_reward


    """
    def get_numpy_arr_all_jumpingFrameInterval_resolution(self, dict_detection_reso_video, current_frm_indx, jump_frm_interval, highest_reso, curre_reso):
        # Inputout: current frm_index, current resolution,  and object tracking position results, and jumping_frame_interval (frame rate)
        # output: each frame's accuracy (accuracy interval)  or average accuracy
        
        dict_cars_each_jumped_frm_higest_reso =  dict_detection_reso_video[highest_reso][str(current_frm_indx)][object_name]
        
        
        jumping_frame_interval = 0
        arr_acc_interval = []   # np.zeros(jumping_frame_interval)       # the maximum 25 frame interval's acc in an array
        while(jumping_frame_interval < jump_frm_interval):
            next_frm_indx = current_frm_indx + jumping_frame_interval
            dict_cars_each_jumped_frm_curr_reso = dict_detection_reso_video[curre_reso][str(next_frm_indx)][object_name]
            
            #dict_cars_each_jumped_frm_other_reso = dict_detection_reso_video[curr_reso][str(next_frm_indx)][object_name]
            #print("dict_cars_each_jumped_frm_higest_reso: ", dict_cars_each_jumped_frm_higest_reso)
            #print("predict_next_configuration_jumping_frm_reso: xxx", dict_cars_each_jumped_frm_higest_reso)
            curr_acc = min_acc_threshold   # 0.0  # # no vechicle existing in the frame
            if dict_cars_each_jumped_frm_higest_reso is not None:
                #print("predict_next_configuration_jumping_frm_reso dict_cars_each_jumped_frm_higest_reso: ", dict_cars_each_jumped_frm_higest_reso)
                curr_acc = self.calculate_accuray_with_highest_reso(dict_cars_each_jumped_frm_higest_reso, highest_reso, dict_cars_each_jumped_frm_curr_reso, curre_reso, min_acc_threshold)
                                    
                #print ("curr_acc: ", curr_acc)
            arr_acc_interval.append(curr_acc)
            
            jumping_frame_interval += 1
            
            #print("arr_acc_interval: ", arr_acc_interval)

        #if len(arr_acc_interval) == 0:
        #    print("xxxxxxxxxxxxxxxxxx empty: ", arr_acc_interval)
        
        arr_acc_interval = np.asarray(arr_acc_interval)
        
        return arr_acc_interval
    """
    
    
    def get_accuracy_fitting_function_interval_time(self, data_obj, dict_detection_reso_video, update_interval_time, start_frame_indx):
        # get the accuracy in the interval time similar to profiling method, but it is not profiling
        # so we have to run the expensive configuration.
        # get the average accuracy and the configuration;  for three configurations that are enough
        
        # and the processing time in this update_interval_time
        # note: frame rate here is actually the jumping frame interval in the programming
        #print("config_est_frm_arr: ", config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        # select at least three points
        
        frm_lst = [2, 10, 25] #  [2,10, 25]  # [1, 10, 25]  # range(1, PLAYOUT_RATE+1)
        
        resolutions_lst = [0, 3, 4]  # [0, 2, 4]  # range(0, len(resoStrLst_OpenPose))   #  [int(x.split('x')[0]) for x in resoStrLst_OpenPose]
        
        #print('frm list: ', frm_lst, resolutions_lst)
        
        POINTS_NUM_SELECTED = len(frm_lst)
        
        selected_frm_rates = rand.sample(frm_lst, POINTS_NUM_SELECTED)
        
        selected_reso_indices = rand.sample(resolutions_lst, POINTS_NUM_SELECTED)
        
        #print('selected_frm_rates: ', selected_frm_rates, selected_reso_indices)
        # read 
        
        highest_reso = resoStrLst_OpenPose[0]        
        # get different accuracy with frame rate points  for frame rate point  to acc y axis 
        #fr_acc_data_point = [[PLAYOUT_RATE*1.0/frame_rate, self.calculate_frame_rate_accuracy(config_est_frm_arr, start_frame_indx, update_interval_time, frame_rate)] for frame_rate in selected_frm_rates] 
        
        fr_acc_data_point = [[PLAYOUT_RATE*1.0/frame_rate, np.average(data_obj.get_numpy_arr_all_jumpingFrameInterval_resolution(dict_detection_reso_video, start_frame_indx, frame_rate, highest_reso, highest_reso))] for frame_rate in selected_frm_rates] 
        
        fr_acc_data_point_jumping_interval = sorted([inter for inter, acc in fr_acc_data_point])
        fr_acc_data_point_acc = sorted([acc for inter, acc in fr_acc_data_point])
        
        fr_acc_data_point = [[fr_acc_data_point_jumping_interval[i], fr_acc_data_point_acc[i]] for i in range(0, len(fr_acc_data_point_jumping_interval))]
        print("fr_acc_data_point: ", fr_acc_data_point)
        
        # get different accuracy with resolution points
        
        dict_map_id_reso = {i: int(resoStrLst_OpenPose[i].split('x')[0]) for i in range(0, len(resoStrLst_OpenPose))}
        reso_acc_data_point = [[dict_map_id_reso[reso_id], np.average(data_obj.get_numpy_arr_all_jumpingFrameInterval_resolution(dict_detection_reso_video, start_frame_indx, 1, highest_reso, resoStrLst_OpenPose[reso_id]))] for reso_id in selected_reso_indices]
        
        print('reso_acc_data_point: ', fr_acc_data_point, reso_acc_data_point) 
        # fitting into function to get fitting function's coefficient
        popt_acc_fr, popt_acc_reso = self.fitting_accuracy(fr_acc_data_point, reso_acc_data_point)

        #print('popt_acc_fr: ', popt_acc_fr, popt_acc_reso)
        #print("test: ", spf_frame_arr.shape, spf_frame_arr[:, 1], spf_frame_arr[0][0:20:2].shape, spf_frame_arr[0][0:20:1])
        # for the three selected processing time and range
        
        acc_lst = []
        processing_time_lst = []       
        
        interval_seconds = update_interval_time*25//PLAYOUT_RATE
        
        for p in range(0, interval_seconds):
            tmp_time = 0.0
            for reso_ind in selected_reso_indices:
                jumping_frm_interval = 1                          #  because we run the frame rate=25, each frame is got, we so don't need to use each frame rate
                #print(":", spf_frame_arr[reso_ind].shape, start_frame_indx, jumping_frm_interval)
                #tmp_time += np.sum(spf_frame_arr[reso_ind][start_frame_indx:(start_frame_indx+PLAYOUT_RATE):jumping_frm_interval])  # here frame rate is the jumping interval
                
                current_tmp = sum([dict_detection_reso_video[resoStrLst_OpenPose[reso_ind]][str(frm_indx)]['spf'] for frm_indx in range(start_frame_indx, start_frame_indx+PLAYOUT_RATE, jumping_frm_interval)])
                tmp_time += current_tmp
            
                #print("tmp_time:", current_tmp)

            start_frame_indx += PLAYOUT_RATE
            processing_time_lst.append(tmp_time)
            acc_lst.append(1.0)                                 # because we profiled, we have the best config.
             
        #print('processing_time_lst: ', processing_time_lst)
        
        return popt_acc_fr, popt_acc_reso, acc_lst, processing_time_lst
        


    def calculate_frame_rate_accuracy(self, config_est_frm_arr, start_frame_indx, update_interval_time, frame_rate):
        # calculate the average average from a starting frame index
        # start from start_frame_indx
        # frame rate
        
        frame_reso_id = 0   # use maxim frame rate
        acc_accumu = 0.0
        gts = config_est_frm_arr[frame_reso_id][start_frame_indx]
        
        start_frame_indx_tmp = start_frame_indx
        
        
        second_in_interval  = update_interval_time*PLAYOUT_RATE//PLAYOUT_RATE
        inteval_frame_in_one_sec = frame_rate       # acutally jumping frame rate in the programming
        
        second_start = 0
        acc_accumulated_cnt = 0
        while(second_start < second_in_interval):
            
            start_frame_indx += PLAYOUT_RATE
            
            cnt_frame = 0
            while(start_frame_indx_tmp < start_frame_indx + PLAYOUT_RATE):
            
                #print("start_frame_indx_tmp :", start_frame_indx_tmp, frame_rate)
                
                dts = config_est_frm_arr[frame_reso_id][start_frame_indx_tmp]
                
                acc = computeOKS_1to1(gts, dts)
                
                acc_accumu += acc
                
                acc_accumulated_cnt += 1
                start_frame_indx_tmp += 1
                
                cnt_frame += 1
                if cnt_frame >= inteval_frame_in_one_sec:  # arrived to one interval, need to change ground truth
                    gts = config_est_frm_arr[frame_reso_id][start_frame_indx_tmp]

            
            second_start += 1
            
        
        aver_acc = acc_accumu/acc_accumulated_cnt
        
        #print("calculate_frame_rate_accuracy avera_acc: ", aver_acc)
        
        return aver_acc


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
        
    
    def get_configuration_total_multiarm(self):
        # 25 configurations --> 25 multi-arms
        
        frm_lst = range(PLAYOUT_RATE, 0, -1)    # (1, PLAYOUT_RATE+1)
        
        resolutions_lst = range(0, len(resoStrLst_OpenPose))  #  [int(x.split('x')[0]) for x in resoStrLst_OpenPose]
        
        configs_tuples = list(itertools.product(frm_lst, resolutions_lst))
      
        # printing unique_combination list
        print("get_configuration_total_multiarm configs_tuples:", frm_lst, resolutions_lst, configs_tuples)
        
        return configs_tuples
    
    
    def get_instance_reward_from_config(self, popt_acc_fr, popt_acc_reso, one_config_tuple):
        # input: accuracy fitting function;   frame rate and reso index config
        # ouput: instance acc
        
        frm_rate = one_config_tuple[0]
        reso = int(resoStrLst_OpenPose[one_config_tuple[1]].split('x')[0])
        
        acc_fr = expo_func(frm_rate, *popt_acc_fr)
        acc_reso = expo_func(reso, *popt_acc_reso)
        
        instance_acc = acc_fr * acc_reso

        return instance_acc
    
    
    """
    def configuraiton__online_epoch_greedy(self, config_est_frm_arr, acc_frame_arr, spf_frame_arr, update_interval_time, processing_interval_time, output_result_dir):
        # online RL algorithm epoch-greedy, random selecting without considering minimum accuracy
        # iteration interval is 1 second unit to decide an arm (configuration) and 
        # step 1: get the interval accuracy fitting function for reward use in the later interview
        
        configs_tuples = self.get_configuration_total_multiarm()
        
        
        FRAME_TOTAL_LEN = len(resoStrLst_OpenPose[0])
        
        
        
        each_iteration_accu_rewards = []
        accumulated_reward = np.zeros(self.Total_CONFIG_NUM, dtype=np.float64)   # defaultdict(float)
        
        accumulated_reward_dict = defaultdict(list)
        accumulated_estimated_accuracy = []
        
        accumulated_estimated_time = []
        
        start_frame_indx = 0
        
        TOTAL_SECONDS = FRAME_TOTAL_LEN//PLAYOUT_RATE
        
        print("TOTAL_SECONDS: ", TOTAL_SECONDS)
        
        start_iteration_second = 1
                
        popt_acc_fr = None
        popt_acc_reso = None
        
        while (start_iteration_second <= TOTAL_SECONDS and start_frame_indx <= FRAME_TOTAL_LEN):   # each second as an iteration
            
            # use interval updating time to get fitting function
            # skip this interval
            
            if (start_iteration_second % (processing_interval_time+1)) == 1:   # updating fitting function interval time
                
                popt_acc_fr, popt_acc_reso, acc_lst, processing_time_lst = self.get_accuracy_fitting_function_interval_time(config_est_frm_arr, acc_frame_arr, spf_frame_arr, update_interval_time, start_frame_indx)
            
                start_frame_indx += update_interval_time * PLAYOUT_RATE
                
                # need to update instance accuracy and processing time
                
                # instance reward 
                for i in range(len(acc_lst)):
                    instance_acc = acc_lst[i]
                    instance_processing_time = processing_time_lst[i]
                    
                    instance_re = self.instance_reward(instance_acc, instance_processing_time)
                    
                    accumulated_reward[0] += instance_re                    # r: reward in first trail  the most expensive configuration
                    
                    accumulated_estimated_accuracy.append(1.0)              # expensive 
                    accumulated_estimated_time.append(instance_processing_time)
                    
                    if len(accumulated_reward_dict[0]) == 0:
                        accumulated_reward_dict[0].append(instance_re)

                    else:
                        accumulated_reward_dict[0].append(accumulated_reward_dict[0][-1] + instance_re)
                    
                    if len(each_iteration_accu_rewards) == 0:
                        each_iteration_accu_rewards.append(instance_re)
                    else:
                        each_iteration_accu_rewards.append(each_iteration_accu_rewards[-1] + instance_re)
                    #print("tttttttttttttttinstance_accinstance_accinstance_acc: ", start_iteration_second, instance_acc)
            
                    
            else:                                                           # each interval is one second
                epoch = min(1,(self.c*self.Total_CONFIG_NUM)/(self.d*self.d*(start_iteration_second+1)))
                val =   rand.random()
                print("val epoch: ", start_iteration_second, val, epoch, val < epoch)
                if val < epoch:
                    one_config_tuple_indx = rand.randrange(len(configs_tuples))
                    
                    one_config_tuple = configs_tuples[one_config_tuple_indx]
                    frm_rate = one_config_tuple[0]
                    reso_id = one_config_tuple[1]
                    
                    instance_acc = self.get_instance_reward_from_config(popt_acc_fr, popt_acc_reso, one_config_tuple)
                    instance_processing_time = np.sum(spf_frame_arr[reso_id][start_frame_indx:(start_frame_indx+PLAYOUT_RATE):PLAYOUT_RATE//frm_rate])

                    instance_re = self.instance_reward(instance_acc, instance_processing_time)
                    accumulated_reward[one_config_tuple_indx] = accumulated_reward[one_config_tuple_indx] + instance_re # r: reward for ind machine in trail t
                    
                     
                    if len(accumulated_reward_dict[one_config_tuple_indx]) == 0:
                        accumulated_reward_dict[one_config_tuple_indx].append(instance_re)

                    else:
                        accumulated_reward_dict[one_config_tuple_indx].append(accumulated_reward_dict[one_config_tuple_indx][-1] + instance_re)
                    #print("instance_accinstance_accinstance_acc: ", one_config_tuple_indx, start_iteration_second, instance_acc)
                    
                    
                    if len(each_iteration_accu_rewards) == 0:
                        each_iteration_accu_rewards.append(instance_re)
                    else:
                        each_iteration_accu_rewards.append(each_iteration_accu_rewards[-1] + instance_re)
                        
                    # estiamted accuracy
                    accumulated_estimated_accuracy.append(instance_acc)              # expensive 
                    accumulated_estimated_time.append(instance_processing_time)
                                        
                
                else:
                    one_config_tuple_indx = np.argmax(accumulated_reward)  # np.where(v == maximum)
                    one_config_tuple = configs_tuples[one_config_tuple_indx]
                    instance_acc = self.get_instance_reward_from_config(popt_acc_fr, popt_acc_reso, one_config_tuple)
                    
                    frm_rate = one_config_tuple[0]
                    reso_id = one_config_tuple[1]
                    instance_processing_time = np.sum(spf_frame_arr[reso_id][start_frame_indx:(start_frame_indx+PLAYOUT_RATE):PLAYOUT_RATE//frm_rate])

                    instance_re = self.instance_reward(instance_acc, instance_processing_time)
                    
                    accumulated_reward[one_config_tuple_indx] = accumulated_reward[one_config_tuple_indx] + instance_re # r: reward for ind machine in trail t
                    
                    if len(accumulated_reward_dict[one_config_tuple_indx]) == 0:
                        accumulated_reward_dict[one_config_tuple_indx].append(instance_re)

                    else:
                        accumulated_reward_dict[one_config_tuple_indx].append(accumulated_reward_dict[one_config_tuple_indx][-1] + instance_re)
                    
                    
                    if len(each_iteration_accu_rewards) == 0:
                        each_iteration_accu_rewards.append(instance_re)
                    else:
                        each_iteration_accu_rewards.append(each_iteration_accu_rewards[-1] + instance_re)
                        
                        
                    #print("instance_accinstance_accinstance_acc: ", one_config_tuple_indx, start_iteration_second, instance_acc)
                    # estiamted accuracy
                    accumulated_estimated_accuracy.append(instance_acc)              # expensive 
                    accumulated_estimated_time.append(instance_processing_time)
                    
                start_frame_indx += PLAYOUT_RATE

            # then for the rest of processing_interval_time for accumulating rewards
            
            #print("accumulated_reward: ", accumulated_reward)
            
            start_iteration_second += 1
            
            
            #if iteration_t >=200:   # test only
            #    break          
    
        #print("accumulated_reward: ", accumulated_reward, iteration_t)
        
        
        output_result_dir += "reinforcement_learning_results/"
        
        if not os.path.exists(output_result_dir):
                os.mkdir(output_result_dir)
        
        print("accumulated_estimated_accuracy: ", accumulated_estimated_accuracy, len(accumulated_estimated_accuracy))
        # plot accuracy
        self.plot_result_online_rl_one_line_accuracy(accumulated_estimated_accuracy, output_result_dir)
        
        # plot processing time
        self.plot_result_online_rl_one_line_sps(accumulated_estimated_accuracy, output_result_dir)
        
        # plot rewards for each config
        self.plot_result_online_rl_rewards_configurations(accumulated_reward_dict, configs_tuples, output_result_dir)
        
        # plot total rewards accumulated
        self.plot_result_online_rl_one_line_accumulated_reward(each_iteration_accu_rewards, output_result_dir)
        return accumulated_reward_dict,  accumulated_estimated_accuracy, accumulated_estimated_time
    """
    
    
    def get_configuration_indx_above_accuracy(self, popt_acc_fr, popt_acc_reso, configs_tuples, min_acc_thres):
        # popt_acc_fr, popt_acc_reso
        
        available_config_tuples_above_minAcc = []
        
        
        all_possible_config_tuples_acc = []
        for one_config_tuple in configs_tuples:
            frm_rate = one_config_tuple[0]
            reso = int(resoStrLst_OpenPose[one_config_tuple[1]].split('x')[0])
            
            acc_fr = expo_func(frm_rate, *popt_acc_fr)
            acc_reso = expo_func(reso, *popt_acc_reso)
            
            #instance_acc = acc_fr * acc_reso
            
            if acc_fr >= min_acc_thres or acc_reso >= min_acc_thres:
                available_config_tuples_above_minAcc.append(one_config_tuple)
                
            all_possible_config_tuples_acc.append((acc_fr, one_config_tuple))
            print("available_config_tuples_above_minAcc: ", acc_fr, available_config_tuples_above_minAcc)
            
        if len(available_config_tuples_above_minAcc) == 0:   # if empty, sort them and select the largest configuration
            all_possible_config_tuples_acc = sorted(all_possible_config_tuples_acc, key=lambda ele: -ele[0])
            available_config_tuples_above_minAcc = [all_possible_config_tuples_acc[0][1]]
            
        return available_config_tuples_above_minAcc
    

    def configuraiton__online_epoch_greedy_min_accury_threshold(self, data_obj, dict_detection_reso_video, update_interval_time, processing_interval_time, min_acc_thres, output_result_dir):
        # online RL algorithm epoch-greedy, random selecting  considering minimum accuracy threshold
        configs_tuples = self.get_configuration_total_multiarm()
        
        config_est_frm_arr = []
        
        FRAME_TOTAL_LEN = len(dict_detection_reso_video[resoStrLst_OpenPose[0]])   # select one resolution's video anallytics result to get frame length
        
        #print("FRAME_TOTAL_LEN: ", FRAME_TOTAL_LEN)
        
        each_iteration_accu_rewards = []
        accumulated_reward = np.zeros(self.Total_CONFIG_NUM, dtype=np.float64)   # defaultdict(float)
        
        accumulated_reward_dict = defaultdict(list)
        accumulated_estimated_accuracy = []
        
        accumulated_estimated_time = []
        
        start_frame_indx = 1    # start from 1 here for traffic tracking
        
        TOTAL_SECONDS = FRAME_TOTAL_LEN//PLAYOUT_RATE
        
        print("TOTAL_SECONDS: ", TOTAL_SECONDS)
        
        start_iteration_second = 1
                
        popt_acc_fr = None
        popt_acc_reso = None
        
        available_config_tuples_above_minAcc = configs_tuples
        while (start_iteration_second <= TOTAL_SECONDS and start_frame_indx <= FRAME_TOTAL_LEN):   # each second as an iteration
            
            # use interval updating time to get fitting function
            # skip this interval
            
            if (start_iteration_second % (processing_interval_time+1)) == 1:   # it's time for updating fitting function, interval time
                
                popt_acc_fr, popt_acc_reso, acc_lst, processing_time_lst = self.get_accuracy_fitting_function_interval_time(data_obj, dict_detection_reso_video, update_interval_time, start_frame_indx)
            
                available_config_tuples_above_minAcc = self.get_configuration_indx_above_accuracy(popt_acc_fr, popt_acc_reso, configs_tuples, min_acc_thres)
        
                start_frame_indx += update_interval_time * PLAYOUT_RATE
                
                # need to update instance accuracy and processing time
                
                # instance reward 
                """
                for i in range(len(acc_lst)):
                    instance_acc = acc_lst[i]
                    instance_processing_time = processing_time_lst[i]
                    
                    instance_re = self.instance_reward(instance_acc, instance_processing_time)
                    
                    accumulated_reward[0] += instance_re                    # r: reward in first trail  the most expensive configuration
                    
                    accumulated_estimated_accuracy.append(1.0)              # expensive 
                    accumulated_estimated_time.append(instance_processing_time)
                    
                    if len(accumulated_reward_dict[0]) == 0:
                        accumulated_reward_dict[0].append(instance_re)

                    else:
                        accumulated_reward_dict[0].append(accumulated_reward_dict[0][-1] + instance_re)
                    
                    if len(each_iteration_accu_rewards) == 0:
                        each_iteration_accu_rewards.append(instance_re)
                    else:
                        each_iteration_accu_rewards.append(each_iteration_accu_rewards[-1] + instance_re)
                    #print("tttttttttttttttinstance_accinstance_accinstance_acc: ", start_iteration_second, instance_acc)
                """
                    
            else:                                                           # each interval is one second
                epoch = min(1,(self.c*self.Total_CONFIG_NUM)/(self.d*self.d*(start_iteration_second+1)))
                val =   rand.random()
                print("val epoch: ", start_iteration_second, val, epoch, val < epoch)
                if val < epoch:
                    #if len(available_config_tuples_above_minAcc) == 0:
                    #    available_config_tuples_above_minAcc = configs_tuples
                    one_config_tuple_indx = rand.randrange(len(available_config_tuples_above_minAcc))
                    
                    one_config_tuple = available_config_tuples_above_minAcc[one_config_tuple_indx]
                    frm_rate = one_config_tuple[0]
                    reso_id = one_config_tuple[1]
                    
                    instance_acc = self.get_instance_reward_from_config(popt_acc_fr, popt_acc_reso, one_config_tuple)
                    #instance_processing_time = np.sum(spf_frame_arr[reso_id][start_frame_indx:(start_frame_indx+PLAYOUT_RATE):PLAYOUT_RATE//frm_rate])
                    instance_processing_time = sum([dict_detection_reso_video[resoStrLst_OpenPose[reso_id]][str(frm_indx)]['spf'] for frm_indx in range(start_frame_indx, start_frame_indx+PLAYOUT_RATE, PLAYOUT_RATE//frm_rate)])
                    instance_re = self.instance_reward(instance_acc, instance_processing_time)
                    accumulated_reward[one_config_tuple_indx] = accumulated_reward[one_config_tuple_indx] + instance_re # r: reward for ind machine in trail t
                    
                     
                    if len(accumulated_reward_dict[one_config_tuple_indx]) == 0:
                        accumulated_reward_dict[one_config_tuple_indx].append(instance_re)

                    else:
                        accumulated_reward_dict[one_config_tuple_indx].append(accumulated_reward_dict[one_config_tuple_indx][-1] + instance_re)
                    #print("instance_accinstance_accinstance_acc: ", one_config_tuple_indx, start_iteration_second, instance_acc)
                    
                    
                    if len(each_iteration_accu_rewards) == 0:
                        each_iteration_accu_rewards.append(instance_re)
                    else:
                        each_iteration_accu_rewards.append(each_iteration_accu_rewards[-1] + instance_re)
                        
                    # estiamted accuracy
                    accumulated_estimated_accuracy.append(instance_acc)              # expensive 
                    accumulated_estimated_time.append(instance_processing_time)
                                        
                
                else:
                    one_config_tuple_indx = np.argmax(accumulated_reward)           #   np.where(v == maximum)
                    one_config_tuple = configs_tuples[one_config_tuple_indx]        #   available_config_tuples_above_minAcc[one_config_tuple_indx]
                    instance_acc = self.get_instance_reward_from_config(popt_acc_fr, popt_acc_reso, one_config_tuple)
                    
                    frm_rate = one_config_tuple[0]
                    reso_id = one_config_tuple[1]
                    #instance_processing_time = np.sum(spf_frame_arr[reso_id][start_frame_indx:(start_frame_indx+PLAYOUT_RATE):PLAYOUT_RATE//frm_rate])

                    instance_processing_time = sum([dict_detection_reso_video[resoStrLst_OpenPose[reso_id]][str(frm_indx)]['spf'] for frm_indx in range(start_frame_indx, start_frame_indx+PLAYOUT_RATE, PLAYOUT_RATE//frm_rate)])
                    instance_re = self.instance_reward(instance_acc, instance_processing_time)
                    accumulated_reward[one_config_tuple_indx] = accumulated_reward[one_config_tuple_indx] + instance_re # r: reward for ind machine in trail t
                    
                    if len(accumulated_reward_dict[one_config_tuple_indx]) == 0:
                        accumulated_reward_dict[one_config_tuple_indx].append(instance_re)

                    else:
                        accumulated_reward_dict[one_config_tuple_indx].append(accumulated_reward_dict[one_config_tuple_indx][-1] + instance_re)
                    
                    
                    if len(each_iteration_accu_rewards) == 0:
                        each_iteration_accu_rewards.append(instance_re)
                    else:
                        each_iteration_accu_rewards.append(each_iteration_accu_rewards[-1] + instance_re)
                        
                        
                    #print("instance_accinstance_accinstance_acc: ", one_config_tuple_indx, start_iteration_second, instance_acc)
                    # estiamted accuracy
                    accumulated_estimated_accuracy.append(instance_acc)              # expensive 
                    accumulated_estimated_time.append(instance_processing_time)
                    
                start_frame_indx += PLAYOUT_RATE

            # then for the rest of processing_interval_time for accumulating rewards
            
            #print("accumulated_reward: ", accumulated_reward)
            
            start_iteration_second += 1
            
            
            #if iteration_t >=200:   # test only
            #    break          
    
        #print("accumulated_reward: ", accumulated_reward, iteration_t)
        
        
        output_result_dir += "reinforcement_learning_results/"
        
        if not os.path.exists(output_result_dir):
                os.mkdir(output_result_dir)
        
        # write into file
        write_lst_to_csv(["Accuracy"], accumulated_estimated_accuracy, output_result_dir + "accumulated_estimated_accuracy.csv")
        write_lst_to_csv(["Processing time"], accumulated_estimated_time, output_result_dir + "accumulated_estimated_time.csv")
 
        print("accumulated_estimated_accuracy: ", accumulated_estimated_accuracy, len(accumulated_estimated_accuracy))
        # plot accuracy
        self.plot_result_online_rl_one_line_accuracy(accumulated_estimated_accuracy, output_result_dir)
        
        # plot processing time
        self.plot_result_online_rl_one_line_sps(accumulated_estimated_time, output_result_dir)
        

        # plot rewards for each config
        self.plot_result_online_rl_rewards_configurations(accumulated_reward_dict, configs_tuples, output_result_dir)
        
        # plot total rewards accumulated
        self.plot_result_online_rl_one_line_accumulated_reward(each_iteration_accu_rewards, output_result_dir)
        return accumulated_reward_dict,  accumulated_estimated_accuracy, accumulated_estimated_time
    
        
    
    
    
    def plot_result_online_rl_one_line_accumulated_reward(self, each_iteration_accu_rewards, output_result_dir):
        # not considering the configurations.
        
        x_lst1 = range(0, len(each_iteration_accu_rewards))
        y_lst1 = each_iteration_accu_rewards
        
        xlabel = "Iteration"
        ylabel = "Reward"
        title_name = ""
        
        fig = plotScatterLineOneFig(x_lst1, y_lst1, xlabel, ylabel, title_name)
        
        file_path_name = output_result_dir + "Learning_reward_accumulated_with_iteration.pdf"
        fig.savefig(file_path_name)
        
        
        
    def plot_result_online_rl_one_line_accuracy(self,accumulated_estimated_accuracy, output_result_dir):
        
        
        x_lst1 = range(0, len(accumulated_estimated_accuracy))
        y_lst1 = accumulated_estimated_accuracy
        
        xlabel = "Iteration"
        ylabel = "Accuracy"
        title_name = ""
        
        fig = plotScatterLineOneFig(x_lst1, y_lst1, xlabel, ylabel, title_name)
        
        file_path_name = output_result_dir + "Learning_accuray_with_iteration.pdf"
        fig.savefig(file_path_name)
        return 



    def plot_result_online_rl_one_line_sps(self,accumulated_estimated_time, output_result_dir):
       # processing time for vide oprocessing with iteration
       # sps: second per second
        x_lst1 = range(0, len(accumulated_estimated_time))
        y_lst1 = accumulated_estimated_time
        
        xlabel = "Iteration"
        ylabel = "Processing time (s)"
        title_name = ""
        
        fig = plotScatterLineOneFig(x_lst1, y_lst1, xlabel, ylabel, title_name)
        
        file_path_name = output_result_dir + "Learning_sps_with_iteration.pdf"
        fig.savefig(file_path_name)



    def plot_result_online_rl_rewards_configurations(self,accumulated_reward_dict, configs_tuples, output_result_dir):
        # plot part or all the rewards 
        
        yLsts = [reward_lst for k, reward_lst in accumulated_reward_dict.items()]
        
        legend_labels = [str(configs_tuples[k][0]) + '-' + resoStrLst_OpenPose[configs_tuples[k][1]].split('x')[0] + 'P'  for k, reward_lst in accumulated_reward_dict.items()]
        #legend_labels = ['P'  for k, reward_lst in accumulated_reward_dict.items()]
        
        print("legend_labels :", len(accumulated_reward_dict), legend_labels)
        
        xlabel = "Iteration"
        ylabel = "Rewards"
        title_name = ""
        
        
        fig = plotMultipleLinesOneFigure(yLsts, legend_labels, xlabel, ylabel, title_name)
        
        file_path_name = output_result_dir + "Learning_rewards_configuration_with_iteration.pdf"
        fig.savefig(file_path_name)
                
        

    def online_learning_RL(self):
        
        
        data_obj = DataGenerate(data_dir)
        
        #1. mixed videos ; Configuration  Prediction  Performance  on  Whole  VideoDataset
        #model_obj.test_on_multiple_mixed_video(data_obj.video_dir_lst, min_acc_threshold)
    
        
        video_dir_lst_tested = data_obj.video_dir_lst[2:3]    # [5:6]   # [5, 15]
        # prediction on one video
        print("video_dir_lst_tested: ", data_dir, data_obj.video_dir_lst)

        update_interval_time = 1       # 1 second
        processing_interval_time = 3   # 9 second
        
        min_acc_thres = 0.9   # 0.90        # minimum accuracy threshold 0
        
        #all_arr_estimated_speed_2_jump_number = blist()  # all video
        for i, video_dir in enumerate(video_dir_lst_tested): # [3:4]):    # [2:3]:   #[1:2]:  # [1:2]:  #[0:1]:        #[1:2]:
            
            data_pickle_dir = video_dir + '/'

            #data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result_each_frm/'
            
            print("data_pickle_dir : ", data_pickle_dir)
            dict_detection_reso_video = read_json_dir(data_pickle_dir)
            print("dict_detection_reso_video : ", dict_detection_reso_video.keys(), dict_detection_reso_video['1120x832']['1'], dict_detection_reso_video['1120x832']['10'])

            #config_est_frm_arr, acc_frame_arr, spf_frame_arr = self.get_data_numpy(data_pose_keypoint_dir, data_pickle_dir, intervalFlag, all_config_flag)
            
            
            self.configuraiton__online_epoch_greedy_min_accury_threshold(data_obj, dict_detection_reso_video, update_interval_time, processing_interval_time, min_acc_thres, data_pickle_dir)
        
            
            xx 
 
            
if __name__== "__main__": 
    
    MultiArmedConfig_obj = MultiArmedConfig()
    MultiArmedConfig_obj.online_learning_RL()
    
            
        