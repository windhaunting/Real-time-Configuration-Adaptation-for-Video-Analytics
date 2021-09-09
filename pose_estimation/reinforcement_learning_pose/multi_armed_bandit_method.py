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
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
 
from fitting_accuracy_func import fit_function
from fitting_accuracy_func import expo_func
from common_plot import plotTwoLinesOneFigure
from common_plot import plotScatterLineOneFig
from common_plot import plotMultipleLinesOneFigure

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


# here frame rate is the jumping frame interval

class MultiArmedConfig(object):
    
    def __init__(self):
        
        self.alpha = 0.8 # accuracy to reward ; the weight for accuracy coefficient
        self.beta = 0.02
        self.c = 0.002*200
        self.d = 0.5
        
        self.Total_CONFIG_NUM =  25*5             # K which is the K arms
        
    
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


    def fitting_accuracy(self, fr_acc_data_point, reso_acc_data_point):
        # with fitting line points to get the fitting function's coefficients
                
        popt_acc_fr = self.fit_accuracy_frame_rate(fr_acc_data_point)
        popt_acc_reso = self.fit_accuracy_resolution(reso_acc_data_point)
        
        return popt_acc_fr, popt_acc_reso
    
    """
    def get_instance_acc_with_fitting_function(self, popt_acc_fr, popt_acc_reso, curr_frm_rate, curr_reso):
        
        # get the instance accuracy with an already fitted function.  the input is a configuration  
        
        acc_fr = expo_func(PLAYOUT_RATE*1.0/curr_frm_rate, *popt_acc_fr)
        
        acc_reso = expo_func(curr_reso, *popt_acc_reso)
        
        instance_acc = acc_fr * acc_reso
        return instance_acc
    """
        
    def instance_reward(self,instance_acc, instance_processing_time):
        # get the instance reward from the accuracy and processing time
        # r = alpha*instance_acc + (1-alpha)*instance_processing_time
        
        inst_reward = self.alpha*instance_acc - self.beta*instance_processing_time
        
        return inst_reward
    

    def get_accuracy_fitting_function_interval_time(self, config_est_frm_arr, acc_frame_arr, spf_frame_arr, update_interval_time, start_frame_indx):
        # get the accuracy in the interval time similar to profiling method, but it is not profiling
        # so we have to run the expensive configuration.
        # get the average accuracy and the configuration;  for three configurations that are enough
        
        # and the processing time in this update_interval_time
        # note: frame rate here is actually the jumping frame interval in the programming
        #print("config_est_frm_arr: ", config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        # select at least three points
        
        frm_lst = [1, 10, 25]  # range(1, PLAYOUT_RATE+1)
        
        resolutions_lst = [0, 2, 4]  # range(0, len(resoStrLst_OpenPose))   #  [int(x.split('x')[0]) for x in resoStrLst_OpenPose]
        
        #print('frm list: ', frm_lst, resolutions_lst)
        
        POINTS_NUM_SELECTED = 3
        
        selected_frm_rates = frm_lst            #  rand.sample(frm_lst, POINTS_NUM_SELECTED)
        
        selected_reso_indices = resolutions_lst  #  rand.sample(resolutions_lst, POINTS_NUM_SELECTED)
        
        #print('selected_frm_rates: ', selected_frm_rates, selected_reso_indices)
        # read 
        
        # get different accuracy with frame rate points  for frame rate point  to acc y axis 
        # PLAYOUT_RATE*1.0/frame_rate
        fr_acc_data_point = [[PLAYOUT_RATE*1.0/frame_rate, self.calculate_frame_rate_accuracy(config_est_frm_arr, start_frame_indx, update_interval_time, frame_rate)] for frame_rate in selected_frm_rates] 
        
        
        # get different accuracy with resolution points
        
        dict_map_id_reso = {i: int(resoStrLst_OpenPose[i].split('x')[0]) for i in range(0, len(resoStrLst_OpenPose))}
        reso_acc_data_point =  [[dict_map_id_reso[reso_id], acc_frame_arr[reso_id][start_frame_indx]] for reso_id in selected_reso_indices]
        
        #print('reso_acc_data_point: ', fr_acc_data_point, reso_acc_data_point) 
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
                jumping_frm_interval = 1                        #  because we run the frame rate=25, each frame is got, we so don't need to use each frame rate
                #print(":", spf_frame_arr[reso_ind].shape, start_frame_indx, jumping_frm_interval)
                tmp_time += np.sum(spf_frame_arr[reso_ind][start_frame_indx:(start_frame_indx+PLAYOUT_RATE):jumping_frm_interval])  # here frame rate is the jumping interval
                
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
        
        
        second_in_interval  = update_interval_time  # *PLAYOUT_RATE//PLAYOUT_RATE
        inteval_frame_in_one_sec = PLAYOUT_RATE//frame_rate       #  jumping frame interval for the programming
        
        second_start = 0
        acc_accumulated_cnt = 0
        while(second_start < second_in_interval):
            
            start_frame_indx += PLAYOUT_RATE
            
            cnt_frame = 0
            while(start_frame_indx_tmp < start_frame_indx + PLAYOUT_RATE and start_frame_indx_tmp < config_est_frm_arr.shape[1]-1):
            
                #print("start_frame_indx_tmp :", start_frame_indx_tmp, frame_rate)
                
                dts = config_est_frm_arr[frame_reso_id][start_frame_indx_tmp]
                
                acc = computeOKS_1to1(gts, dts)
                
                acc_accumu += acc
                
                acc_accumulated_cnt += 1
                start_frame_indx_tmp += 1
                
                cnt_frame += 1
                if cnt_frame >= inteval_frame_in_one_sec:  # reach to one interval, need to change ground truth
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
        
        acc_fr = expo_func(PLAYOUT_RATE*1.0/frm_rate, *popt_acc_fr)
        acc_reso = expo_func(reso, *popt_acc_reso)
        
        instance_acc = acc_fr * acc_reso

        return instance_acc
        
    
    def get_configuration_indx_above_accuracy(self, popt_acc_fr, popt_acc_reso, configs_tuples, min_acc_thres):
        # popt_acc_fr, popt_acc_reso
        
        available_config_tuples_above_minAcc = []
        
        
        while(True):
            for one_config_tuple in configs_tuples:
                frm_rate = one_config_tuple[0]
                reso = int(resoStrLst_OpenPose[one_config_tuple[1]].split('x')[0])
                
                acc_fr = expo_func(PLAYOUT_RATE*1.0/frm_rate, *popt_acc_fr)
                acc_reso = expo_func(reso, *popt_acc_reso)
                
                instance_acc = acc_fr * acc_reso
                print("get_configuration_indx_above_accuracy, acc_fr, acc_reso: ", frm_rate, reso, acc_fr, acc_reso)
                
                if acc_fr >= min_acc_thres or acc_reso >= min_acc_thres:
                    available_config_tuples_above_minAcc.append(one_config_tuple)
            
            if len(available_config_tuples_above_minAcc) != 0:
                break
            else:
                min_acc_thres -= 0.05
            
        return available_config_tuples_above_minAcc
    

    def configuraiton__online_epoch_greedy_min_accury_threshold(self, config_est_frm_arr, acc_frame_arr, spf_frame_arr, update_interval_time, processing_interval_time, min_acc_thres, output_result_dir):
        # online RL algorithm epoch-greedy, random selecting  considering minimum accuracy threshold
        configs_tuples = self.get_configuration_total_multiarm()
        
        FRAME_TOTAL_LEN = config_est_frm_arr.shape[1]
                
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
        
        available_config_tuples_above_minAcc = configs_tuples
        while (start_iteration_second <= TOTAL_SECONDS and start_frame_indx <= FRAME_TOTAL_LEN):   # each second as an iteration
            
            # use interval updating time to get fitting function
            # skip this interval
            
            if (start_iteration_second % (processing_interval_time+1)) == 1:   # updating fitting function interval time
                
                popt_acc_fr, popt_acc_reso, acc_lst, processing_time_lst = self.get_accuracy_fitting_function_interval_time(config_est_frm_arr, acc_frame_arr, spf_frame_arr, update_interval_time, start_frame_indx)
                
                if min_acc_thres == 0:
                    available_config_tuples_above_minAcc = configs_tuples
                else:
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
                    one_config_tuple_indx = rand.randrange(len(available_config_tuples_above_minAcc))
                    
                    one_config_tuple = available_config_tuples_above_minAcc[one_config_tuple_indx]
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
                    one_config_tuple = available_config_tuples_above_minAcc[one_config_tuple_indx]
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
                
        
        
    def configs_online_epoch_greedy_min_accuracy_threshold_prediction(self, config_est_frm_arr, acc_frame_arr, spf_frame_arr, update_interval_time, processing_interval_time, min_acc_thres, output_result_dir):
        # online RL algorithm epoch-greedy, random selecting  considering minimum accuracy threshold
        configs_tuples = self.get_configuration_total_multiarm()
        
        FRAME_TOTAL_LEN = config_est_frm_arr.shape[1]
        
        each_iteration_accu_rewards = []
        accumulated_reward = np.zeros(self.Total_CONFIG_NUM, dtype=np.float64)   # defaultdict(float)
        
        #accumulated_reward_dict = defaultdict(list)
        #accumulated_estimated_accuracy = []
        #accumulated_estimated_time = []

        start_frame_indx = 0
        
        TOTAL_SECONDS =  8  # FRAME_TOTAL_LEN//PLAYOUT_RATE
        print("TOTAL_SECONDS: ", TOTAL_SECONDS)
        start_iteration_second = 1
                
        popt_acc_fr = None
        popt_acc_reso = None
        
        available_config_tuples_above_minAcc = configs_tuples
        
        configs_frm_rate_selected = []   # online learning selected configs frame rate
        configs_resolutions_selected = []   # online learning selected configs
        
        configs_frm_rate_ground_truth = []
        configs_reso_ground_truth = []
        
        last_reso_indx = 0
        
        while (start_iteration_second <= TOTAL_SECONDS and start_frame_indx <= FRAME_TOTAL_LEN):   # each second as an iteration
            
            # use interval updating time to get fitting function
            # skip this interval
            
            if (start_iteration_second % (processing_interval_time+1)) == 1:   # updating fitting function interval time
                
                popt_acc_fr, popt_acc_reso, acc_lst, processing_time_lst = self.get_accuracy_fitting_function_interval_time(config_est_frm_arr, acc_frame_arr, spf_frame_arr, update_interval_time, start_frame_indx)
                
                if min_acc_thres == 0:
                    available_config_tuples_above_minAcc = configs_tuples
                else:
                    available_config_tuples_above_minAcc = self.get_configuration_indx_above_accuracy(popt_acc_fr, popt_acc_reso, configs_tuples, min_acc_thres)
                
                start_frame_indx += update_interval_time * PLAYOUT_RATE
                
                # need to update instance accuracy and processing time 
                # similar to profiling, hence the accuracy is 1.0
                # instance reward 
                
                configs_frm_rate_selected.append(25)  # jumping frame interval is 1
                configs_resolutions_selected.append(0)   # resolution is "1120x832"
                
                configs_frm_rate_ground_truth.append(25)
                configs_reso_ground_truth.append(0)
                
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
                    one_config_tuple_indx = rand.randrange(len(available_config_tuples_above_minAcc))
                    
                    one_config_tuple = available_config_tuples_above_minAcc[one_config_tuple_indx]
                    frm_rate = one_config_tuple[0]
                    reso_id = one_config_tuple[1]
                    
                    instance_acc = self.get_instance_reward_from_config(popt_acc_fr, popt_acc_reso, one_config_tuple)
                    instance_processing_time = np.sum(spf_frame_arr[reso_id][start_frame_indx:(start_frame_indx+PLAYOUT_RATE):PLAYOUT_RATE//frm_rate])

                    instance_re = self.instance_reward(instance_acc, instance_processing_time)
                    accumulated_reward[one_config_tuple_indx] = accumulated_reward[one_config_tuple_indx] + instance_re # r: reward for ind machine in trail t
                    
                    
                    if len(each_iteration_accu_rewards) == 0:
                        each_iteration_accu_rewards.append(instance_re)
                    else:
                        each_iteration_accu_rewards.append(each_iteration_accu_rewards[-1] + instance_re)
                                                           
                else:
                    one_config_tuple_indx = np.argmax(accumulated_reward)  # np.where(v == maximum)
                    one_config_tuple = available_config_tuples_above_minAcc[one_config_tuple_indx]
                    instance_acc = self.get_instance_reward_from_config(popt_acc_fr, popt_acc_reso, one_config_tuple)
                    
                    frm_rate = one_config_tuple[0]
                    reso_id = one_config_tuple[1]
                    instance_processing_time = np.sum(spf_frame_arr[reso_id][start_frame_indx:(start_frame_indx+PLAYOUT_RATE):PLAYOUT_RATE//frm_rate])

                    instance_re = self.instance_reward(instance_acc, instance_processing_time)
                    
                    accumulated_reward[one_config_tuple_indx] = accumulated_reward[one_config_tuple_indx] + instance_re # r: reward for ind machine in trail t
                    
                 
                    if len(each_iteration_accu_rewards) == 0:
                        each_iteration_accu_rewards.append(instance_re)
                    else:
                        each_iteration_accu_rewards.append(each_iteration_accu_rewards[-1] + instance_re)
                        
                    
                # get learned configuration online from multi-armed bandit algorithm
                print("configs_jumping_frm_interval_selected epoch: ", frm_rate, reso_id)
                configs_frm_rate_selected.append(frm_rate)      # frame rate
                configs_resolutions_selected.append(reso_id)    # reso id
                
                # get ground truth
                frm_rate_selected, jumping_frm_number, segment_average_acc = self.estimate_best_frame_rate_as_ground_truth(config_est_frm_arr, acc_frame_arr, last_reso_indx,  start_frame_indx, max_jump_number = 25,  min_acc_threshold = min_acc_thres)
                        
                
                last_reso_indx, average_acc_resolution = self.select_resolution(config_est_frm_arr, start_frame_indx, jumping_frm_number, segment_average_acc, min_acc_thres)
       
                configs_frm_rate_ground_truth.append(frm_rate_selected)
                configs_reso_ground_truth.append(last_reso_indx)
                
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
        
        print("configs_frm_rate_selected :", configs_frm_rate_selected, configs_resolutions_selected, len(configs_resolutions_selected))

        print("configs_ground truth:", configs_frm_rate_ground_truth, configs_reso_ground_truth, len(configs_reso_ground_truth))

        self.calculate_acccury(configs_frm_rate_selected, configs_frm_rate_ground_truth, configs_resolutions_selected, configs_reso_ground_truth)
        
    
    def estimate_best_frame_rate_as_ground_truth(self, config_est_frm_arr, acc_frame_arr, last_reso_indx,  start_frame_index, max_jump_number = 25,  min_acc_threshold = 0.95):
        #input: from a starting frame index, we use the current start frame to replace the several later frame
        # and calculate the accuracy, until we get an accuracy smaller than min_acc_threshold
        # ouput: the no. of the several frames jumped, and the replaced detection pose 
        # consider outlier and noisy data, we use max_jump_number here
        
        # get start frame's pose estimation result
        # print("estimate_jumping_number config_est_frm_arr_all: ", ans_reso_indx, config_est_frm_arr_all.shape, config_est_frm_arr_all[0].shape)
        
        frame_length = config_est_frm_arr.shape[1]  # [last_reso_indx].shape[0]
        
        curr_frame_index = start_frame_index + 1
        ref_pose = config_est_frm_arr[0][start_frame_index]     # reference pose ground truth
        #print("ref_pose shape: ", ref_pose.shape)
        
        cnt_frames = 1      # include the first frame detected by DNN itself, how many jumping frame number
        # the acc for start_frame_index
        #print("acc_frame_arrrrrrrr :", acc_frame_arr.shape, last_reso_indx, start_frame_index)
        
        curr_accumulated_acc = acc_frame_arr[last_reso_indx][start_frame_index]  # acc_frame_arr[start_frame_index]  # considered as 1.0 now
        average_acc = curr_accumulated_acc
        # print ("curr_accumulated_acc: ", curr_accumulated_acc)
        
        last_oks = 0
        next_frm_kp_arr = ref_pose
        
        while(curr_frame_index < frame_length and average_acc >= min_acc_threshold):
            # get the oks similarity acc
            curr_pose = config_est_frm_arr[last_reso_indx][curr_frame_index]
            oks = computeOKS_1to1(next_frm_kp_arr, curr_pose, sigmas = None)     # oks with reference pose
            
            #next_frm_kp_arr = self.get_estimated_keypoint_by_overall_speed(next_frm_kp_arr, arr_ema_absolute_speed)
            #next_frm_kp_arr = self.get_estimated_arm_leg_relative_speed(next_frm_kp_arr, arr_ema_relative_speed)
            
            curr_accumulated_acc += oks
            #print ("oks: ", oks)
            average_acc = curr_accumulated_acc/(cnt_frames+1)
            
            curr_frame_index += 1
            cnt_frames += 1
            last_oks = oks
        
        cnt_frames = min(cnt_frames, max_jump_number)
        frm_rate_selected = PLAYOUT_RATE // cnt_frames
        
        #seg_detected_pos_frm_list = self.get_detected_pos_frm_arr(ref_pose, cnt_frames)
        if cnt_frames == 1:
            segment_average_acc = curr_accumulated_acc
        else:
            segment_average_acc = (curr_accumulated_acc - last_oks)/(cnt_frames-1)
        
        print ("frm_rate_selected cnt_frames: ", frm_rate_selected, cnt_frames)
        return frm_rate_selected, cnt_frames, segment_average_acc


    def select_resolution(self, config_est_frm_arr, start_frame_index, jumping_frm_number, current_aver_acc, min_acc_threshold):
        #after getting the jumping number, pick the resolution above the min_acc and minimize delay
        # mininum delay
        # current_aver_acc is the acc achieved with frame jumping number selection
        
        # 5 resolutions to select
        # print ('select_resolution acc_frame_arr', config_est_frm_arr_all.shape, acc_frame_arr_all.shape, spf_frame_arr_all.shape)

        # count_jumping_frames = 1
        
        ans_reso_indx = 4   # ground truth
        
        ref_pose = config_est_frm_arr[0][start_frame_index]     # reference pose

        last_average_acc = current_aver_acc
        for reso_indx in range(4, -1, -1):    # 1, 2, 3, 4
            # check the accuracy for this range    # if we can not find, use the last higher resolution
            curr_frame_index = start_frame_index
            curr_accumulated_acc = 0.0
            while(curr_frame_index < (start_frame_index + jumping_frm_number)):
                curr_pose = config_est_frm_arr[reso_indx][curr_frame_index]
                
                oks = computeOKS_1to1(ref_pose, curr_pose, sigmas = None)     # oks with reference pose
                curr_accumulated_acc += oks
                #print ("oks: ", oks)
            
                curr_frame_index += 1
                
            average_acc = curr_accumulated_acc/(jumping_frm_number)  # jumping_frm_number include the start_frame_index
            
            if average_acc >= min_acc_threshold:
                ans_reso_indx = reso_indx
                last_average_acc = average_acc
                break
            # print ("select_resolution average_acc: ", reso_indx, average_acc, min_acc_threshold)
            
            last_average_acc = average_acc
        
        return ans_reso_indx, last_average_acc


    def calculate_acccury(self, configs_frm_rate_selected, configs_frm_rate_ground_truth, configs_resolutions_selected, configs_reso_ground_truth):
        
        # how to get the accuracy,  coefficient determination, or acc
        r2_frm_rate = r2_score(configs_frm_rate_selected, configs_frm_rate_ground_truth)
        r2_reso = r2_score(configs_resolutions_selected, configs_reso_ground_truth)
        
        acc_frm_rate = accuracy_score(configs_frm_rate_selected, configs_frm_rate_ground_truth)
        acc_reso = accuracy_score(configs_resolutions_selected, configs_reso_ground_truth)
        print("r2_frm_rate, r2_reso: ", r2_frm_rate, r2_reso, acc_frm_rate, acc_reso)
        
        
    def online_learning_RL(self):
        
        #global dataDir3
        dataDir3_update = "../" + dataDir3 
        
        update_interval_time = 1       # 1 second
        processing_interval_time = 3   # 9 second
        
        min_acc_thres = 0.95   # 0.95 # 0.9         # minimum accuracy threshold
        
        #all_arr_estimated_speed_2_jump_number = blist()  # all video
        for i, video_dir in enumerate(video_dir_lst[0:1]): # [3:4]):    # [2:3]:   #[1:2]:  # [1:2]:  #[0:1]:        #[1:2]:
            data_pose_keypoint_dir = dataDir3_update + video_dir
            
            data_pickle_dir = dataDir3_update + video_dir + 'frames_pickle_result/'
            #data_pickle_dir = dataDir3_update + video_dir + 'frames_pickle_result_each_frm/'
            intervalFlag = 'frame'
            all_config_flag = False
            config_est_frm_arr, acc_frame_arr, spf_frame_arr = self.get_data_numpy(data_pose_keypoint_dir, data_pickle_dir, intervalFlag, all_config_flag)
            
            #self.configuraiton__online_epoch_greedy(config_est_frm_arr, acc_frame_arr, spf_frame_arr, update_interval_time, processing_interval_time, dataDir3_update + video_dir)
            
            #self.configuraiton__online_epoch_greedy_min_accury_threshold(config_est_frm_arr, acc_frame_arr, spf_frame_arr, update_interval_time, processing_interval_time, min_acc_thres, dataDir3_update + video_dir)
            
            self.configs_online_epoch_greedy_min_accuracy_threshold_prediction(config_est_frm_arr, acc_frame_arr, spf_frame_arr, update_interval_time, processing_interval_time, min_acc_thres, dataDir3_update + video_dir)
       
            xx 
 
            
if __name__== "__main__": 
    
    MultiArmedConfig_obj = MultiArmedConfig()
    MultiArmedConfig_obj.online_learning_RL()
    
            
        