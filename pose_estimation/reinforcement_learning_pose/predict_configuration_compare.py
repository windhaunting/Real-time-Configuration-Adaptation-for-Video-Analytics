#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 12:23:31 2021

@author: fubao
"""

"""
explore the configuration prediction accuracy based on the contextual bandit epoch policy algorithm.
The ground truth configuration compared are the configuration with the highest accuracy above min accuracy threshold with the least processing time.
"""
import numpy as np


def configs_online_epoch_greedy_min_accuracy_threshold_prediction(config_est_frm_arr, acc_frame_arr, spf_frame_arr, update_interval_time, processing_interval_time, min_acc_thres, output_result_dir):
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