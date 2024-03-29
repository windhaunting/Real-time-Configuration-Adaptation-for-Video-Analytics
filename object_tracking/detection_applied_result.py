#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 19:23:24 2021

@author: fubao
"""

# read the frames from the video and then applied the adaptive configuration result to the video analytics

import os
import time
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, f1_score, precision_recall_fscore_support
from sklearn.decomposition import PCA

from tracking_data_preprocess import read_json_dir
from tracking_data_preprocess import read_pickle_data
from tracking_data_preprocess import write_pickle_data
from tracking_data_preprocess import bb_intersection_over_union

from tracking_get_training_data import data_dir
from tracking_get_training_data import PLAYOUT_RATE
from tracking_get_training_data import reso_list
from tracking_get_training_data import max_jump_number
from tracking_get_training_data import min_acc_threshold
from tracking_get_training_data import interval_frm
from tracking_get_training_data import DataGenerate
from tracking_get_training_data import object_name

from prediction_jumpingNumber_resolution import ModelClassifier
from prediction_jumpingNumber_resolution import n_components



class VideoApply(object):
    def __init__(self):
        pass


    """
    def get_prediction_acc_delay(self, predicted_video_dir, min_acc_threshold):
        # after applying adaptation of configuration
        
        # get the predicted video's accuracy anpredicted_video_dird delay
        #  predicted_video_id as the testing data
        # predicted_video_dir:  such as output_021_dance
        # predicted_out_file is the prediction jumping number and delay
        
         # speaker_box_arr, acc_arr, spf_arr = get_data_numpy(input_dir + predicted_video_dir)
        dict_detection_reso_video = read_json_dir(data_dir + predicted_video_dir)
        
        output_pickle_dir = data_dir + predicted_video_dir + "jumping_number_result/" 

        model_dir = output_pickle_dir + "jumpingNumber_resolution_selection/intervalFrm-5_speedType-ema_minAcc-" + str(min_acc_threshold) + "/"    
        test_x_instances_file = model_dir + "data_instance_speed_jumpingNumber_resolution_objectSizeRatio_xy.pkl"

        X = read_pickle_data(test_x_instances_file)
        #print ("X shape:", X.shape)
        model_file = data_dir + predicted_video_dir + "data_instance_xy/minAcc_"  + str(min_acc_threshold) + "/" + "model_classifier.joblib_exclusive_" + str(predicted_video_dir[:-1]) + ".pkl"
        pca = PCA(n_components=min(n_components, X.shape[1])).fit(X)
        ModelClassifyObj = ModelClassifier()
        model = joblib.load(model_file)


        frm_no = len(dict_detection_reso_video[reso_list[0]])  #   min(len(imagePathLst), spf_arr.shape[1])
        
        
        ans_reso_indx = 0
        last_frm_indx = 1
        current_frm_indx = interval_frm + 1                     # 1  starting index = 0  left frame no = 1
        
        DataGenerateObj = DataGenerate()  
   
        print ("frm_no :", frm_no)

        # initially used resolution
        #last_frm_box = dict_detection_reso_video[reso_list[ans_reso_indx]][str(frm_lst_indx)][object_name]      # initialize
          
        #print("last_frm_box: ", last_frm_box)

        # use predicted result to apply to this new video and get delay and accuracy
        #delay_arr = []      
        #up_to_delay = max(0, spf_arr[ans_reso_indx][0] - 1.0/PLAYOUT_RATE)             # up to current delay, processing time - streaming time

        arr_ema_absolute_velocity = np.zeros((4, 2))

        acc_pred_arr = []
        delay_arr = []      
        up_to_delay = max(0, float(dict_detection_reso_video[reso_list[ans_reso_indx]][str(last_frm_indx)]['spf'])  - 1.0/PLAYOUT_RATE)         # up to current delay, processing time - streaming time
     
        while(current_frm_indx < frm_no):
            
            # calculate the relative feature
            
            frm_reso = reso_list[ans_reso_indx]
            arr_ema_absolute_velocity = DataGenerateObj.get_movement_feature(dict_detection_reso_video, current_frm_indx, last_frm_indx, frm_reso, arr_ema_absolute_velocity)
            
            feature_vect_mean = np.mean(arr_ema_absolute_velocity, axis = 0)
            feature_vect_var = np.var(arr_ema_absolute_velocity, axis = 0)
            #print("arr_ema_speed: ", feature_vect_mean, feature_vect_var, np.asarray([feature_vect_mean, feature_vect_var]))
            # get the jumping number y
            
            arr_movement_feature = np.hstack((arr_ema_absolute_velocity.flatten(), feature_vect_mean, feature_vect_var))
            #print("abso_velocity_feature_x: ", abso_velocity_feature_x.shape)
            
            feature_x_object_size = DataGenerateObj.get_object_size_x(dict_detection_reso_video, current_frm_indx, ans_reso_indx)      
            
            feature_x = np.hstack((arr_movement_feature, feature_x_object_size))
            
            #ans_jfr, ans_reso_indx, aver_acc = self.predict_next_configuration_jumping_frm_reso(dict_detection_reso_video, min_acc_threshold, curent_frm_indx, FRM_NO)
            # predict how many frame jumped from this starting point

            #print ("feature_x cur_absolute_speed: " , cur_absolute_speed)
        
            #print ("feature_x : ", feature_x.shape, feature_x)
            predicted_y = ModelClassifyObj.test_on_data_y_unknown(model, feature_x, pca)
            
            jumping_frm_number = int(predicted_y[0][0])
            ans_reso_indx = int(predicted_y[0][1])
            
            # get delay up to this segment
            #up_to_delay = max(0, spf_arr[ans_reso_indx][frm_curr_indx] - (1.0/PLAYOUT_RATE) * jumping_frm_number)
            #delay_arr.append(up_to_delay)
            
            #print ("jumping_frm_number: ", jumping_frm_number, ans_reso_indx)

         
            highest_reso = reso_list[0]
            curre_reso = reso_list[ans_reso_indx]
            acc_tmp, time_spf_tmp, calc_frm_num_tmp = self.get_online_video_analytics_accuracy_spf(data_dir, dict_detection_reso_video, current_frm_indx, jumping_frm_number, highest_reso, curre_reso)
            #print ("feature_x : ", feature_x.shape, feature_x, acc_tmp, jumping_frm_number, ans_reso_indx)
            
            acc_pred_arr.append(acc_tmp)
                
            up_to_delay = max(0, time_spf_tmp- (1.0/PLAYOUT_RATE) * jumping_frm_number)
            #up_to_delay = max(0, float(dict_detection_reso_video[reso_list[ans_reso_indx]][str(current_frm_indx)]['spf'])- (1.0/PLAYOUT_RATE) * jumping_frm_number)
            delay_arr.append(up_to_delay)
            
            last_frm_indx = current_frm_indx
            current_frm_indx += jumping_frm_number  #

            #EMA_absolute_prev = EMA_absolute_curr   # update EMA

        acc_pred_arr = np.asarray(acc_pred_arr)
        delay_arr = np.asarray(delay_arr)
        print ("get_prediction_acc_delay acc_pred_arr, delay_arr: ",  acc_pred_arr, delay_arr)
        
        
        detect_out_result_dir = output_pickle_dir + "video_applied_detection_result/"
        if not os.path.exists(detect_out_result_dir):

            os.mkdir(detect_out_result_dir)

        arr_acc_segment_file = detect_out_result_dir + "arr_acc_segment_.pkl"
        arr_delay_up_to_segment_file = detect_out_result_dir + "arr_delay_up_to_segment_.pkl"
        write_pickle_data(acc_pred_arr, arr_acc_segment_file)
        write_pickle_data(delay_arr, arr_delay_up_to_segment_file)
    """
        
        
    def get_one_video_spf_latency(self, data_dir, predicted_video_dir, min_acc, imagePathLst):
        
        """
        # speaker_box_arr, acc_arr, spf_arr = get_data_numpy(input_dir + predicted_video_dir)
        dict_detection_reso_video = read_json_dir(data_dir + predicted_video_dir)
        
        output_pickle_dir = data_dir + predicted_video_dir + "data_instance_xy/minAcc_" + str(min_acc) + "/"
        
        model_file = output_pickle_dir + "model_classifier.joblib" + "_exclusive_" + str(predicted_video_dir[:-1])  + ".pkl"       # with other videos
                        
        
        print ("model_file:", model_file)
        
        
        test_x_instances_file = output_pickle_dir + "all_data_instance_speed_JumpingNumber_resolution_objectSizeRatio_xy.pkl"
        X = read_pickle_data(test_x_instances_file)
        #print ("X shape:", X.shape)
        pca = PCA(n_components=min(n_components, X.shape[1])).fit(X)
        """
        
        video_frame_dir =  predicted_video_dir   # +  '*_frames'  # 'car_traffic_' + str(video_id) + '_frames'
        
        for root, subdirs, files in os.walk(video_frame_dir):
            flag = False
            for frame_folder_name in subdirs:
                if '_frames' in frame_folder_name:
                    video_frame_dir = video_frame_dir + frame_folder_name
                    flag = True
                    break
            if flag:
                break
            
            
        speed_type = 'ema'

        print("predicted_video_dir: ", predicted_video_dir)
        
        dict_detection_reso_video = read_json_dir(predicted_video_dir)
        
        
        output_data_pickle_dir = predicted_video_dir + "jumping_number_result/jumpingNumber_resolution_selection/" + "intervalFrm-" + str(interval_frm) + "_speedType-" + str(speed_type) + "_minAcc-" + str(min_acc_threshold) + "/"
            
        test_x_instances_file = output_data_pickle_dir + "data_instance_speed_jumpingNumber_resolution_objectSizeRatio_xy.pkl"
        
        X = read_pickle_data(test_x_instances_file)[:, :-2]
        #print ("X shape:", X.shape)
        pca = PCA(n_components=min(n_components, X.shape[1])).fit(X)



        ModelClassifyObj = ModelClassifier()

        output_pickle_dir = predicted_video_dir +  "data_instance_xy/"  + "minAcc_" + str(min_acc_threshold) + "/"
        model_file1 = output_pickle_dir + "model_classifier_frame_rate_joblib_exclusive_" + str(predicted_video_dir.split('/')[-2])  + ".pkl"
        mor1 = joblib.load(model_file1)


        #x_input_arr_file = write_subDir + "all_other_trained_x_instances.pkl"
        #write_pickle_data(x_input_arr, x_input_arr_file)
        
        model_file2 = output_pickle_dir + "model_classifier_resolution_joblib_exclusive_" + str(predicted_video_dir.split('/')[-2])  + ".pkl"
        mor2 = joblib.load(model_file2)
    

        print("mor1, mor2: ", mor1, mor2)
        frm_no = 10000  # test only 10000 for paper ploting experiment len(dict_detection_reso_video[reso_list[0]])  #   min(len(imagePathLst), spf_arr.shape[1])
        
        acc_accumulate = 0.0
        time_spf_accumulate = 0.0
        calculated_frm_num = 0
        
        ans_reso_indx = 0
        last_reso_indx = 0

        last_frm_indx = 1
        current_frm_indx = interval_frm + 1                     # 1  starting index = 0  left frame no = 1
        
        DataGenerateObj = DataGenerate(data_dir)  
 
    
        print ("frm_no :", frm_no)
        acc_tmp = 0
        time_spf_tmp = 0
        calc_frm_num_tmp = 0        # debug only

        
        highest_reso = reso_list[0]
        curre_reso = reso_list[ans_reso_indx]
        acc_tmp, time_spf_tmp, calc_frm_num_tmp = self.get_online_video_analytics_accuracy_spf(data_dir, dict_detection_reso_video, last_frm_indx, interval_frm, highest_reso, curre_reso)
        
        acc_accumulate += acc_tmp
        time_spf_accumulate += time_spf_tmp
        calculated_frm_num += calc_frm_num_tmp
                
        # initially used resolution
        #last_frm_box = dict_detection_reso_video[reso_list[ans_reso_indx]][str(frm_lst_indx)][object_name]      # initialize
          
        #print("last_frm_box: ", last_frm_box)

        # use predicted result to apply to this new video and get delay and accuracy
        #delay_arr = []      
        #up_to_delay = max(0, spf_arr[ans_reso_indx][0] - 1.0/PLAYOUT_RATE)             # up to current delay, processing time - streaming time

        arr_ema_absolute_velocity = np.zeros((4, 2))

        acc_pred_arr = []
        delay_arr = []      
        up_to_delay = max(0, float(dict_detection_reso_video[reso_list[ans_reso_indx]][str(last_frm_indx)]['spf'])  - 1.0/PLAYOUT_RATE)         # up to current delay, processing time - streaming time
     
        
        while(current_frm_indx < frm_no):
            
            # calculate the relative feature
            
            frm_reso = reso_list[ans_reso_indx]
            arr_ema_absolute_velocity = DataGenerateObj.get_movement_feature(dict_detection_reso_video, current_frm_indx, last_frm_indx, frm_reso, arr_ema_absolute_velocity)
            
            feature_vect_mean = np.mean(arr_ema_absolute_velocity, axis = 0)
            feature_vect_var = np.var(arr_ema_absolute_velocity, axis = 0)
            #print("arr_ema_speed: ", feature_vect_mean, feature_vect_var, np.asarray([feature_vect_mean, feature_vect_var]))
            # get the jumping number y
            
            arr_movement_feature = np.hstack((arr_ema_absolute_velocity.flatten(), feature_vect_mean, feature_vect_var))
            #print("abso_velocity_feature_x: ", abso_velocity_feature_x.shape)
            
            #feature_x_object_size = DataGenerateObj.get_object_size_x(dict_detection_reso_video, current_frm_indx, ans_reso_indx)      
            
            feature_x_object_size = DataGenerateObj.get_object_size_change(dict_detection_reso_video, current_frm_indx, ans_reso_indx, last_frm_indx, last_reso_indx)
            #optical_flow_next_pt = self.get_optical_flow_feature(video_frame_dir, dict_detection_reso_video, current_frm_indx, ans_reso_indx, last_frm_indx, last_reso_indx)

                
            #feature_x = np.hstack((arr_movement_feature, feature_x_object_size))
            
            try:
                optical_flow_next_pt = DataGenerateObj.get_optical_flow_feature(video_frame_dir, dict_detection_reso_video, current_frm_indx, ans_reso_indx, last_frm_indx, last_reso_indx)
                    
            except:
                last_reso_indx = ans_reso_indx
                # predict how many frame jumped from this starting point
                
                #print("arr_ema_absolute_velocity ans_jfr ans_reso_indx aver_acc: ", arr_ema_absolute_velocity, ans_jfr, ans_reso_indx, aver_acc)
                
                print("optical_flow_next_pt erro: ", current_frm_indx)

                
                last_frm_indx = current_frm_indx
                current_frm_indx += 1  #
        
                continue
            
            
            feature_x = np.hstack((arr_movement_feature, feature_x_object_size, optical_flow_next_pt))
            #ans_jfr, ans_reso_indx, aver_acc = self.predict_next_configuration_jumping_frm_reso(dict_detection_reso_video, min_acc_threshold, curent_frm_indx, FRM_NO)
            # predict how many frame jumped from this starting point

            #print ("feature_x cur_absolute_speed: " , cur_absolute_speed)
        
            #print ("feature_x : ", feature_x.shape, feature_x)
            
            predicted_y_frameRate = ModelClassifyObj.test_on_data_y_unknown(mor1, feature_x, pca)
            
            predicted_y_Reso = ModelClassifyObj.test_on_data_y_unknown(mor2, feature_x, pca)
            
            #ModelClassifyObj.test_on_data_y_known_two_models(mor1, feature_x, y_test1, pca, mor2, feature_x, y_test2, pca2)
                
            jumping_frm_number = int(predicted_y_frameRate)   # int(predicted_y[0][0])
            ans_reso_indx = int(predicted_y_Reso)             #  int(predicted_y[0][1])
            
            if jumping_frm_number > max_jump_number:
                    jumping_frm_number = max_jump_number
                    
            last_reso_indx = ans_reso_indx
            # get delay up to this segment
            #up_to_delay = max(0, spf_arr[ans_reso_indx][frm_curr_indx] - (1.0/PLAYOUT_RATE) * jumping_frm_number)
            #delay_arr.append(up_to_delay)
            
            #print ("jumping_frm_number: ", jumping_frm_number, ans_reso_indx)

            highest_reso = reso_list[0]
            curre_reso = reso_list[ans_reso_indx]
            acc_tmp, time_spf_tmp, calc_frm_num_tmp = self.get_online_video_analytics_accuracy_spf(data_dir, dict_detection_reso_video, current_frm_indx, jumping_frm_number, highest_reso, curre_reso)
            #print ("feature_x : ", feature_x.shape, feature_x, acc_tmp, jumping_frm_number, ans_reso_indx)
            
            
            acc_pred_arr.append(acc_tmp)
            up_to_delay = max(0, time_spf_tmp- (1.0/PLAYOUT_RATE) * jumping_frm_number)
            #up_to_delay = max(0, float(dict_detection_reso_video[reso_list[ans_reso_indx]][str(current_frm_indx)]['spf'])- (1.0/PLAYOUT_RATE) * jumping_frm_number)
            delay_arr.append(up_to_delay)
   
    
            calculated_frm_num += calc_frm_num_tmp
            #curr_detect_time += spf_arr[ans_reso_indx][frm_curr_indx]
    
            last_frm_indx = current_frm_indx
            current_frm_indx += jumping_frm_number  #
            #EMA_absolute_prev = EMA_absolute_curr   # update EMA
            
            
            
        acc_pred_arr = np.asarray(acc_pred_arr)
        delay_arr = np.asarray(delay_arr)
        print ("get_prediction_acc_delay acc_pred_arr, delay_arr: ",  acc_pred_arr.shape, delay_arr.shape, output_pickle_dir)
        
        
        detect_out_result_dir = output_pickle_dir + "video_applied_detection_result/"
        if not os.path.exists(detect_out_result_dir):
            os.mkdir(detect_out_result_dir)

        arr_acc_segment_file = detect_out_result_dir + "arr_acc_segment_.pkl"
        arr_delay_up_to_segment_file = detect_out_result_dir + "arr_delay_up_to_segment_.pkl"
        write_pickle_data(acc_pred_arr, arr_acc_segment_file)
        write_pickle_data(delay_arr, arr_delay_up_to_segment_file)
        

        #print ("acc_average, time_spf: ", acc_average, time_spf)
        
        return acc_pred_arr, delay_arr



    def get_one_video_spf_spf_each_frm(self, data_dir, predicted_video_dir, min_acc, imagePathLst):
        
        # get each fram accuarcy spf
        
        video_frame_dir =  predicted_video_dir   # +  '*_frames'  # 'car_traffic_' + str(video_id) + '_frames'
        
        for root, subdirs, files in os.walk(video_frame_dir):
            flag = False
            for frame_folder_name in subdirs:
                if '_frames' in frame_folder_name:
                    video_frame_dir = video_frame_dir + frame_folder_name
                    flag = True
                    break
            if flag:
                break
            
            
        speed_type = 'ema'

        print("predicted_video_dir: ", predicted_video_dir)
        
        dict_detection_reso_video = read_json_dir(predicted_video_dir)
        
        
        output_data_pickle_dir = predicted_video_dir + "jumping_number_result/jumpingNumber_resolution_selection/" + "intervalFrm-" + str(interval_frm) + "_speedType-" + str(speed_type) + "_minAcc-" + str(min_acc_threshold) + "/"
            
        test_x_instances_file = output_data_pickle_dir + "data_instance_speed_jumpingNumber_resolution_objectSizeRatio_xy.pkl"
        
        X = read_pickle_data(test_x_instances_file)[:, :-2]
        #print ("X shape:", X.shape)
        pca = PCA(n_components=min(n_components, X.shape[1])).fit(X)



        ModelClassifyObj = ModelClassifier()

        output_pickle_dir = predicted_video_dir +  "data_instance_xy/"  + "minAcc_" + str(min_acc_threshold) + "/"
        model_file1 = output_pickle_dir + "model_classifier_frame_rate_joblib_exclusive_" + str(predicted_video_dir.split('/')[-2])  + ".pkl"
        mor1 = joblib.load(model_file1)


        #x_input_arr_file = write_subDir + "all_other_trained_x_instances.pkl"
        #write_pickle_data(x_input_arr, x_input_arr_file)
        
        model_file2 = output_pickle_dir + "model_classifier_resolution_joblib_exclusive_" + str(predicted_video_dir.split('/')[-2])  + ".pkl"
        mor2 = joblib.load(model_file2)
    

        print("mor1, mor2: ", mor1, mor2)
        frm_no = 3000  # test only 10000 for paper ploting experiment len(dict_detection_reso_video[reso_list[0]])  #   min(len(imagePathLst), spf_arr.shape[1])
        
        acc_accumulate = 0.0
        time_spf_accumulate = 0.0
        calculated_frm_num = 0
        
        ans_reso_indx = 0
        last_reso_indx = 0

        last_frm_indx = 1
        current_frm_indx = interval_frm + 1                     # 1  starting index = 0  left frame no = 1
        
        DataGenerateObj = DataGenerate(data_dir)  
 
    
        print ("frm_no :", frm_no)
        acc_tmp = 0
        time_spf_tmp = 0
        calc_frm_num_tmp = 0        # debug only

        
        highest_reso = reso_list[0]
        curre_reso = reso_list[ans_reso_indx]
        acc_tmp, time_spf_tmp, calc_frm_num_tmp = self.get_online_video_analytics_accuracy_spf(data_dir, dict_detection_reso_video, last_frm_indx, interval_frm, highest_reso, curre_reso)
        
        acc_accumulate += acc_tmp
        time_spf_accumulate += time_spf_tmp
        calculated_frm_num += calc_frm_num_tmp
                
        # initially used resolution
        #last_frm_box = dict_detection_reso_video[reso_list[ans_reso_indx]][str(frm_lst_indx)][object_name]      # initialize
          
        #print("last_frm_box: ", last_frm_box)

        # use predicted result to apply to this new video and get delay and accuracy
        #delay_arr = []      
        #up_to_delay = max(0, spf_arr[ans_reso_indx][0] - 1.0/PLAYOUT_RATE)             # up to current delay, processing time - streaming time

        arr_ema_absolute_velocity = np.zeros((4, 2))

        acc_pred_arr = []
        spf_arr = []      
        up_to_delay = max(0, float(dict_detection_reso_video[reso_list[ans_reso_indx]][str(last_frm_indx)]['spf'])  - 1.0/PLAYOUT_RATE)         # up to current delay, processing time - streaming time
     
        
        while(current_frm_indx < frm_no):
            
            # calculate the relative feature
            
            frm_reso = reso_list[ans_reso_indx]
            arr_ema_absolute_velocity = DataGenerateObj.get_movement_feature(dict_detection_reso_video, current_frm_indx, last_frm_indx, frm_reso, arr_ema_absolute_velocity)
            
            feature_vect_mean = np.mean(arr_ema_absolute_velocity, axis = 0)
            feature_vect_var = np.var(arr_ema_absolute_velocity, axis = 0)
            #print("arr_ema_speed: ", feature_vect_mean, feature_vect_var, np.asarray([feature_vect_mean, feature_vect_var]))
            # get the jumping number y
            
            arr_movement_feature = np.hstack((arr_ema_absolute_velocity.flatten(), feature_vect_mean, feature_vect_var))
            #print("abso_velocity_feature_x: ", abso_velocity_feature_x.shape)
            
            #feature_x_object_size = DataGenerateObj.get_object_size_x(dict_detection_reso_video, current_frm_indx, ans_reso_indx)      
            
            feature_x_object_size = DataGenerateObj.get_object_size_change(dict_detection_reso_video, current_frm_indx, ans_reso_indx, last_frm_indx, last_reso_indx)
            #optical_flow_next_pt = self.get_optical_flow_feature(video_frame_dir, dict_detection_reso_video, current_frm_indx, ans_reso_indx, last_frm_indx, last_reso_indx)

                
            #feature_x = np.hstack((arr_movement_feature, feature_x_object_size))
            
            try:
                optical_flow_next_pt = DataGenerateObj.get_optical_flow_feature(video_frame_dir, dict_detection_reso_video, current_frm_indx, ans_reso_indx, last_frm_indx, last_reso_indx)
                    
            except:
                last_reso_indx = ans_reso_indx
                # predict how many frame jumped from this starting point
                
                #print("arr_ema_absolute_velocity ans_jfr ans_reso_indx aver_acc: ", arr_ema_absolute_velocity, ans_jfr, ans_reso_indx, aver_acc)
                
                print("optical_flow_next_pt erro: ", current_frm_indx)

                
                last_frm_indx = current_frm_indx
                current_frm_indx += 1  #
        
                continue
            
            
            feature_x = np.hstack((arr_movement_feature, feature_x_object_size, optical_flow_next_pt))
            #ans_jfr, ans_reso_indx, aver_acc = self.predict_next_configuration_jumping_frm_reso(dict_detection_reso_video, min_acc_threshold, curent_frm_indx, FRM_NO)
            # predict how many frame jumped from this starting point

            #print ("feature_x cur_absolute_speed: " , cur_absolute_speed)
        
            #print ("feature_x : ", feature_x.shape, feature_x)
            
            predicted_y_frameRate = ModelClassifyObj.test_on_data_y_unknown(mor1, feature_x, pca)
            
            predicted_y_Reso = ModelClassifyObj.test_on_data_y_unknown(mor2, feature_x, pca)
            
            #ModelClassifyObj.test_on_data_y_known_two_models(mor1, feature_x, y_test1, pca, mor2, feature_x, y_test2, pca2)
                
            jumping_frm_number = int(predicted_y_frameRate)   # int(predicted_y[0][0])
            ans_reso_indx = int(predicted_y_Reso)             #  int(predicted_y[0][1])
            
            if jumping_frm_number > max_jump_number:
                    jumping_frm_number = max_jump_number
                    
            last_reso_indx = ans_reso_indx
            # get delay up to this segment
            #up_to_delay = max(0, spf_arr[ans_reso_indx][frm_curr_indx] - (1.0/PLAYOUT_RATE) * jumping_frm_number)
            #delay_arr.append(up_to_delay)
            
            #print ("jumping_frm_number: ", jumping_frm_number, ans_reso_indx)

            highest_reso = reso_list[0]
            curre_reso = reso_list[ans_reso_indx]
            acc_tmp, time_spf_tmp, calc_frm_num_tmp = self.get_online_video_analytics_accuracy_spf(data_dir, dict_detection_reso_video, current_frm_indx, jumping_frm_number, highest_reso, curre_reso)
            #print ("feature_x : ", feature_x.shape, feature_x, acc_tmp, jumping_frm_number, ans_reso_indx)
            
            acc_lst, time_lst, calculated_frm_num = self.get_online_video_analytics_accuracy_spf_each_frame(data_dir, dict_detection_reso_video, current_frm_indx, jumping_frm_number, highest_reso, curre_reso)
            
            acc_pred_arr  += acc_lst
            
            #up_to_delay = max(0, time_spf_tmp- (1.0/PLAYOUT_RATE) * jumping_frm_number)
            #up_to_delay = max(0, float(dict_detection_reso_video[reso_list[ans_reso_indx]][str(current_frm_indx)]['spf'])- (1.0/PLAYOUT_RATE) * jumping_frm_number)
            spf_arr += time_lst  #  .append(up_to_delay)
    
            calculated_frm_num += calc_frm_num_tmp
            #curr_detect_time += spf_arr[ans_reso_indx][frm_curr_indx]
    
            last_frm_indx = current_frm_indx
            current_frm_indx += jumping_frm_number  #
            #EMA_absolute_prev = EMA_absolute_curr   # update EMA
            
            
            
        tmp_acc_lst = []
        i = 0
        interval_sec_frm = 25    # 1 sec 25 frame
        while (i < len(acc_pred_arr)):
            if i+ interval_sec_frm < len(acc_pred_arr):
                
                tmp_acc_lst.append(sum(acc_pred_arr[i:i+interval_sec_frm])/interval_sec_frm)
            
            i += interval_sec_frm
            
            
        tmp_spf_lst = []
        i = 0
        interval_sec_frm = 25    # 1 sec 25 frame
        while (i < len(spf_arr)):
            if i+ interval_sec_frm < len(spf_arr):
                
                tmp_spf_lst.append(sum(spf_arr[i:i+interval_sec_frm]))
            
            i += interval_sec_frm
            
        acc_pred_arr = np.asarray(acc_pred_arr)
        spf_arr = np.asarray(spf_arr)
        print ("get_prediction_acc_delay acc_pred_arr, delay_arr: ",  acc_pred_arr.shape, spf_arr.shape, output_pickle_dir)
        
        
        detect_out_result_dir = output_pickle_dir + "video_applied_detection_result/"
        if not os.path.exists(detect_out_result_dir):
            os.mkdir(detect_out_result_dir)

        arr_acc_segment_file = detect_out_result_dir + "MOTrack_arr_acc_segment_.pkl"
        arr_spf_segment_file = detect_out_result_dir + "MOTrack_arr_spf_segment_.pkl"
        write_pickle_data(acc_pred_arr, arr_acc_segment_file)
        write_pickle_data(spf_arr, arr_spf_segment_file)
        

        #print ("acc_average, time_spf: ", acc_average, time_spf)
        
        return acc_pred_arr, spf_arr
    
    def read_video_frm(self, predicted_video_frm_dir):
        
        imagePathLst = sorted(glob(predicted_video_frm_dir + "*.jpg"))  # , key=lambda filePath: int(filePath.split('/')[-1][filePath.split('/')[-1].find(start)+len(start):filePath.split('/')[-1].rfind(end)]))          # [:75]   5 minutes = 75 segments

        #print ("imagePathLst: ", len(imagePathLst))
        
        return imagePathLst
        
    
    def get_image_name_id(self, frm_curr_indx):
        # from frm_curr_indx get image path name  e.g. index: 0 to 00001.jpg
        
        return str(frm_curr_indx).zfill(5)


    def draw_boundingbox_write(self, img_path_lst, img_out_parent_dir, dict_car_bounding_boxes, frm_curr_indx, ans_reso_indx, jumping_frm_number, cur_absolute_speed):
        
        
        resize_wh = reso_list[ans_reso_indx]  # resize width x height string
        
        resize_w = int(resize_wh.split("x")[0])
        resize_h = int(resize_wh.split("x")[1])
        

        start_frm_indx = frm_curr_indx
        cur_absolute_speed = np.mean(cur_absolute_speed, axis=0)
        average_speed = cur_absolute_speed/PLAYOUT_RATE  #  [s*(1/PLAYOUT_RATE) for s in cur_absolute_speed]
        #print ("dict_car_bounding_boxes: ", type(dict_car_bounding_boxes), dict_car_bounding_boxes, average_speed)
        
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
            
        while (start_frm_indx <= frm_curr_indx + jumping_frm_number):
            
            if start_frm_indx >= len(img_path_lst):
                break
            img_path = img_path_lst[start_frm_indx]
            #img_path = self.get_image_name_id(frm_curr_indx)
            img_name = img_path.split('/')[-1]
            #print ("img_path: ", img_path)
            img = cv2.imread(img_path)
                   
            #print ("frm_curr_indx img resize w h: ", frm_curr_indx, img.shape, resize_w, resize_h)
            
            img = cv2.resize(img, (resize_w, resize_h))
            
            for car_id, curr_bounding_box in dict_car_bounding_boxes.items():
                if start_frm_indx == frm_curr_indx:
                    next_frm_bounding_box = curr_bounding_box
                    #print ("resize img: ", img.shape)
                else:
                    next_frm_bounding_box = [curr_bounding_box[0] + average_speed[0], curr_bounding_box[1] + average_speed[1], 
                                             curr_bounding_box[2] + average_speed[0], curr_bounding_box[3] + average_speed[1]]
                                      
                #print ("curr_bounding_box curr_bounding_box: ", curr_bounding_box, next_frm_bounding_box, type(next_frm_bounding_box), next_frm_bounding_box[0], resize_w)
                # bounding_box: [x1, y1, x2, y2]  vertical height is y, horiztonal width is x
                x1 = int(next_frm_bounding_box[0])  # int(next_frm_bounding_box[0] * resize_w)
                y1 = int(next_frm_bounding_box[1])  # int(next_frm_bounding_box[1] * resize_h)
                
                x2 = int(next_frm_bounding_box[2])  # int(next_frm_bounding_box[2] * resize_w)
                y2 = int(next_frm_bounding_box[3])  # int(next_frm_bounding_box[3] * resize_h)
                
                """
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                """
                
                car_id = car_id.split('_')[-1]
                color = colors[int(car_id) % len(colors)]

                color = (0, 255, 0)
                thick = 4
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
                cv2.putText(img, object_name + "-" + str(car_id),(x1, y1-10),0, 0.75, (255,255,255),2)
            
            #print("imge: write:" , img_out_parent_dir + img_name)
            cv2.imwrite(img_out_parent_dir + img_name, img)
        
            
            start_frm_indx += 1
            

    def get_online_video_analytics_accuracy_spf(self, data_dir, dict_detection_reso_video, current_frm_indx, max_frm_interval, highest_reso, curre_reso):
        
        # get accuracy for jumping frame rate
        DataGenerateObj = DataGenerate(data_dir)
        
        dict_cars_each_jumped_frm_higest_reso =  dict_detection_reso_video[highest_reso][str(current_frm_indx)][object_name]
        
        jumping_frame_interval = 0
        acc_accumulate = 0.0   # np.zeros(jumping_frame_interval)       # the maximum 25 frame interval's acc in an array
        
        time_accumualate =  dict_detection_reso_video[highest_reso][str(current_frm_indx)]['spf']
        
        calculated_frm_num = 1
        start_time = time.time()

        while(jumping_frame_interval < max_frm_interval):
            next_frm_indx = current_frm_indx + jumping_frame_interval
            if next_frm_indx >= len(dict_detection_reso_video[curre_reso]):  # end of index
                break
            
            dict_cars_each_jumped_frm_curr_reso = dict_detection_reso_video[curre_reso][str(next_frm_indx)][object_name]
            
            #dict_cars_each_jumped_frm_other_reso = dict_detection_reso_video[curr_reso][str(next_frm_indx)][object_name]
            #print("dict_cars_each_jumped_frm_higest_reso: ", dict_cars_each_jumped_frm_higest_reso)
            #print("predict_next_configuration_jumping_frm_reso: xxx", dict_cars_each_jumped_frm_higest_reso)
            curr_acc = min_acc_threshold   # 0.0  # # no vechicle existing in the frame
            if dict_cars_each_jumped_frm_higest_reso is not None:
                #print("predict_next_configuration_jumping_frm_reso dict_cars_each_jumped_frm_higest_reso: ", dict_cars_each_jumped_frm_higest_reso)
                curr_acc = DataGenerateObj.calculate_accuray_with_highest_reso(dict_cars_each_jumped_frm_higest_reso, highest_reso, dict_cars_each_jumped_frm_curr_reso, curre_reso, min_acc_threshold)
                #print ("curr_acc: ", curr_acc)
                
            acc_accumulate +=curr_acc   #  .append(curr_acc)
            
            jumping_frame_interval += 1
            calculated_frm_num += 1
            
            #print("arr_acc_interval: ", arr_acc_interval)

        #if len(arr_acc_interval) == 0:
        #    print("xxxxxxxxxxxxxxxxxx empty: ", arr_acc_interval)
        
        #print("arr_acc_interval: ", arr_acc_interval)

        end_time = time.time()
        time_accumualate += (end_time - start_time)  # /max_frm_interval   # total time here
        
        return acc_accumulate, time_accumualate, calculated_frm_num
    


    def get_online_video_analytics_accuracy_spf_each_frame(self, data_dir, dict_detection_reso_video, current_frm_indx, max_frm_interval, highest_reso, curre_reso):
        # get accuracy for each frame  in the jumping frame rate
        DataGenerateObj = DataGenerate(data_dir)
        
        dict_cars_each_jumped_frm_higest_reso =  dict_detection_reso_video[highest_reso][str(current_frm_indx)][object_name]
        
        jumping_frame_interval = 0
        acc_lst = []   # np.zeros(jumping_frame_interval)       # the maximum 25 frame interval's acc in an array
        
        time_accumualate =  dict_detection_reso_video[highest_reso][str(current_frm_indx)]['spf']
        
        calculated_frm_num = 1
        start_time = time.time()

        while(jumping_frame_interval < max_frm_interval):
            next_frm_indx = current_frm_indx + jumping_frame_interval
            if next_frm_indx >= len(dict_detection_reso_video[curre_reso]):  # end of index
                break
            
            dict_cars_each_jumped_frm_curr_reso = dict_detection_reso_video[curre_reso][str(next_frm_indx)][object_name]
            
            #dict_cars_each_jumped_frm_other_reso = dict_detection_reso_video[curr_reso][str(next_frm_indx)][object_name]
            #print("dict_cars_each_jumped_frm_higest_reso: ", dict_cars_each_jumped_frm_higest_reso)
            #print("predict_next_configuration_jumping_frm_reso: xxx", dict_cars_each_jumped_frm_higest_reso)
            curr_acc = min_acc_threshold   # 0.0  # # no vechicle existing in the frame
            if dict_cars_each_jumped_frm_higest_reso is not None:
                #print("predict_next_configuration_jumping_frm_reso dict_cars_each_jumped_frm_higest_reso: ", dict_cars_each_jumped_frm_higest_reso)
                curr_acc = DataGenerateObj.calculate_accuray_with_highest_reso(dict_cars_each_jumped_frm_higest_reso, highest_reso, dict_cars_each_jumped_frm_curr_reso, curre_reso, min_acc_threshold)
                #print ("curr_acc: ", curr_acc)
                
            acc_lst.append(curr_acc)   #  .append(curr_acc)
            
            jumping_frame_interval += 1
            calculated_frm_num += 1
            
            #print("arr_acc_interval: ", arr_acc_interval)

        #if len(arr_acc_interval) == 0:
        #    print("xxxxxxxxxxxxxxxxxx empty: ", arr_acc_interval)
        
        #print("acc_lst: ", len(acc_lst), calculated_frm_num)
        
        
        end_time = time.time()
        time_accumualate += end_time - start_time
        
        time_lst = [time_accumualate/jumping_frame_interval] * calculated_frm_num
        
        return acc_lst, time_lst, calculated_frm_num



    def get_ground_truth_configuration(self, data_dir, dict_detection_reso_video, current_frm_indx):
        # get the ground truth configuration which satisfies the standard
        x = 1
        
        
    def show_vehicle_bounding_box_detection_result(self, data_dir, predicted_video_dir, min_acc, imagePathLst, write_detection_box_flag, feature_removed):
        #detect speaker after applying the predictive model for configuration adaptation
        #not real apply to video; use profiling results to simulate 
        # get bounding box of the test video in different resolutions (i.e profiling result of the video)
        # then draw on the frame when doing video analytics
        
        if write_detection_box_flag:
            img_path = imagePathLst[0]   # get a random image path
            img_out_parent_dir = '/'.join(img_path.split('/')[:-2]) + "/" + img_path.split('/')[-2] + "_boundingbox_out/"
            
            print ("img_out_parent_dir: ", img_out_parent_dir)
            
            
            if not os.path.exists(img_out_parent_dir):
                os.mkdir(img_out_parent_dir)
        
        
        video_frame_dir =  predicted_video_dir   # +  '*_frames'  # 'car_traffic_' + str(video_id) + '_frames'
        
        for root, subdirs, files in os.walk(video_frame_dir):
            flag = False
            for frame_folder_name in subdirs:
                if '_frames' in frame_folder_name:
                    video_frame_dir = video_frame_dir + frame_folder_name
                    flag = True
                    break
            if flag:
                break
            
        
        # speaker_box_arr, acc_arr, spf_arr = get_data_numpy(input_dir + predicted_video_dir)
        dict_detection_reso_video = read_json_dir(predicted_video_dir)
        
        
        print ("ppppredicted_video_dir: ", predicted_video_dir)
        output_pickle_dir = predicted_video_dir + "data_instance_xy_" + feature_removed + "/minAcc_" + str(min_acc) + "/"
        
        model_file_fr = output_pickle_dir + "model_classifier_frame_rate_joblib_exclusive_" + str(predicted_video_dir.split('/')[-2])  + ".pkl"       # with other videos
        model_file_reso = output_pickle_dir + "model_classifier_resolution_joblib_exclusive_" + str(predicted_video_dir.split('/')[-2])  + ".pkl"       # with other videos
        #print ("model_file:", model_file)
        
        
        test_x_instances_file = output_pickle_dir + "trained_x_instances.pkl"
        X = read_pickle_data(test_x_instances_file)
        #print ("X shape:", X.shape)
        
        pca = PCA(n_components=min(n_components, X.shape[1])).fit(X)
        ModelClassifyObj = ModelClassifier()
        mor_fr = joblib.load(model_file_fr)
        mor_reso = joblib.load(model_file_reso)

        frm_no = len(dict_detection_reso_video[reso_list[0]])  #   min(len(imagePathLst), spf_arr.shape[1])
        
        acc_accumulate = 0.0
        time_spf_accumulate_image_process = 0.0
        
        time_spf_accumulate_ml_prediction = 0.0
        
        calculated_frm_num = 0
        
        ans_reso_indx = 0
        last_reso_indx = 0

        last_frm_indx = 1
        current_frm_indx = interval_frm + 1                     # 1  starting index = 0  left frame no = 1
        
        DataGenerateObj = DataGenerate(data_dir)  
        # draw first segment use default resolution and index
        if write_detection_box_flag:
            car_bounding_boxes = dict_detection_reso_video[reso_list[ans_reso_indx]][str(last_frm_indx)][object_name]
            self.draw_boundingbox_write(imagePathLst, img_out_parent_dir, car_bounding_boxes, last_frm_indx, ans_reso_indx, interval_frm, [0, 0])

        print ("frm_no :", frm_no)
        acc_tmp = 0
        time_spf_tmp = 0
        calc_frm_num_tmp = 0        # debug only

        
        highest_reso = reso_list[0]
        curre_reso = reso_list[ans_reso_indx]
        acc_tmp, time_spf_tmp, calc_frm_num_tmp = self.get_online_video_analytics_accuracy_spf(data_dir, dict_detection_reso_video, last_frm_indx, interval_frm, highest_reso, curre_reso)
        
        acc_accumulate += acc_tmp
        time_spf_accumulate_image_process += time_spf_tmp
        calculated_frm_num += calc_frm_num_tmp
                
        # initially used resolution
        #last_frm_box = dict_detection_reso_video[reso_list[ans_reso_indx]][str(frm_lst_indx)][object_name]      # initialize
          
        #print("last_frm_box: ", last_frm_box)

        # use predicted result to apply to this new video and get delay and accuracy
        #delay_arr = []      
        #up_to_delay = max(0, spf_arr[ans_reso_indx][0] - 1.0/PLAYOUT_RATE)             # up to current delay, processing time - streaming time

        arr_ema_absolute_velocity = np.zeros((4, 2))
        
        while(current_frm_indx < frm_no):
            
            start_time = time.time()
            # calculate the relative feature
            
            frm_reso = reso_list[ans_reso_indx]
            arr_ema_absolute_velocity = DataGenerateObj.get_movement_feature(dict_detection_reso_video, current_frm_indx, last_frm_indx, frm_reso, arr_ema_absolute_velocity)
            
            feature_vect_mean = np.mean(arr_ema_absolute_velocity, axis = 0)
            feature_vect_var = np.var(arr_ema_absolute_velocity, axis = 0)
            #print("arr_ema_speed: ", feature_vect_mean, feature_vect_var, np.asarray([feature_vect_mean, feature_vect_var]))
            # get the jumping number y
            
            arr_movement_feature = np.hstack((arr_ema_absolute_velocity.flatten(), feature_vect_mean, feature_vect_var))
            #print("abso_velocity_feature_x: ", abso_velocity_feature_x.shape)
            
            #feature_x_object_size = DataGenerateObj.get_object_size_x(dict_detection_reso_video, current_frm_indx, ans_reso_indx)      
            
            feature_x_object_size = DataGenerateObj.get_object_size_change(dict_detection_reso_video, current_frm_indx, ans_reso_indx, last_frm_indx, last_reso_indx)
            optical_flow_next_pt = DataGenerateObj.get_optical_flow_feature(video_frame_dir, dict_detection_reso_video, current_frm_indx, ans_reso_indx, last_frm_indx, last_reso_indx)

            feature_x = np.hstack((arr_movement_feature, feature_x_object_size, optical_flow_next_pt)).reshape(1, -1)
            
            
            if feature_removed == 'keypoint_velocity_removed':
                feature_x = feature_x[:, 12:]
                #y_arr = y_arr[:, 12:]
            elif feature_removed == 'objectSizeChange_removed':
                feature_x = np.hstack((feature_x[:, 0:12], feature_x[:, 14:]))   # X_test[:, 12:14]
                #y_arr = np.hstack((y_arr[:, 0:12], y_arr[:, 14:]))
            elif feature_removed == 'opticalFlow_removed':
                feature_x = feature_x[:, 0:14]   #  114
                
            
            #ans_jfr, ans_reso_indx, aver_acc = self.predict_next_configuration_jumping_frm_reso(dict_detection_reso_video, min_acc_threshold, curent_frm_indx, FRM_NO)
            # predict how many frame jumped from this starting point

            #print ("feature_x cur_absolute_speed: " , cur_absolute_speed)
        
            #print ("feature_x : ", feature_x.shape, feature_x)
            
            jumping_frm_number, ans_reso_indx =  ModelClassifyObj.test_on_data_y_unknown_two_models(mor_fr, feature_x, pca, mor_reso, feature_x, pca)
        
            jumping_frm_number = int(jumping_frm_number)
            ans_reso_indx = int(ans_reso_indx)
            #jumping_frm_number = int(predicted_y[0][0])
            #ans_reso_indx = int(predicted_y[0][1])
            
            last_reso_indx = ans_reso_indx
            # get delay up to this segment
            #up_to_delay = max(0, spf_arr[ans_reso_indx][frm_curr_indx] - (1.0/PLAYOUT_RATE) * jumping_frm_number)
            #delay_arr.append(up_to_delay)
            
            #print ("jumping_frm_number: ", jumping_frm_number, ans_reso_indx)
            
            end_time = time.time()
    
            time_spf_accumulate_ml_prediction += (end_time - start_time)
            #draw bounding box on the image and output frm image with bounding box
            if write_detection_box_flag:
                car_bounding_boxes = dict_detection_reso_video[reso_list[ans_reso_indx]][str(current_frm_indx)][object_name]
                self.draw_boundingbox_write(imagePathLst, img_out_parent_dir, car_bounding_boxes, current_frm_indx, ans_reso_indx, jumping_frm_number, arr_ema_absolute_velocity)
            
            highest_reso = reso_list[0]
            curre_reso = reso_list[ans_reso_indx]
            acc_tmp, time_spf_tmp, calc_frm_num_tmp = self.get_online_video_analytics_accuracy_spf(data_dir, dict_detection_reso_video, current_frm_indx, jumping_frm_number, highest_reso, curre_reso)
            #print ("feature_x : ", feature_x.shape, feature_x, acc_tmp, jumping_frm_number, ans_reso_indx)
            acc_accumulate += acc_tmp
            time_spf_accumulate_image_process += time_spf_tmp
            calculated_frm_num += calc_frm_num_tmp
            #curr_detect_time += spf_arr[ans_reso_indx][frm_curr_indx]
    
            last_frm_indx = current_frm_indx
            current_frm_indx += jumping_frm_number  #
            #EMA_absolute_prev = EMA_absolute_curr   # update EMA
            
        
        acc_average = acc_accumulate/(calculated_frm_num)
        image_time_spf = time_spf_accumulate_image_process/calculated_frm_num
        
        ml_time_spf = time_spf_accumulate_ml_prediction/calculated_frm_num
        #print ("acc_average, time_spf: ", acc_average, time_spf)
        
        return acc_average, image_time_spf, ml_time_spf



    def get_acc_spf_under_different_minThreshold(self, data_dir, video_dir_lst):
        min_acc_threshold_lst = [0.9, 0.92, 0.94, 0.96, 0.98]  # , 1.0]
        acc_lst = []
        SPF_spent_lst = []
        ml_spf_spent_lst = []
        
        for min_acc_thres in min_acc_threshold_lst[1:2]:
                
            acc_average = 0.0
            spf_average = 0.0
            ml_spf_average = 0.0
            analyzed_video_lst = video_dir_lst[0:2]   # [0:10]
            for predicted_video_dir in analyzed_video_lst:
                predicted_video_dir = predicted_video_dir + '/' 
                
                predicted_video_frm_dir = "/".join(data_dir.split("/")[:-2]) + "/" + "_".join(predicted_video_dir.split("/")[:-2]) + "_1120x832_frames/"
                
                print ("predicted_video_frm_dir: ", predicted_video_frm_dir)  # ../input_output/speaker_video_dataset/sample_03_frames/
                
                
                imagePathLst = self.read_video_frm(predicted_video_frm_dir)
                write_detection_box_flag = False # True  # False
                
                # get one del
                #self.get_one_video_spf_latency(data_dir, predicted_video_dir, min_acc_thres, imagePathLst)
                #self.get_one_video_spf_spf_each_frm(data_dir, predicted_video_dir, min_acc_thres, imagePathLst)
               
                feature_removed = "all"   # objectSizeChange_removed"  # "objectSizeChange_removed"   # "all" 
                acc, img_spf, ml_spf = self.show_vehicle_bounding_box_detection_result(data_dir, predicted_video_dir, min_acc_thres, imagePathLst, write_detection_box_flag, feature_removed)
                acc_average += acc
                spf_average += img_spf
                ml_spf_average += ml_spf
                
            acc_lst.append(acc_average/len(analyzed_video_lst))
                
            SPF_spent_lst.append(spf_average/len(analyzed_video_lst))
            ml_spf_spent_lst.append(ml_spf_average/len(analyzed_video_lst))
            
        print("acc_lst, SPF_spent_lst: ", acc_lst, SPF_spent_lst, ml_spf_spent_lst)



if __name__=="__main__":
    
    data_obj = DataGenerate(data_dir)
    video_dir_lst = data_obj.video_dir_lst    # [5:6]   # [5, 15]


    videoApplyObj = VideoApply()            
        
    #for predicted_video_dir in video_dir_lst[0:6]:
        
        #videoApplyObj.get_prediction_acc_delay(predicted_video_dir, min_acc_threshold)
    
    
    videoApplyObj.get_acc_spf_under_different_minThreshold(data_dir, video_dir_lst)
