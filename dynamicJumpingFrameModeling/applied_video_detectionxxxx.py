#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11  10:46:05 2020

@author: fubao
"""

# after predicting the jumping frame number and resolution number on a testing video
# we apply the result to a testing video for pose estimation,
# then observe the result

import sys
import os
import numpy as np
import joblib

from data_file_process import write_pickle_data
from data_file_process import read_pickle_data
from common_plot import plotOneScatterLine
from common_plot import plotOneScatter
from get_data_ground_truth import DataGenerate

from sklearn.decomposition import PCA

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from profiling.common_prof import computeOKS_1to1
from profiling.common_prof import dataDir3
from profiling.common_prof import NUM_KEYPOINT   
from profiling.common_prof import PLAYOUT_RATE


class ModelApply(object):
    def __init__(self):
        #self.data_classification_dir = data_classification_dir  # input file directory
        pass

    def test_on_data_y_unknown(self, model, X_test, pca):
        # test on a test data with model already trained, y label unknown
        # X_test is just one instance
        #X_test = np.reshape(X_test, (1, -1))
        #print ("test_on_data_y_unknown X_test shape: ", X_test)
        #pca = PCA(n_components=10)
        
        X_test = np.reshape(X_test, (1, -1))
        X_test = pca.transform(X_test)
        #print ("test_on_data_y_unknown X_test pcaaa shape: ", X_test)
        y_test_pred = model.predict(X_test)
        
        #print ("test_on_data_y_unknown y_test_pred r2: ", X_test,  y_test_pred)
        
        return y_test_pred
    
    
    def get_prediction_acc_delay(self, predicted_video_dir):
        # get the predicted video's accuracy and predicted_video_dird delay
        #  predicted_video_id as the testing data
        # predicted_video_dir:  such as output_021_dance
        # predicted_out_file is the prediction jumping number and delay
        
        interval_frm = 1
        
        data_pose_keypoint_dir = dataDir3 + predicted_video_dir

        data_pickle_dir = dataDir3 + predicted_video_dir + 'frames_pickle_result/'
        ResoDataGenerateObj = DataGenerate()
        config_est_frm_arr, acc_frame_arr, spf_frame_arr = ResoDataGenerateObj.get_data_numpy(data_pose_keypoint_dir, data_pickle_dir)
     
        print ("get_prediction_acc_delay config_est_frm_arr: ", config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        # get jumping frm number prediction first
        output_pickle_dir = dataDir3 + predicted_video_dir + "jumping_number_result/" 

        subDir_jumping_frm = output_pickle_dir + "jumping_number_prediction/intervalFrm-1_speedType-ema_minAcc-0.92/"
        model_file = subDir_jumping_frm + "model_regression_jumping_frm.joblib.pkl"       # with other videos
        trained_x_instances_file = subDir_jumping_frm + "trained_x_instances.pkl"
        model_jumping_frm = joblib.load(model_file)
        X_jumping_frm = read_pickle_data(trained_x_instances_file)
        pca_jumping_frm = PCA(n_components=10).fit(X_jumping_frm)
        
        
        # get resolution prediction 
        subDir_reso = output_pickle_dir + "resolution_prediction/intervalFrm-1_speedType-ema_minAcc-0.92/"
        model_file = subDir_reso + "model_classifi_reso.joblib.pkl"       # with other videos
        trained_x_instances_file = subDir_jumping_frm + "trained_x_instances.pkl"
        model_reso = joblib.load(model_file)
        
        X_reso = read_pickle_data(trained_x_instances_file)
        pca_reso = PCA(n_components=10).fit(X_reso)
        
        
        # get estimated accuracy and delay
        # start from 2nd frame 
        reso_curr = 0   # current resolution
        prev_indx = 0
        current_indx = 1
        FRM_NO =  spf_frame_arr[reso_curr].shape[0]                # total frame no for the whole video

        # use predicted result to apply to this new video and get delay and accuracy
        delay_arr = []      
        up_to_delay = max(0, spf_frame_arr[reso_curr][0] - 1.0/PLAYOUT_RATE)         # up to current delay, processing time - streaming time
        
        acc_arr = []
        acc_seg = 0        # current segment's acc
        
        segment_indx = 0
        
        jumping_frm_number_lst = []
        
        arr_ema_absolute_speed = np.zeros((NUM_KEYPOINT, 3))
        arr_ema_relative_speed  = np.zeros((8, 3))    # 8 points to calculate relative speed
        
        while(current_indx < FRM_NO):
            # get jumping frame number 
            # get feature x for up to current frame
            current_frm_est = config_est_frm_arr[reso_curr][current_indx]  # detected_est_frm_arr[current_indx] #  current frame is detected, so we use

            #print("detected_est_frm_arr: ", len(detected_est_frm_arr), current_indx, prev_used_indx)
            prev_frm_est = config_est_frm_arr[reso_curr][prev_indx]  # last frame used detected_est_frm_arr[prev_used_indx]
                
            arr_ema_absolute_speed = ResoDataGenerateObj.get_absolute_speed(current_frm_est, prev_frm_est, interval_frm, arr_ema_absolute_speed)

            # get relative speed 
            arr_ema_relative_speed = ResoDataGenerateObj.get_relative_speed_to_body_center(current_frm_est, prev_frm_est, interval_frm, arr_ema_relative_speed)
            
            feature_x_absolute = ResoDataGenerateObj.get_feature_x(arr_ema_absolute_speed)

            feature_x_relative = ResoDataGenerateObj.get_feature_x(arr_ema_relative_speed)
            feature_x = np.hstack((feature_x_absolute, feature_x_relative))
                    
            predicted_jumping_frm = self.test_on_data_y_unknown(model_jumping_frm, feature_x, pca_jumping_frm)
            
            predicted_reso = self.test_on_data_y_unknown(model_reso, feature_x, pca_reso)

            #print ("get_prediction_acc_delay: ", predicted_y)
            
            jumping_frm_number = int(predicted_jumping_frm)   # int(predicted_y[0][0])
            
            reso_curr = int(predicted_reso)   # int(predicted_y[0][1])
            
            
            # get accuracy of this segment
            acc_seg = self.get_accuracy_segment(current_indx, jumping_frm_number, reso_curr, config_est_frm_arr)
            
            # get delay up to this segment
            up_to_delay = max(0, spf_frame_arr[reso_curr][current_indx] - (1.0/PLAYOUT_RATE) * jumping_frm_number)
            
            delay_arr.append(up_to_delay)
            acc_arr.append(acc_seg)
            
            prev_indx = current_indx            # update prev_indx as current index
            
            current_indx += jumping_frm_number         # not jumping_frm_number + 1
            segment_indx += 1
            
            #jumping_frm_number_lst.append(jumping_frm_number)
            
            #print ("get_prediction_acc_delay current_indx: ", FRM_NO, current_indx, segment_indx, acc_seg, up_to_delay)
            
            #break   # debug only

        acc_arr = np.asarray(acc_arr)
        delay_arr = np.asarray(delay_arr)
        print ("get_prediction_acc_delay acc_arr, delay_arr: ",  acc_arr, delay_arr)
        
        detect_out_result_dir = subDir_reso + "video_applied_detection_result/"
        if not os.path.exists(detect_out_result_dir):
            os.mkdir(detect_out_result_dir)

        arr_acc_segment_file = detect_out_result_dir + "arr_acc_segment_.pkl"
        arr_delay_up_to_segment_file = detect_out_result_dir + "arr_delay_upt_to_segment_.pkl"
        write_pickle_data(acc_arr, arr_acc_segment_file)
        write_pickle_data(delay_arr, arr_delay_up_to_segment_file)
     
        return 

    def get_accuracy_segment(self, start_frm_indx, jumping_frm_number, reso_curr, config_est_frm_arr):
        # get the accuracy when jumping frm number
          
        acc_cur = 0
        accumulated_acc = 0
        ref_pose = config_est_frm_arr[0][start_frm_indx]     # reference pose ground truth
          
        end_indx = start_frm_indx + jumping_frm_number
        curr_indx = start_frm_indx
        while (curr_indx < end_indx):
          
          # get accuracy
            if curr_indx >= config_est_frm_arr.shape[1]:  # finished video streaming
                jumping_frm_number = curr_indx - start_frm_indx   # last segment
                break
            curr_pose = config_est_frm_arr[reso_curr][curr_indx]
            oks = computeOKS_1to1(ref_pose, curr_pose, sigmas = None)     # oks with reference pose
            accumulated_acc += oks
      
            curr_indx += 1
        average_acc = accumulated_acc/(jumping_frm_number)  # jumping_frm_number include the start_frame_index
        
        # print ("get_accuracy_segment average_acc: ", average_acc)
        return average_acc
    
    
    
if __name__== "__main__": 
    
    model_obj = ModelApply()
    
    video_dir_lst =  ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/']
    
    for predicted_video_dir in video_dir_lst:
        
        #predicted_video_dir = 'output_021_dance/'     # select different video id for testing
        #model_obj.train_rest_test_one_video(predicted_video_dir)
        
        model_obj.get_prediction_acc_delay(predicted_video_dir)