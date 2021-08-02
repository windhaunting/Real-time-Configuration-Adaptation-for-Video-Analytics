#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:53:51 2020

@author: fubao
"""

# read the frames from the video and then applied the adaptive configuration result to the video analytics

import os
import time
import cv2

import numpy as np
from glob import glob

from data_preproc import input_dir
from data_preproc import resoStrLst
from data_preproc import PLAYOUT_RATE

from data_preproc import read_numpy
from data_preproc import bb_intersection_over_union
from data_preproc import write_pickle_data
from data_preproc import read_pickle_data
from read_feature_speaker import *
from prediction_speaker_training_test import *


class VideoApply(object):
    def __init__(self):
        pass
    
    

    
    def get_accuracy_segment(self, start_frm_indx, jumping_frm_number, reso_curr, speaker_box_arr):
        # get the accuracy when jumping frm number
          
        accumulated_acc = 0
        ref_box = speaker_box_arr[0][start_frm_indx]     # reference pose ground truth
          
        end_indx = start_frm_indx + jumping_frm_number
        curr_indx = start_frm_indx
        while (curr_indx < end_indx):
          
          # get accuracy
            if curr_indx >= speaker_box_arr.shape[1]:  # finished video streaming
                jumping_frm_number = curr_indx - start_frm_indx   # last segment
                break
            curr_box = speaker_box_arr[reso_curr][curr_indx]
            sdq = bb_intersection_over_union(ref_box, curr_box)     # oks with reference pose
            accumulated_acc += sdq
      
            curr_indx += 1
        average_acc = accumulated_acc/(jumping_frm_number)  # jumping_frm_number include the start_frame_index
        
        # print ("get_accuracy_segment average_acc: ", average_acc)
        return average_acc
    
    def get_prediction_acc_delay(self, predicted_video_dir, min_acc_threshold):
        # after applying adaptation of configuration
        
        # get the predicted video's accuracy anpredicted_video_dird delay
        #  predicted_video_id as the testing data
        # predicted_video_dir:  such as output_021_dance
        # predicted_out_file is the prediction jumping number and delay
        
        #interval_frm = 1
        speaker_box_arr, acc_arr, spf_arr = get_data_numpy(input_dir + predicted_video_dir)


        output_pickle_dir = input_dir + predicted_video_dir + "data_instance_xy/minAcc_" + str(min_acc_threshold) + "/"
        
        
        model_file = output_pickle_dir + "model_regression.joblib" + "_exclusive_" + str(predicted_video_dir[:-1])  + ".pkl"       # with other videos
                
        # read the model 
        #model_dir = input_dir + "all_data_instance_xy/minAcc_" + str(min_acc) +"/"
        #model_file = model_dir + "model_regression.joblib_all_videos.pkl"    
        
        
        test_x_instances_file = output_pickle_dir + "trained_x_instances.pkl"
        X = read_pickle_data(test_x_instances_file)

        pca = PCA(n_components=min(n_components, X.shape[1])).fit(X)
        
        model = joblib.load(model_file)
        ModelRegressionObj = ModelRegression()
       
        reso_no = spf_arr.shape[0]
        frm_no = spf_arr.shape[1]
        
        
        ans_reso_indx = 0
        
        k = 25     # 
        frm_lst_indx = 0
        frm_curr_indx = k          # traverse from frame kth (index starts at 0)
        ema_absolute_curr = 0.0
        
        # initially used resolution
        last_frm_box = speaker_box_arr[ans_reso_indx][frm_lst_indx]   # initialize
          
        # use predicted result to apply to this new video and get delay and accuracy
        delay_arr = []      
        up_to_delay = max(0, spf_arr[ans_reso_indx][0] - 1.0/PLAYOUT_RATE)         # up to current delay, processing time - streaming time
        
        
        acc_pred_arr = []
        
        data_one_video_instances_xy = blist()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        while(frm_curr_indx < frm_no):
            
            # calculate the relative feature
            
            curre_frm_box = speaker_box_arr[ans_reso_indx][frm_curr_indx]
            
            cur_absolute_speed = get_absolute_speed(last_frm_box, frm_lst_indx, curre_frm_box, frm_curr_indx)
             
            ema_absolute_curr = get_ema_absolute_speed(ema_absolute_curr, cur_absolute_speed)
             
            normalized_object_ratio = get_object_size(curre_frm_box)
            #print ("normalized_object_ratio: " , normalized_object_ratio)
    
            feature_x = np.hstack((ema_absolute_curr, normalized_object_ratio))
            #print ("feature_x: " , feature_x)
            
            #jumping_frm_number, current_aver_acc = estimate_frame_rate(speaker_box_arr, acc_arr, frm_curr_indx, ans_reso_indx, max_jump_number, min_acc_threshold)
            #ans_reso_indx, average_acc = get_reso_selected(speaker_box_arr, current_aver_acc, frm_curr_indx, jumping_frm_number, min_acc_threshold)
            
            #print ("jumping_frm_number:feature_x,  ", jumping_frm_number, feature_x, ans_reso_indx)
            
            predicted_y = ModelRegressionObj.test_on_data_y_unknown(model, feature_x, pca)
            
            #print ("get_prediction_acc_delay: ", predicted_y)
            
            jumping_frm_number = int(predicted_y[0][0])
            
            ans_reso_indx = int(predicted_y[0][1])
            
            # get delay up to this segment
            up_to_delay = max(0, spf_arr[ans_reso_indx][frm_curr_indx] - (1.0/PLAYOUT_RATE) * jumping_frm_number)
            delay_arr.append(up_to_delay)
            
            
            acc_seg = self.get_accuracy_segment(frm_curr_indx, jumping_frm_number, ans_reso_indx, speaker_box_arr)
            
            acc_pred_arr.append(acc_seg)
            
            #curr_detect_time += spf_arr[ans_reso_indx][frm_curr_indx]
    
            last_frm_box = curre_frm_box
            
            frm_curr_indx += jumping_frm_number  #
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

    def show_detection_result_video_analytics(self, video_test_dir, min_acc):
        # after flexible configuration prediction traing, apply to a video stream and
        #show the detection result
        # here we simulate the add the predicted result on the test video
        # run get_prediction_acc_delay(self, predicted_video_dir, min_acc_threshold) in training_test_speaker.py to get predicted result
        
        prediction_dir =  input_dir +  video_test_dir  + "data_instance_xy/minAcc_" + str(min_acc) + "video_applied_detection_result/"
        
        acc_file =  prediction_dir + "arr_acc_segment_.pkl"
        
        seg_acc = read_pickle_data(acc_file)


    def read_video_frm(self, predicted_video_frm_dir):
        
        imagePathLst = sorted(glob(predicted_video_frm_dir + "*.jpg"))  # , key=lambda filePath: int(filePath.split('/')[-1][filePath.split('/')[-1].find(start)+len(start):filePath.split('/')[-1].rfind(end)]))          # [:75]   5 minutes = 75 segments

        #print ("imagePathLst: ", len(imagePathLst))
        
        return imagePathLst
        
    
    def get_image_name_id(self, frm_curr_indx):
        # from frm_curr_indx get image path name  e.g. index: 0 to 00001.jpg
        
        return str(frm_curr_indx).zfill(5)
    

    def draw_boundingbox_write(self, img_path_lst, img_out_parent_dir, speaker_box_arr, frm_curr_indx, ans_reso_indx, jumping_frm_number, cur_absolute_speed):
        
        
        resize_wh = resoStrLst[ans_reso_indx]  # resize width x height string
        
        resize_w = int(resize_wh.split("x")[0])
        resize_h = int(resize_wh.split("x")[1])
        
        boundingbox = speaker_box_arr[ans_reso_indx][frm_curr_indx] 

        next_frm_bounding_box = boundingbox
        start_frm_indx = frm_curr_indx
        
        average_speed = [s/jumping_frm_number for s in cur_absolute_speed] #  [s*(1/PLAYOUT_RATE) for s in cur_absolute_speed]
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
            
            #print ("resize img: ", img.shape)
            next_frm_bounding_box = [next_frm_bounding_box[0] + average_speed[1], next_frm_bounding_box[1] + average_speed[0], 
                                     next_frm_bounding_box[2] + average_speed[1], next_frm_bounding_box[3] + average_speed[0]]
                                     
            #print ("boundingbox img: ", boundingbox)
            # bounding_box: [y1, x1, y2, x2]  vertical height is y, horiztonal width is x
            x1 = int(next_frm_bounding_box[1] * resize_w)
            y1 = int(next_frm_bounding_box[0] * resize_h)
            
            x2 = int(next_frm_bounding_box[3] * resize_w)
            y2 = int(next_frm_bounding_box[2] * resize_h)
            color = (0, 255, 0)
            thick = 7
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
            
            cv2.imwrite(img_out_parent_dir + img_name, img)
        
            start_frm_indx += 1
            
        
    def get_online_video_analytics_accuracy_spf(self, speaker_box_arr, spf_arr, frm_curr_indx, ans_reso_indx, jumping_frm_number, cur_absolute_speed):
        
        
        resize_wh = resoStrLst[ans_reso_indx]  # resize width x height string
        
        resize_w = int(resize_wh.split("x")[0])
        resize_h = int(resize_wh.split("x")[1])
        
        boundingbox = speaker_box_arr[ans_reso_indx][frm_curr_indx] 

        next_frm_bounding_box = boundingbox
        start_frm_indx = frm_curr_indx
        
        average_speed = [s for s in cur_absolute_speed]  #  unit is in frame, not second [s*(1/PLAYOUT_RATE) for s in cur_absolute_speed]
        
        gt_start_box = speaker_box_arr[0][frm_curr_indx] 

        acc_accumulate = bb_intersection_over_union(gt_start_box, boundingbox)  # 0
        time_accumualate = spf_arr[ans_reso_indx][frm_curr_indx]   # initialized
        calculated_frm_num = 1

        start_time = time.time()
        while (start_frm_indx <= frm_curr_indx + jumping_frm_number):
            
            if start_frm_indx >= speaker_box_arr.shape[1]:
                break
            #img_path = img_path_lst[start_frm_indx]
            #img_path = self.get_image_name_id(frm_curr_indx)
            #img_name = img_path.split('/')[-1]
           
            #print ("resize img: ", img.shape)
            start_frm_indx += 1

            next_frm_bounding_box = [next_frm_bounding_box[0] + average_speed[1], next_frm_bounding_box[1] + average_speed[0], 
                                     next_frm_bounding_box[2] + average_speed[1], next_frm_bounding_box[3] + average_speed[0]]
           
            acc_accumulate += bb_intersection_over_union(speaker_box_arr[0][frm_curr_indx] , next_frm_bounding_box)
            #print ("bbbbbbacc_accumulate: ", acc_accumulate)
            calculated_frm_num += 1
            
            
        end_time = time.time()
        time_accumualate += end_time - start_time
        
        #print ("acc_accumulate: ", acc_accumulate, jumping_frm_number)
        
        
        return acc_accumulate, time_accumualate, calculated_frm_num


    def show_speaker_bounding_box_detection_result(self, predicted_video_dir, min_acc, imagePathLst, write_detection_box_flag):
        #detect speaker after applying the predictive model for configuration adaptation
        #not real apply to video; use profiling results to simulate 
        # get bounding box of the test video in different resolutions (i.e profiling result of the video)
        # then draw on the frame when doing video analytics
        
        if write_detection_box_flag:
            img_path = imagePathLst[0]   # get a random image path
            img_out_parent_dir = '/'.join(img_path.split('/')[:-2]) + "/" + img_path.split('/')[-2] + "_boundingbox_out/"
            
            if not os.path.exists(img_out_parent_dir):
                os.mkdir(img_out_parent_dir)
        
        
        speaker_box_arr, acc_arr, spf_arr = get_data_numpy(input_dir + predicted_video_dir)

        output_pickle_dir = input_dir + predicted_video_dir + "data_instance_xy/minAcc_" + str(min_acc) + "/"
        
        
        model_file = output_pickle_dir + "model_regression.joblib" + "_exclusive_" + str(predicted_video_dir[:-1])  + ".pkl"       # with other videos
                
        # read the model 
        #model_dir = input_dir + "all_data_instance_xy/minAcc_" + str(min_acc) +"/"
        #model_file = model_dir + "model_regression.joblib_all_videos.pkl"    
        
        
        #print ("speaker_box_arr:", speaker_box_arr[0])
        
        test_x_instances_file = output_pickle_dir + "trained_x_instances.pkl"
        X = read_pickle_data(test_x_instances_file)

        pca = PCA(n_components=min(n_components, X.shape[1])).fit(X)
        ModelRegressionObj = ModelRegression()
        model = joblib.load(model_file)


        reso_no = spf_arr.shape[0]
        frm_no = spf_arr.shape[1] #   min(len(imagePathLst), spf_arr.shape[1])
        
        acc_accumulate = 0.0
        time_spf_accumulate = 0.0
        calculated_frm_num = 0
        
        ans_reso_indx = 0
        k = 25     # 
        frm_lst_indx = 0
        # draw first segment use default resolution and index
        if write_detection_box_flag:
            self.draw_boundingbox_write(imagePathLst, img_out_parent_dir, speaker_box_arr, frm_lst_indx, ans_reso_indx, k, [0, 0])
        
        acc_tmp, time_spf_tmp, calc_frm_num_tmp  = self.get_online_video_analytics_accuracy_spf(speaker_box_arr, spf_arr, frm_lst_indx, ans_reso_indx, k, [0, 0])
        
        acc_accumulate += acc_tmp
        time_spf_accumulate += time_spf_tmp
        calculated_frm_num += calc_frm_num_tmp
        
        frm_curr_indx = k          # traverse from frame kth (index starts at 0)
        ema_absolute_curr = 0.0
        
        # initially used resolution
        last_frm_box = speaker_box_arr[ans_reso_indx][frm_lst_indx]   # initialize
          
        # use predicted result to apply to this new video and get delay and accuracy
        #delay_arr = []      
        #up_to_delay = max(0, spf_arr[ans_reso_indx][0] - 1.0/PLAYOUT_RATE)         # up to current delay, processing time - streaming time
        
        
        while(frm_curr_indx < frm_no):
            
            # calculate the relative feature
            
            curre_frm_box = speaker_box_arr[ans_reso_indx][frm_curr_indx]
            
            cur_absolute_speed = get_absolute_speed(last_frm_box, frm_lst_indx, curre_frm_box, frm_curr_indx)
             
            ema_absolute_curr = get_ema_absolute_speed(ema_absolute_curr, cur_absolute_speed)
             
            normalized_object_ratio = get_object_size(curre_frm_box)
            #print ("normalized_object_ratio: " , normalized_object_ratio)
    
            feature_x = np.hstack((ema_absolute_curr, normalized_object_ratio))
            #print ("feature_x cur_absolute_speed: " , cur_absolute_speed)
            
            
            
            #jumping_frm_number, current_aver_acc = estimate_frame_rate(speaker_box_arr, acc_arr, frm_curr_indx, ans_reso_indx, max_jump_number, min_acc_threshold)
            #ans_reso_indx, average_acc = get_reso_selected(speaker_box_arr, current_aver_acc, frm_curr_indx, jumping_frm_number, min_acc_threshold)
            
            #print ("jumping_frm_number:feature_x,  ", jumping_frm_number, feature_x, ans_reso_indx)
            
            predicted_y = ModelRegressionObj.test_on_data_y_unknown(model, feature_x, pca)
            
            #print ("get_prediction_y: ", predicted_y)
            
            
            jumping_frm_number = int(predicted_y[0][0])
            
            ans_reso_indx = int(predicted_y[0][1])
            
            
            # get delay up to this segment
            #up_to_delay = max(0, spf_arr[ans_reso_indx][frm_curr_indx] - (1.0/PLAYOUT_RATE) * jumping_frm_number)
            #delay_arr.append(up_to_delay)
            
            #print ("jumping_frm_number: ", jumping_frm_number, ans_reso_indx, frm_curr_indx)

            #draw bounding box on the image and output frm image with bounding box
            if write_detection_box_flag:
                self.draw_boundingbox_write(imagePathLst, img_out_parent_dir, speaker_box_arr, frm_curr_indx, ans_reso_indx, jumping_frm_number, cur_absolute_speed)
            
            acc_tmp, time_spf_tmp, calc_frm_num_tmp = self.get_online_video_analytics_accuracy_spf(speaker_box_arr, spf_arr,  frm_curr_indx, ans_reso_indx, jumping_frm_number, cur_absolute_speed)
            acc_accumulate += acc_tmp
            time_spf_accumulate += time_spf_tmp
            calculated_frm_num += calc_frm_num_tmp
            #curr_detect_time += spf_arr[ans_reso_indx][frm_curr_indx]
    
            last_frm_box = curre_frm_box
            
            frm_curr_indx += jumping_frm_number  #
            #EMA_absolute_prev = EMA_absolute_curr   # update EMA
        
        acc_average = acc_accumulate/(calculated_frm_num)
        time_spf = time_spf_accumulate/calculated_frm_num
     
        print ("acc_average, time_spf: ", acc_average, time_spf)
        
        return acc_average, time_spf



    def show_speaker_bounding_box_detection_result_each_second(self, predicted_video_dir, min_acc, imagePathLst, write_detection_box_flag):
        #detect speaker after applying the predictive model for configuration adaptation
        #not real apply to video; use profiling results to simulate 
        # get bounding box of the test video in different resolutions (i.e profiling result of the video)
        # then draw on the frame when doing video analytics
        
        if write_detection_box_flag:
            img_path = imagePathLst[0]   # get a random image path
            img_out_parent_dir = '/'.join(img_path.split('/')[:-2]) + "/" + img_path.split('/')[-2] + "_boundingbox_out/"
            
            if not os.path.exists(img_out_parent_dir):
                os.mkdir(img_out_parent_dir)
        
        
        speaker_box_arr, acc_arr, spf_arr = get_data_numpy(input_dir + predicted_video_dir)

        output_pickle_dir = input_dir + predicted_video_dir + "data_instance_xy/minAcc_" + str(min_acc) + "/"
        
        
        model_file = output_pickle_dir + "model_regression.joblib" + "_exclusive_" + str(predicted_video_dir[:-1])  + ".pkl"       # with other videos
                
        # read the model 
        #model_dir = input_dir + "all_data_instance_xy/minAcc_" + str(min_acc) +"/"
        #model_file = model_dir + "model_regression.joblib_all_videos.pkl"    
        
        
        #print ("speaker_box_arr:", speaker_box_arr[0])
        
        test_x_instances_file = output_pickle_dir + "trained_x_instances.pkl"
        X = read_pickle_data(test_x_instances_file)

        pca = PCA(n_components=min(n_components, X.shape[1])).fit(X)
        ModelRegressionObj = ModelRegression()
        model = joblib.load(model_file)


        reso_no = spf_arr.shape[0]
        frm_no = spf_arr.shape[1] #   min(len(imagePathLst), spf_arr.shape[1])
        
        acc_accumulate = 0.0
        time_spf_accumulate = 0.0
        calculated_frm_num = 0
        
        ans_reso_indx = 0
        k = 25     # 
        frm_lst_indx = 0
        # draw first segment use default resolution and index
        if write_detection_box_flag:
            self.draw_boundingbox_write(imagePathLst, img_out_parent_dir, speaker_box_arr, frm_lst_indx, ans_reso_indx, k, [0, 0])
        
        acc_tmp, time_spf_tmp, calc_frm_num_tmp  = self.get_online_video_analytics_accuracy_spf(speaker_box_arr, spf_arr, frm_lst_indx, ans_reso_indx, k, [0, 0])
        
        acc_accumulate += acc_tmp
        time_spf_accumulate += time_spf_tmp
        calculated_frm_num += calc_frm_num_tmp
        
        frm_curr_indx = k          # traverse from frame kth (index starts at 0)
        ema_absolute_curr = 0.0
        
        # initially used resolution
        last_frm_box = speaker_box_arr[ans_reso_indx][frm_lst_indx]   # initialize
          
        # use predicted result to apply to this new video and get delay and accuracy
        #delay_arr = []      
        #up_to_delay = max(0, spf_arr[ans_reso_indx][0] - 1.0/PLAYOUT_RATE)         # up to current delay, processing time - streaming time
        
        
        while(frm_curr_indx < frm_no):
            
            # calculate the relative feature
            
            curre_frm_box = speaker_box_arr[ans_reso_indx][frm_curr_indx]
            
            cur_absolute_speed = get_absolute_speed(last_frm_box, frm_lst_indx, curre_frm_box, frm_curr_indx)
             
            ema_absolute_curr = get_ema_absolute_speed(ema_absolute_curr, cur_absolute_speed)
             
            normalized_object_ratio = get_object_size(curre_frm_box)
            #print ("normalized_object_ratio: " , normalized_object_ratio)
    
            feature_x = np.hstack((ema_absolute_curr, normalized_object_ratio))
            #print ("feature_x cur_absolute_speed: " , cur_absolute_speed)
            
            
            
            #jumping_frm_number, current_aver_acc = estimate_frame_rate(speaker_box_arr, acc_arr, frm_curr_indx, ans_reso_indx, max_jump_number, min_acc_threshold)
            #ans_reso_indx, average_acc = get_reso_selected(speaker_box_arr, current_aver_acc, frm_curr_indx, jumping_frm_number, min_acc_threshold)
            
            #print ("jumping_frm_number:feature_x,  ", jumping_frm_number, feature_x, ans_reso_indx)
            
            predicted_y = ModelRegressionObj.test_on_data_y_unknown(model, feature_x, pca)
            
            #print ("get_prediction_y: ", predicted_y)
            
            
            jumping_frm_number = int(predicted_y[0][0])
            
            ans_reso_indx = int(predicted_y[0][1])
            
            
            # get delay up to this segment
            #up_to_delay = max(0, spf_arr[ans_reso_indx][frm_curr_indx] - (1.0/PLAYOUT_RATE) * jumping_frm_number)
            #delay_arr.append(up_to_delay)
            
            #print ("jumping_frm_number: ", jumping_frm_number, ans_reso_indx, frm_curr_indx)

            #draw bounding box on the image and output frm image with bounding box
            if write_detection_box_flag:
                self.draw_boundingbox_write(imagePathLst, img_out_parent_dir, speaker_box_arr, frm_curr_indx, ans_reso_indx, jumping_frm_number, cur_absolute_speed)
            
            acc_tmp, time_spf_tmp, calc_frm_num_tmp = self.get_online_video_analytics_accuracy_spf(speaker_box_arr, spf_arr,  frm_curr_indx, ans_reso_indx, jumping_frm_number, cur_absolute_speed)
            
            acc_accumulate += acc_tmp
            time_spf_accumulate += time_spf_tmp
            calculated_frm_num += calc_frm_num_tmp
            #curr_detect_time += spf_arr[ans_reso_indx][frm_curr_indx]
    
            last_frm_box = curre_frm_box
            
            frm_curr_indx += jumping_frm_number  #
            #EMA_absolute_prev = EMA_absolute_curr   # update EMA
        
        acc_average = acc_accumulate/(calculated_frm_num)
        time_spf = time_spf_accumulate/calculated_frm_num
     
        print ("acc_average, time_spf: ", acc_average, time_spf)
        
        return acc_average, time_spf


    def get_acc_spf_under_different_minThreshold(self):
        min_acc_threshold_lst = [0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
        acc_lst = []
        SPF_spent_lst = []
            
        for min_acc_thres in min_acc_threshold_lst:
                
            acc_average = 0.0
            spf_average = 0.0
            analyzed_video_lst = file_dir_lst[4:5]
            for predicted_video_dir in analyzed_video_lst:
                
                predicted_video_frm_dir = "/".join(input_dir.split("/")[:-2]) + "/" + "_".join(predicted_video_dir.split("_")[:-1]) + "_frames/"
                
                print ("predicted_video_frm_dir: ", predicted_video_frm_dir)  # ../input_output/speaker_video_dataset/sample_03_frames/
            
                imagePathLst = videoApplyObj.read_video_frm(predicted_video_frm_dir)
                write_detection_box_flag = True  # False
                acc, spf = videoApplyObj.show_speaker_bounding_box_detection_result(predicted_video_dir, min_acc_thres, imagePathLst, write_detection_box_flag)
    
                xx
                acc_average += acc
                spf_average += spf
            
            acc_lst.append(acc_average/len(analyzed_video_lst))
                
            SPF_spent_lst.append(spf_average/len(analyzed_video_lst))
                
        print("acc_lst, SPF_spent_lst: ", acc_lst, SPF_spent_lst)
        
        
        
        
if __name__=="__main__":
    
    videoApplyObj = VideoApply()
    
    #for predicted_video_dir in file_dir_lst[0:1]:
        
    #    videoApplyObj.get_prediction_acc_delay(predicted_video_dir, min_acc_threshold)
        
    
    videoApplyObj.get_acc_spf_under_different_minThreshold()
