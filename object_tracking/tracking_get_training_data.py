#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:53:21 2021

@author: fubao
"""

import os
import cv2
import glob
import json
import numpy as np
from blist import blist

from skimage import io



from collections import defaultdict



from tracking_data_preprocess import write_pickle_data
from tracking_data_preprocess import read_json_dir
from tracking_data_preprocess import bb_intersection_over_union

# get feature of car tracking

PLAYOUT_RATE = 25

ALPHA = 0.8

#data_dir = "../input_output/vehicle_tracking/sample_video_out/"


home_dir = "/home/fubao/workDir/researchProject/videoAnalytics_objectTracking/object_tracking/"
data_dir = "../input_output/object_tracking/sample_video_out/"



reso_list = ["1120x832", "960x720", "640x480",  "480x352", "320x240"] 

max_jump_number = 25            # float('inf')  # 10  # float('inf')  # 25
min_acc_threshold = 0.92     # 0.92, 0.94
interval_frm = 5


object_name = 'object' #  'object' # 'car'
#same_car_threshold = 0.7     # overlapping threshold
 
# here the YOLO4 tracking model output bounding box [x1,y1, x2, y2] => [topLeft_x, topLeft_y, bottomRight_x, bottomRighty]

optical_flow_object_size = 100
#optical_flow_object_size_flag = False

class DataGenerate(object):
    # generate features and target configuration
    
    def __init__(self, data_dir):
        self.data_dir = data_dir  # "../input_output/vehicle_tracking/sample_video_out/"
        self.video_dir_lst = self.read_json_list(self.data_dir)

    
    def read_json_list(self, data_dir):
        # read all the json file in teh directory
        #data_dir = "../input_output/vehicle_tracking/sample_video_out/"
        
        video_dir_lst = sorted(glob.glob(home_dir + data_dir + '*json*'))
        #print("data_dir: ", data_dir, video_dir_lst)
        
        return video_dir_lst
    
    
    def get_data_numpy(file_dir):
        # read pickle data
        # output: most expensive configuration's  numpy  (13390, 17, 3) (13390,) (13365,)
        
        speaker_box_file = file_dir +  "single_speaker_box.npy"
        spf_file =  file_dir +  "single_spf.npy"
        speaker_box_arr = read_numpy(speaker_box_file)   # (5, 37977, 4) 
        spf_arr = read_numpy(spf_file)       # (5, 37977)
        
        acc_file = file_dir + "single_acc.npy"
        
        acc_arr = read_numpy(acc_file)  
        #print("get_data_numpy speaker_box_arr number: ", speaker_box_arr.shape, spf_arr.shape, acc_arr.shape)
        # start from 0, 1 frames
        
        return speaker_box_arr, acc_arr, spf_arr


    def get_movement_feature(self, dict_detection_reso_video, current_frm_indx, last_frm_indx, frm_reso, arr_ema_absolute_velocity):
        # calculate object movement absolute velocity feature
        
        
        #print ("current_frm_indx: ", current_frm_indx, last_frm_indx)
        dict_current_frm_cars_positions =  dict_detection_reso_video[frm_reso][str(current_frm_indx)]
        
        dict_last_frm_cars_positions = dict_detection_reso_video[frm_reso][str(last_frm_indx)]
        
        #print ("dict_last_frm_cars_positions: ", dict_last_frm_cars_positions)
        #print ("dict_current_frm_cars_positions: ", dict_current_frm_cars_positions)
        
        # get all cars information
        dict_cars_current_frm = dict_current_frm_cars_positions[object_name]
        dict_cars_last_frm = dict_last_frm_cars_positions[object_name]

        abso_velocity_feature_x = self.calculate_velocity_traffic(dict_cars_current_frm, dict_cars_last_frm, current_frm_indx, last_frm_indx, frm_reso, arr_ema_absolute_velocity)
        
        return abso_velocity_feature_x


    def normalize_bounding_box(self, resolution, box):
    
        # bounding_box: [x1, y1, x2, y2]  vertical height is x, horiztonal width is y
        width = int(resolution.split("x")[0])
        height = int(resolution.split("x")[1])
        #print("normalize_bounding_box width, height:", width, height, box)
        
        yA = box[0]/width          # width
        xA = box[1]/height         # height
        yB = box[2]/width
        xB = box[3]/height
        
        return [yA, xA, yB, xB]


    def absolute_velocity_from_one_vechicle(self, current_frm_bounding_box_A, last_frm_bounding_box_B, frm_interval_len, frm_reso):
        # bounding_box_A and bounding_box_B velocity
        # get four angle's coordinate
        
        arr_curr_frm_pos = None  # np.zeros((4, 2))
        
        if len(current_frm_bounding_box_A) != 0:
            #current_frm_bounding_box_A = (frm_reso, current_frm_bounding_box_A)
            
            #print("current_frm_bounding_box_A :", current_frm_bounding_box_A)
        
            curr_frm_topLeft = [current_frm_bounding_box_A[0], current_frm_bounding_box_A[1]]
            curr_frm_topRight = [current_frm_bounding_box_A[2], current_frm_bounding_box_A[1]]
            curr_frm_bottomLeft = [current_frm_bounding_box_A[0], current_frm_bounding_box_A[3]]
            curr_frm_bottomRight = [current_frm_bounding_box_A[2], current_frm_bounding_box_A[3]]
        
            arr_curr_frm_pos = np.asarray([curr_frm_topLeft, curr_frm_topRight, curr_frm_bottomLeft, curr_frm_bottomRight])
        
        arr_last_frm_pos = None  # np.zeros((4, 2))
        if len(last_frm_bounding_box_B) != 0:
            
            #last_frm_bounding_box_B = (frm_reso, last_frm_bounding_box_B)

            last_frm_topLeft = [last_frm_bounding_box_B[0], last_frm_bounding_box_B[1]]
            last_frm_topRight = [last_frm_bounding_box_B[2], last_frm_bounding_box_B[1]]
            last_frm_bottomLeft = [last_frm_bounding_box_B[0], last_frm_bounding_box_B[3]]
            last_frm_bottomRight = [last_frm_bounding_box_B[2], last_frm_bounding_box_B[3]]
            
            arr_last_frm_pos = np.asarray([last_frm_topLeft, last_frm_topRight, last_frm_bottomLeft, last_frm_bottomRight])
            
        #print ("arr_curr_frm_pos.shape: ", arr_curr_frm_pos.shape, arr_last_frm_pos)
        
        arr_abso_velocity_feature = np.zeros((4, 2))
        if arr_curr_frm_pos is not None and arr_last_frm_pos is not None:
            arr_abso_velocity_feature = (arr_curr_frm_pos -  arr_last_frm_pos)/frm_interval_len
        
        #print ("22222 arr_abso_velocity_feature.shape: ", arr_abso_velocity_feature.shape, arr_abso_velocity_feature)
        arr_abso_velocity_feature = np.abs(arr_abso_velocity_feature)
        
        return arr_abso_velocity_feature
    

    def draw_bounding_box(self, img, bbox):
               
        #img = cv2.resize(img, (416, 416))      # no need to resize
        
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        return img
    
        """
        def same_car_boundingbox(self, boundingbox_A, boundingbox_B, same_car_threshold):
            # same resolution's boundingbox
            IOU = bb_intersection_over_union(boundingbox_A, boundingbox_B)
            
            if IOU >= same_car_threshold:
                return True
            return False
        """
        

    def calculate_velocity_traffic(self, dict_cars_current_frm, dict_cars_last_frm, current_frm_indx, last_frm_indx, frm_reso, arr_ema_absolute_velocity):
        # calculate the traffic's velocity
        
        #print("dict_cars_current_frm, dict_cars_last_frm: ", dict_cars_current_frm,  dict_cars_last_frm)
        #keys_cars_current = dict_cars_current_frm.keys()
        
        #aver_velocity_speed = 0.0    # all car average velocity
        frm_interval_len = current_frm_indx - last_frm_indx
        
        #frm_file_path = "/media/fubao/TOSHIBAEXT/research_bakup/data_video_analytics/input_output/vechicle_tracking/car_traffic_01_1120x832_frames/000050.jpg"
        #img = cv2.imread(frm_file_path)
        arr_abso_velocity_average_feature = np.zeros((4, 2))
        vechicle_count = 0
        """
        for key_cur_a, current_frm_bounding_box_A in dict_cars_current_frm.items():
            
            #print("key_cur_a: ", current_frm_bounding_box_A, len(dict_cars_last_frm), current_frm_indx, last_frm_indx)
            
            arr_abso_velocity_one_vechicle_feature = np.zeros((4, 2))
            
            #if key_cur_a in dict_cars_last_frm:
                #last_frm_bounding_box_B = dict_cars_last_frm[key_cur_a]

                # calculate the velocity
            for key_cur_b, last_frm_bounding_box_B in dict_cars_last_frm.items():
                
                #print("key_cur_b: ", last_frm_bounding_box_B)

                if self.same_car_boundingbox(current_frm_bounding_box_A, last_frm_bounding_box_B, same_car_threshold):
                
                    arr_abso_velocity_one_vechicle_feature = self.absolute_velocity_from_one_vechicle(current_frm_bounding_box_A, last_frm_bounding_box_B, frm_interval_len, frm_reso)
                    
                    arr_abso_velocity_average_feature += np.abs(arr_abso_velocity_one_vechicle_feature)
                
                    vechicle_count += 1
                
                    print ("3333333 arr_abso_velocity_average_feature: ", arr_abso_velocity_average_feature, current_frm_indx, last_frm_indx)
        """
        
        for key_cur_a, current_frm_bounding_box_A in dict_cars_current_frm.items():
            
            #print("key_cur_a: ", current_frm_bounding_box_A, len(dict_cars_last_frm), current_frm_indx, last_frm_indx)
            
            arr_abso_velocity_one_vechicle_feature = np.zeros((4, 2))
            
            if key_cur_a in dict_cars_last_frm:
                last_frm_bounding_box_B = dict_cars_last_frm[key_cur_a]

                # calculate the velocity
                
                #print("key_cur_b: ", last_frm_bounding_box_B)
                
                arr_abso_velocity_one_vechicle_feature = self.absolute_velocity_from_one_vechicle(current_frm_bounding_box_A, last_frm_bounding_box_B, frm_interval_len, frm_reso)
                
                arr_abso_velocity_average_feature += np.abs(arr_abso_velocity_one_vechicle_feature)
            
                vechicle_count += 1
            
                #print ("444444 arr_abso_velocity_average_feature: ", arr_abso_velocity_average_feature, current_frm_indx, last_frm_indx)
    
        if vechicle_count != 0:
            arr_abso_velocity_average_feature /= vechicle_count
        
        #print ("arr_abso_velocity_average_feature: ", arr_abso_velocity_average_feature)
        
        #frm_file_out_path = "/media/fubao/TOSHIBAEXT/research_bakup/data_video_analytics/input_output/vechicle_tracking/car_traffic_01_1120x832_frames/000050_box_out.jpg"
        #cv2.imwrite(frm_file_out_path, img)      
        
        arr_ema_absolute_velocity = self.get_ema_absolute_velocity(arr_ema_absolute_velocity, arr_abso_velocity_average_feature)
         
        
        return arr_ema_absolute_velocity
    
    
    
    def get_ema_absolute_velocity(self, EMA_pre, cur_absolute_velocity):
        # get the EMA speed
        ema_velocity =  ALPHA * cur_absolute_velocity  + (1-ALPHA) * EMA_pre
        #print("get_ema_absolute_speed ema_speed: ", ema_velocity)
        return ema_velocity


    def get_numpy_arr_all_jumpingFrameInterval_resolution(self, dict_detection_reso_video, current_frm_indx, max_frm_interval, highest_reso, curre_reso):
        
        jumping_frame_interval = 0
        arr_acc_interval = []   # np.zeros(jumping_frame_interval)       # the maximum 25 frame interval's acc in an array
        while(jumping_frame_interval < max_frm_interval):
            next_frm_indx = current_frm_indx + jumping_frame_interval
            dict_cars_each_jumped_frm_higest_reso =  dict_detection_reso_video[highest_reso][str(next_frm_indx)][object_name]
            
            # use current_frm_indx result to pass later video analytics result
            dict_cars_each_jumped_frm_curr_reso = dict_detection_reso_video[curre_reso][str(current_frm_indx)][object_name]
            
            #dict_cars_each_jumped_frm_other_reso = dict_detection_reso_video[curr_reso][str(next_frm_indx)][object_name]
            #print("dict_cars_each_jumped_frm_higest_reso: ", dict_cars_each_jumped_frm_higest_reso)
            #print("predict_next_configuration_jumping_frm_reso: xxx", dict_cars_each_jumped_frm_higest_reso)
            curr_acc = min_acc_threshold   # 0.0  # # no vechicle existing in the frame
            if dict_cars_each_jumped_frm_higest_reso is not None:
                #print("predict_next_configuration_jumping_frm_reso dict_cars_each_jumped_frm_higest_reso: ", dict_cars_each_jumped_frm_higest_reso)
                curr_acc = self.calculate_accuray_with_highest_reso(dict_cars_each_jumped_frm_higest_reso, highest_reso, dict_cars_each_jumped_frm_curr_reso, curre_reso, min_acc_threshold)
                                    
                #print ("dict_cars_each_jumped_frm_higest_reso curr_acc: ",  curr_acc)
            
            arr_acc_interval.append(curr_acc)
            jumping_frame_interval += 1
            
            # print("arr_acc_interval: ", arr_acc_interval)

        #if len(arr_acc_interval) == 0:
        #    print("xxxxxxxxxxxxxxxxxx empty: ", arr_acc_interval)
        
        arr_acc_interval = np.asarray(arr_acc_interval)
        
        return arr_acc_interval
    
    
    def predict_next_configuration_jumping_frm_reso(self, dict_detection_reso_video, min_acc_threshold, current_frm_indx, FRM_NO):
        
        # predict the next frm and reso  in greedy heuristic method or exhaustic search?
        
        # start from highest resolution 1120x832p
        highest_reso = reso_list[0]
        
        #print ("predict_next_configuration_jumping_frm_reso current_frm_indx: ", current_frm_indx)
        #dict_cars_each_jumped_frm_higest_reso =  dict_detection_reso_video[highest_reso][str(current_frm_indx)][object_name]
                
        #print ("dict_last_frm_cars_positions: ", dict_last_frm_cars_positions)
        #print ("dict_current_frm_cars_positions: ", dict_current_frm_cars_positions)
        
        # try from jumping_frame_interval = 25 to 0 store into a numpy of each frame's array
        max_frm_interval = min(FRM_NO-current_frm_indx, 25)
        arr_acc_interval = self.get_numpy_arr_all_jumpingFrameInterval_resolution(dict_detection_reso_video, current_frm_indx, max_frm_interval, highest_reso, highest_reso)
        
        # print ("arr_acc_intervalllllllllllllll: ", current_frm_indx, arr_acc_interval.shape, arr_acc_interval)
        
        #try each frame interval and find the interval with min_acc_threshold
        
        ans_jfr = 1  # the answer of jumping_frame_interval
        jumping_frame_interval = arr_acc_interval.shape[0]    # 25 
        while(jumping_frame_interval > 0):
            
            aver_acc_curr_interval = np.mean(arr_acc_interval[0:jumping_frame_interval])
            
            if aver_acc_curr_interval >= min_acc_threshold:   # find the jumping frame interval
                ans_jfr = jumping_frame_interval
                break

            jumping_frame_interval -= 1
        
        # then find the resolution under this jumping frame interval
        #reso_list = ["1120x832", "960x720", "640x480",  "480x352", "320x240"] 
        # start from the lowest frame rate and this jumping frame interval
        #print ("ans_jfr: ", ans_jfr)
        
        ans_reso_indx = len(reso_list) - 1
        aver_acc = np.mean(arr_acc_interval[0:ans_jfr+1])
        
        #print ("ans_jfr : ", ans_jfr, ans_reso_indx, arr_acc_interval)
        for i, lower_reso in enumerate(reso_list[::-1]):
            arr_acc_interval_curr_jfr = self.get_numpy_arr_all_jumpingFrameInterval_resolution(dict_detection_reso_video, current_frm_indx, ans_jfr, highest_reso, lower_reso)
            
            #print ("arr_acc_interval_curr_jfrrrrrrrr: ", arr_acc_interval_curr_jfr)
            if np.mean(arr_acc_interval_curr_jfr[0:ans_jfr]) >= min_acc_threshold:
                ans_reso_indx = len(reso_list) - 1 - i
                aver_acc = np.mean(arr_acc_interval_curr_jfr[0:ans_jfr])
                break
        
        # print ("current_frm_indx, ans_jfr, ans_reso_indx, aver_acc: ", current_frm_indx, ans_jfr, ans_reso_indx, aver_acc)
        
        if aver_acc == 0.0:  # no vechile existing. no need to use highest resolution
            aver_acc = min_acc_threshold
            ans_reso_indx = 0
            
        return ans_jfr, ans_reso_indx, aver_acc
    


    def calculate_accuray_with_highest_reso(self, dict_cars_each_jumped_frm_higest_reso, highest_reso, dict_cars_each_jumped_frm_other_reso, curr_reso, min_acc_threshold):
        # treat the highest resolution detection result as ground truth to detect 
        
        # We dont know each car in highest resolution frame correspoinding to each car in lower resolution frame; naive way is to use two loops
        # detect if the bounding box overlapping >= 0.5, think it maybe detecting the same car?
        # print ("high_reso_bounding_box: ", dict_cars_each_jumped_frm_other_reso.values())
        same_car_IOU_threshold = 0.9
        aver_acc = 0.0
        count = 0.0
        for high_reso_bounding_box in dict_cars_each_jumped_frm_higest_reso.values():
            for other_reso_bounding_box in dict_cars_each_jumped_frm_other_reso.values():
                #print ("high_reso_bounding_box: ", high_reso_bounding_box, other_reso_bounding_box)
                if len(high_reso_bounding_box) == 0 or len(other_reso_bounding_box) == 0:
                    continue
                    
                high_reso_bounding_box = self.normalize_bounding_box(highest_reso, high_reso_bounding_box)
                other_reso_bounding_box = self.normalize_bounding_box(curr_reso, other_reso_bounding_box)

                IOU = bb_intersection_over_union(high_reso_bounding_box, other_reso_bounding_box)
                #print ("IOU :" , IOU)
                if IOU >= same_car_IOU_threshold:
                    aver_acc += IOU
                    count += 1
        if count != 0:
            aver_acc /= count
        
        #print ("aver_acc: ", aver_acc)
        if aver_acc == 0.0 and len(dict_cars_each_jumped_frm_higest_reso) != 0:   # no vechicle count
            aver_acc = 0.0
        elif aver_acc == 0.0:
            aver_acc = min_acc_threshold
        
            #print("cccccalculate_accuray_with_highest_reso: ", aver_acc)
        return aver_acc


    def get_object_size_x(self, dict_detection_reso_video, current_frm_indx, ans_reso_indx):
        # get object size 
        curre_reso = reso_list[ans_reso_indx]
        dict_cars_each_jumped_frm_curr_reso = dict_detection_reso_video[curre_reso][str(current_frm_indx)][object_name]

        #print ("dict_cars_each_jumped_frm_curr_reso: ", dict_cars_each_jumped_frm_curr_reso)
        
        curre_reso_int = int(curre_reso.split('x')[0]) * int(curre_reso.split('x')[1])
        normalized_object_size_ratio = 0
        
        aver_object_size = 0.0
        for car_id, car_pos in dict_cars_each_jumped_frm_curr_reso.items():
            
            #print("car_id: ", car_id, car_pos)
            
            area = (car_pos[2] - car_pos[0]) * (car_pos[3]-car_pos[1])
            
            normalized_object_size_ratio = area/curre_reso_int
            #print("normalized_object_size_ratio: ", normalized_object_size_ratio, area, curre_reso_int)
            
            aver_object_size += normalized_object_size_ratio
        
        #print("aver_object_size: ", aver_object_size)
        
        if len(dict_cars_each_jumped_frm_curr_reso) != 0:
            aver_object_size = aver_object_size/len(dict_cars_each_jumped_frm_curr_reso)
    
        return aver_object_size
    
    
    def get_object_size_change(self, dict_detection_reso_video, current_frm_indx, ans_reso_indx, last_frm_indx, last_reso_indx):
        
        curre_reso = reso_list[ans_reso_indx]
        dict_cars_each_jumped_frm_curr_reso = dict_detection_reso_video[curre_reso][str(current_frm_indx)][object_name]

        #print ("dict_cars_each_jumped_frm_curr_reso: ", dict_cars_each_jumped_frm_curr_reso)
        
        curre_reso_int = int(curre_reso.split('x')[0]) * int(curre_reso.split('x')[1])
        normalized_object_size_ratio = 0
        
        curr_aver_object_size = np.zeros(2)   # x_axis and y_axis
        
        for car_id, car_pos in dict_cars_each_jumped_frm_curr_reso.items():
            
            #print("car_id: ", car_id, car_pos)
            #area = (car_pos[2] - car_pos[0]) * (car_pos[3]-car_pos[1])
            
            size_arr = np.array([(car_pos[2] - car_pos[0]), (car_pos[3]-car_pos[1])])                       # [x1,y1, x2, y2]
            normalized_object_size_ratio = size_arr/curre_reso_int
            #print("normalized_object_size_ratio: ", normalized_object_size_ratio, area, curre_reso_int)
            
            curr_aver_object_size += normalized_object_size_ratio
            
            #print("curr_aver_object_size: ", curr_aver_object_size)
        
        if len(dict_cars_each_jumped_frm_curr_reso) != 0:
            curr_aver_object_size = curr_aver_object_size/len(dict_cars_each_jumped_frm_curr_reso)
    
    
        curre_reso = reso_list[last_reso_indx]
        dict_cars_each_jumped_frm_curr_reso = dict_detection_reso_video[curre_reso][str(last_frm_indx)][object_name]

        #print ("dict_cars_each_jumped_frm_curr_reso: ", dict_cars_each_jumped_frm_curr_reso)
        
        curre_reso_int = int(curre_reso.split('x')[0]) * int(curre_reso.split('x')[1])
        normalized_object_size_ratio = 0
        
        prev_aver_object_size = np.zeros(2)   # x_axis and y_axis
        for car_id, car_pos in dict_cars_each_jumped_frm_curr_reso.items():
            
            #print("car_id: ", car_id, car_pos)
            
            size_arr = np.array([(car_pos[2] - car_pos[0]), (car_pos[3]-car_pos[1])])                       # [x1,y1, x2, y2]
            normalized_object_size_ratio = size_arr/curre_reso_int
            #print("normalized_object_size_ratio: ", normalized_object_size_ratio, area, curre_reso_int)
            
            prev_aver_object_size += normalized_object_size_ratio
        
        #print("prev_aver_object_size: ", prev_aver_object_size)
        
        if len(dict_cars_each_jumped_frm_curr_reso) != 0:
            prev_aver_object_size = prev_aver_object_size/len(dict_cars_each_jumped_frm_curr_reso)
            
        return curr_aver_object_size - prev_aver_object_size
        
        
    def read_one_frame_arr(self, prev_frame_path, curr_frame_path, curre_reso):
        # frame to numpy arr
        
        #prev_frame_arr = cv2.imread(prev_frame_path)
        #current_frame_arr = cv2.imread(curr_frame_path)
       
        width = int(curre_reso.split('x')[0])
        height = int(curre_reso.split('x')[1])
        prev_frame_arr = io.imread(prev_frame_path)
        prev_frame_arr = cv2.resize(prev_frame_arr, (width, height))
        
        current_frame_arr = io.imread(curr_frame_path)
        current_frame_arr = cv2.resize(current_frame_arr, (width, height))
        #print("prev_frame_arr, current_frame_arr: ", prev_frame_arr.shape, current_frame_arr.shape)
        
        return prev_frame_arr, current_frame_arr
    
    
    
    def get_one_boundingbox(self, frame_arr, bounding_box):
        
        #print("bounding_box: ", frame_arr.shape, bounding_box)
        top_left_x = int(bounding_box[0])
        top_left_y = int(bounding_box[1])
        btm_rt_x = int(bounding_box[2])
        btm_rt_y = int(bounding_box[3])
        
        box_arr = frame_arr[top_left_y:btm_rt_y, top_left_x: btm_rt_x]
        
        #print("box_arr: ", box_arr.shape)
        
        return box_arr
        
    def calculate_optical_flow(self, prev_frame_arr, current_frame_arr, optical_flow_object_size):
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.3,
                       minDistance = 3,
                       blockSize = 3 )
        
        prev_gray = cv2.cvtColor(prev_frame_arr, cv2.COLOR_BGR2GRAY)
        prev_kps = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

        current_gray = cv2.cvtColor(current_frame_arr, cv2.COLOR_BGR2GRAY)
        optical_flow = cv2.calcOpticalFlowPyrLK(prev_gray,
                                                 current_gray,
                                                 prev_kps,
                                                 None)
        
        
        #print("none_optical_flow: ", type(optical_flow), optical_flow[0].shape, prev_kps.shape)
        
        optical_flow_next_pt = optical_flow[0].flatten()
        optical_flow_next_pt = np.resize(optical_flow_next_pt, (optical_flow_object_size, ))   # fixed size
        return optical_flow_next_pt
        
        
    def get_optical_flow_feature(self, video_frame_dir, dict_detection_reso_video, current_frm_indx, ans_reso_indx, last_frm_indx, last_reso_indx):
        
        prev_frame_path =  video_frame_dir + '/' + str(last_frm_indx).zfill(5) + '.jpg'
        
        curr_frame_path =  video_frame_dir + '/' + str(current_frm_indx).zfill(5) + '.jpg'
        #print("curr_frame_path: ", curr_frame_path)
        curre_reso = reso_list[ans_reso_indx]
        
        prev_frame_arr, current_frame_arr = self.read_one_frame_arr(prev_frame_path, curr_frame_path, curre_reso)
        
        dict_current_frm_cars_positions =  dict_detection_reso_video[curre_reso][str(current_frm_indx)]
        dict_last_frm_cars_positions = dict_detection_reso_video[curre_reso][str(last_frm_indx)]
        
        dict_cars_current_frm = dict_current_frm_cars_positions[object_name]
        dict_cars_last_frm = dict_last_frm_cars_positions[object_name]
        
        optical_flow_arr = self.calculate_optical_flow(prev_frame_arr, current_frame_arr, optical_flow_object_size)

        count = 1
        for key_cur_a, current_frm_bounding_box_A in dict_cars_current_frm.items():
            #global optical_flow_object_size_flag        # can not change it, otherwise the test dataset will have different size

            #if not optical_flow_object_size_flag:
                #global optical_flow_object_size
                #optical_flow_object_size = abs(current_frm_bounding_box_A[2] - current_frm_bounding_box_A[0]) * abs(current_frm_bounding_box_A[3] - current_frm_bounding_box_A[1])
                
                #global optical_flow_object_size_flag
                #optical_flow_object_size_flag = True
            #print("key_cur_a: ", current_frm_bounding_box_A, len(dict_cars_last_frm), current_frm_indx, last_frm_indx)
            
            prev_box_arr = self.get_one_boundingbox(current_frame_arr, current_frm_bounding_box_A)
            
            if key_cur_a in dict_cars_last_frm:
                last_frm_bounding_box_B = dict_cars_last_frm[key_cur_a]
                curr_box_arr = self.get_one_boundingbox(prev_frame_arr, last_frm_bounding_box_B)
                
                prev_size = prev_box_arr.size
                curr_size = curr_box_arr.size
                if prev_size >= curr_size:
                    curr_box_arr = np.resize(curr_box_arr, prev_box_arr.shape)
                else:
                    prev_box_arr = np.resize(prev_box_arr, curr_box_arr.shape)
                    
                #print("prev_box_arr: ", prev_box_arr.shape, curr_box_arr.shape)
                #if optical_flow_arr is None:
                #    optical_flow_arr += self.calculate_optical_flow(prev_box_arr, curr_box_arr)

                #else:
                try:
                    optical_flow_arr += self.calculate_optical_flow(prev_box_arr, curr_box_arr, optical_flow_object_size)
                except:
                    continue
                count += 1
        
        optical_flow_arr /= count
        
        
        #print("optical_flow_arr: ", optical_flow_arr.shape)
        return optical_flow_arr


    def get_data_instances(self, one_video_input_dir, max_jump_number = 5, min_acc_threshold = 0.95, speed_type = 'ema', interval_frm = 10):
        # read input_dir
        
        #video_id = one_video_input_dir.split('/')[-2].split('_')[-1]
        #print("one_video_input_dir :", one_video_input_dir)
        
        video_frame_dir =  one_video_input_dir   # +  '*_frames'  # 'car_traffic_' + str(video_id) + '_frames'
        
        for root, subdirs, files in os.walk(video_frame_dir):
            flag = False
            for frame_folder_name in subdirs:
                if '_frames' in frame_folder_name:
                    video_frame_dir = video_frame_dir + frame_folder_name
                    flag = True
                    break
            if flag:
                break
            
        #print("video_frame_dir :", video_frame_dir)

        dict_detection_reso_video = read_json_dir(one_video_input_dir)
        
        
        FRM_NO = len(dict_detection_reso_video[reso_list[0]])
        
        #print ("FRM_NO: ", one_video_input_dir,  FRM_NO)
        
        last_frm_indx = 1
        
        list_estimated_velocity_2_jumpinNumberResolution = blist()           # speed => jumping_frame_number and resolution
        
        segment_acc_arr = blist()
        
        ans_reso_indx = 0
        last_reso_indx = 0
        arr_ema_absolute_velocity = np.zeros((4, 2))
        
        if speed_type == 'ema':
            # Price(current)  x Multiplier)   + (1-Multiplier) * EMA(prev) 
            # Vi= ALPHA * (M_i) + (1-ALPHA)* V_{i-1}
            # current speed use previous current frame - previous frame as estimation now
            # start from 2nd frame
            current_frm_indx = interval_frm + 1              # 1  starting index = 0  left frame no = 1

            while(current_frm_indx < FRM_NO):
                
                # start from highest resolution
                #frm_reso = reso_list[0]
                frm_reso = reso_list[ans_reso_indx]
                
                
                arr_ema_absolute_velocity = self.get_movement_feature(dict_detection_reso_video, current_frm_indx, last_frm_indx, frm_reso, arr_ema_absolute_velocity)
                
                feature_vect_mean = np.mean(arr_ema_absolute_velocity, axis = 0)
                feature_vect_var = np.var(arr_ema_absolute_velocity, axis = 0)
                #print("arr_ema_speed: ", feature_vect_mean, feature_vect_var, np.asarray([feature_vect_mean, feature_vect_var]))
                # get the jumping number y
                
                arr_movement_feature = np.hstack((arr_ema_absolute_velocity.flatten(), feature_vect_mean, feature_vect_var))
                
                #print("abso_velocity_feature_x: ", arr_ema_absolute_velocity.shape)
                
                feature_x_object_size = self.get_object_size_change(dict_detection_reso_video, current_frm_indx, ans_reso_indx, last_frm_indx, last_reso_indx)      # self.get_object_size_x(dict_detection_reso_video, current_frm_indx, ans_reso_indx)      
                
                #optical_flow_next_pt = self.get_optical_flow_feature(video_frame_dir, dict_detection_reso_video, current_frm_indx, ans_reso_indx, last_frm_indx, last_reso_indx)
                
                try:
                    optical_flow_next_pt = self.get_optical_flow_feature(video_frame_dir, dict_detection_reso_video, current_frm_indx, ans_reso_indx, last_frm_indx, last_reso_indx)
                    
                except:
                    last_reso_indx = ans_reso_indx
                    ans_jfr, ans_reso_indx, aver_acc = self.predict_next_configuration_jumping_frm_reso(dict_detection_reso_video, min_acc_threshold, current_frm_indx, FRM_NO)
                    # predict how many frame jumped from this starting point
                    
                    #print("arr_ema_absolute_velocity ans_jfr ans_reso_indx aver_acc: ", arr_ema_absolute_velocity, ans_jfr, ans_reso_indx, aver_acc)
                    
                    #print("ans_jfr ans_jfr ans_reso_indx aver_acc: ", ans_jfr, ans_reso_indx, aver_acc)
                    if ans_jfr > max_jump_number:
                        ans_jfr = max_jump_number
                    
                    last_frm_indx = current_frm_indx
                    current_frm_indx += (ans_jfr+1)   # (ans_jfr+1)    # 1   # (ans_jfr+1)                 # 1 to use for generate training data
                    continue
                
                
                #print("optical_flow_next_pt: ", arr_movement_feature.shape, optical_flow_next_pt.shape, current_frm_indx, FRM_NO)
                #print("feature_x_object_size: ", arr_movement_feature.shape, feature_x_object_size.shape)
                feature_x = np.hstack((arr_movement_feature, feature_x_object_size, optical_flow_next_pt))
                    
                #print("feature_x: ", arr_movement_feature.shape, optical_flow_next_pt.shape, feature_x.shape, current_frm_indx, FRM_NO)
                                  
                last_reso_indx = ans_reso_indx
                ans_jfr, ans_reso_indx, aver_acc = self.predict_next_configuration_jumping_frm_reso(dict_detection_reso_video, min_acc_threshold, current_frm_indx, FRM_NO)
                # predict how many frame jumped from this starting point
                
                #print("arr_ema_absolute_velocity ans_jfr ans_reso_indx aver_acc: ", arr_ema_absolute_velocity, ans_jfr, ans_reso_indx, aver_acc)
                
                #print("ans_jfr ans_jfr ans_reso_indx aver_acc: ", ans_jfr, ans_reso_indx, aver_acc)
                if ans_jfr > max_jump_number:
                    ans_jfr = max_jump_number
                
                y_out = np.asarray([ans_jfr, ans_reso_indx])
                
                data_one_instance_jumpingNumberReso = np.append(feature_x, y_out)
                list_estimated_velocity_2_jumpinNumberResolution.append(data_one_instance_jumpingNumberReso)

                last_frm_indx = current_frm_indx
                current_frm_indx += (ans_jfr+1)   # (ans_jfr+1)    # 1   # (ans_jfr+1)                 # 1 to use for generate training data
                
                segment_acc_arr.append(aver_acc)   # (average_acc)


        segment_acc_arr = np.asarray(segment_acc_arr)
        
        arr_estimated_velocity_2_jumpingNumber_reso = np.asarray(list_estimated_velocity_2_jumpinNumberResolution)
        print("arr_estimated_velocity_2_jumpingNumber_reso: ", arr_estimated_velocity_2_jumpingNumber_reso.shape)
        return arr_estimated_velocity_2_jumpingNumber_reso, segment_acc_arr

    
    def getDataExamples_features(self):
    
        """
        #all_arr_estimated_speed_2_jump_number = blist()  # all video
        speed_type = 'ema'
        interval_frm = 50
        for i, video_dir in enumerate(self.video_dir_lst[0:1]):
            
            one_video_input_dir = data_dir + video_dir
            self.get_data_instances(one_video_input_dir, max_jump_number, min_acc_threshold, speed_type, interval_frm )
        """
        
        speed_type = 'ema'
        
        #all_arr_estimated_speed_2_jump_number = blist()  # all video
        all_arr_estimated_speed_2_reso = None
        for i, video_dir in enumerate(self.video_dir_lst[0:31]): # [0:31]): # [3:4]):    # [2:3]:   #[1:2]:  # [1:2]:  #[0:1]:        #[1:2]:
            
            one_video_input_dir = video_dir + '/'

            #print("config_est_frm_arr shape:  ", i, type(config_est_frm_arr), acc_frame_arr.shape, spf_frame_arr.shape)
            arr_estimated_velocity_2_jumpingNumber_reso, segment_acc_arr = self.get_data_instances(one_video_input_dir, max_jump_number, min_acc_threshold, speed_type, interval_frm)
            
            #print("all_arr_estimated_speed_2_reso: ", arr_estimated_velocity_2_jumpingNumber_reso.shape)
            
            
            if i == 0:
                all_arr_estimated_speed_2_reso = arr_estimated_velocity_2_jumpingNumber_reso
            else:
                all_arr_estimated_speed_2_reso = np.vstack((all_arr_estimated_speed_2_reso, arr_estimated_velocity_2_jumpingNumber_reso))
            
            print("eeeeeeall_arr_estimated_speed_2_jump_number: ", arr_estimated_velocity_2_jumpingNumber_reso.shape,  all_arr_estimated_speed_2_reso.shape)
            #out_pickle_dir =  data_dir + video_dir + "/jumping_number_result/" + "/resolution_selection/"
            
    
            sub_dir_1 = video_dir + "/jumping_number_result/"
                
            #out_pickle_dir =  data_dir + video_dir + "jumping_number_result_each_frm/" + "jumpingNumber_resolution_selection/"

            if not os.path.exists(sub_dir_1):
                os.mkdir(sub_dir_1)

            sub_dir_2 = sub_dir_1  + "jumpingNumber_resolution_selection/"


            if not os.path.exists(sub_dir_2):
                os.mkdir(sub_dir_2)
            
            out_pickle_dir = sub_dir_2 + "intervalFrm-" + str(interval_frm) + "_speedType-" + str(speed_type) + "_minAcc-" + str(min_acc_threshold) + "/"
            
            if not os.path.exists(out_pickle_dir):
                os.mkdir(out_pickle_dir)
                
            out_data_pickle_file = out_pickle_dir + "data_instance_speed_jumpingNumber_resolution_objectSizeRatio_xy.pkl" 
            write_pickle_data(arr_estimated_velocity_2_jumpingNumber_reso, out_data_pickle_file)


        write_out_dir_1 = data_dir + "jumping_number_result/"   # data_dir + "jumping_number_result_each_frm/"

        if not os.path.exists(write_out_dir_1):
            os.mkdir(write_out_dir_1)
                
       
        write_out_dir_2 = write_out_dir_1 + "dynamic_jumpingNumber_resolution_selection_output/"

        if not os.path.exists(write_out_dir_2):
            os.mkdir(write_out_dir_2)
        
        output_dir = write_out_dir_2 + "intervalFrm-" + str(interval_frm) + "_speedType-" + str(speed_type) + "_minAcc-" + str(min_acc_threshold) + "/"

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.write_all_instance_data(output_dir, all_arr_estimated_speed_2_reso)


    def write_all_instance_data(self, output_dir, all_arr_estimated_speed_2_jump_number):
        # write all the current video into the output
        out_data_pickle_file = output_dir + "all_data_instance_speed_JumpingNumber_resolution_objectSizeRatio_xy.pkl" 
        write_pickle_data(all_arr_estimated_speed_2_jump_number, out_data_pickle_file)       
        


if __name__== "__main__": 
    

    data_obj = DataGenerate(data_dir)
    
    #print("data_obj video_dir_lst : ", data_obj.video_dir_lst)
    
    data_obj.getDataExamples_features()
    
    

    
