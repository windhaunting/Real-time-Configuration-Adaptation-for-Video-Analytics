#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  24 19:35:10 2020

@author: fubao
"""

# get the relative features of person movement


import os
import numpy as np
from blist import blist
import glob
import cv2
from skimage import io

from data_preproc import read_numpy
from data_preproc import bb_intersection_over_union
from data_preproc import write_pickle_data

from data_preproc import input_dir
from data_preproc import resoStrLst
from data_preproc import PLAYOUT_RATE

ALPHA = 0.8
# use EMA EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
# = Multiplier * Price(current)  + (1-Multiplier) * EMA(prev) 

#  bounding_box: [y1, x1, y2, x2]  vertical height is y, horiztonal width is x


'''
file_dir_lst = ["sample_01_out/", "sample_02_out/",  \
                    "sample_03_out/", "sample_04_out/", \
                    "sample_05_out/", "sample_06_out/",  \
                    "sample_07_out/", "sample_08_out/", \
                    "sample_09_out/", "sample_10_out/", \
                    "sample_11_out/", "sample_12_out/",  \
                    "sample_13_out/", "sample_14_out/", \
                    "sample_15_out/", "sample_16_out/", \
                    "sample_17_out/", "sample_18_out/",  \
                    "sample_19_out/", "sample_20_out/", \
                    "sample_21_out/", "sample_22_out/", \
                    "sample_23_out/", "sample_24_out/",  \
                    "sample_25_out/", "sample_26_out/", \
                    "sample_27_out/", "sample_28_out/"]
'''

min_acc_threshold = 0.92        # 0.90
max_jump_number = 25
optical_flow_object_size = 100
 
    
class speakerDataGenerate():
    
    def __init__(self, data_dir):
        self.data_dir = data_dir  # "../input_output/vehicle_tracking/sample_video_out/"
        self.per_video_dir_lst = self.read_json_list(self.data_dir)
    
    
    def read_json_list(self, data_dir):
        # read all the json file in teh directory
        #data_dir = "../input_output/vehicle_tracking/sample_video_out/"
        
        #print("data_dir: ", data_dir)
        video_dir_lst = sorted(glob.glob(data_dir + '*sample_*_out*'))
        
        return video_dir_lst
    
    def get_data_numpy(self, file_dir):
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
    
            
    def get_absolute_speed(self, last_frm_box, frm_lst_indx, curre_frm_box, frm_curr_indx):
        
        #print("get_absolute_speed last_frm_box, frm_lst_indx, curre_frm_box, frm_curr_indx: ", last_frm_box, frm_lst_indx, curre_frm_box, frm_curr_indx)
        
        time_inteval =  frm_curr_indx - frm_lst_indx              # time interval unit is in frames, not in second
        
        center_a = np.array([(last_frm_box[3] + last_frm_box[1])/2, (last_frm_box[2] + last_frm_box[0])/2])
        
        center_b = np.array([(curre_frm_box[3] + curre_frm_box[1])/2, (curre_frm_box[2] + curre_frm_box[0])/2])
        
        cur_absolute_speed = (center_b  - center_a)/time_inteval
        
        #print("get_absolute_speed: ", cur_absolute_speed)
        return cur_absolute_speed
    
    def get_ema_absolute_speed(self, EMA_pre, cur_absolute_speed):
        # get the EMA speed
        ema_speed =  ALPHA * cur_absolute_speed  + (1-ALPHA) * EMA_pre
        #print("get_ema_absolute_speed ema_speed: ", ema_speed)
        return ema_speed
    
    
    def get_object_size(curr_box):
        # bounding_box: [y1, x1, y2, x2]  vertical height is y, horiztonal width is x. reso: (width, height)
        
        # coordinate is already normalized, no need to use reso
        	
        normalized_object_ratio = np.array([(curr_box[2] - curr_box[0]) * (curr_box[3] - curr_box[1])])
    
        return normalized_object_ratio


    def get_object_size_change(self, curr_box, last_frm_box):
        
        # bounding_box: [y1, x1, y2, x2]  vertical height is y, horiztonal width is x. reso: (width, height)
        
        # coordinate is already normalized, no need to use reso
        	
        curr_normalized_object_ratio = np.array([(curr_box[2] - curr_box[0]) * (curr_box[3] - curr_box[1])])
    
        prev_normalized_object_ratio = np.array([(last_frm_box[2] - last_frm_box[0]) * (last_frm_box[3] - last_frm_box[1])])
    
        return curr_normalized_object_ratio - prev_normalized_object_ratio
    
    
    def estimate_frame_rate(self, speaker_box_arr, acc_arr, frm_start_indx, ans_reso_indx, max_jump_number, min_acc_threshold):
        # estimate the jumping number
        
        frm_indx = frm_start_indx
        accumu_acc = 0.0
        
        max_frm_no = speaker_box_arr.shape[1]
        #init_acc = acc_arr[ans_reso_indx][frm_start_indx]  # acc_frame_arr[start_frame_index]  # considered as 1.0 now
    
        cnt_frames = 1   # frame rate selected
        ref_box = speaker_box_arr[0][frm_start_indx]
        
        while(frm_indx < min(max_frm_no, (max_jump_number + frm_start_indx))):
            
            # calculate the accuracy
            curr_box = speaker_box_arr[ans_reso_indx][frm_indx]
            acc = bb_intersection_over_union(ref_box, curr_box)
            
            accumu_acc += acc
            
            #print ("accumu_acc/cnt_frames: ", accumu_acc/cnt_frames, min_acc_threshold, cnt_frames)
            if accumu_acc/cnt_frames < min_acc_threshold:
                break
            
            frm_indx += 1
            cnt_frames += 1
        # final average acc
        
        jump_frame_number = min(cnt_frames, max_jump_number)
    
        #if cnt_frames == 1:
        #    average_acc = init_acc
        #else:
        
        average_acc = accumu_acc/jump_frame_number
        
        #print("estimate_frame_rate, jump_frames:", average_acc,  jump_frame_number)
        return jump_frame_number, average_acc
    
    
    def get_reso_selected(self, speaker_box_arr, current_aver_acc, frm_start_indx, jumping_frm_number, min_acc_threshold):
        # select the least reso that could satisfy the acc
        
        #print ("get_reso_selected enter here")
        max_frm_no = speaker_box_arr.shape[1]
    
        ans_reso_indx = 0   # ground truth
            
        ref_box = speaker_box_arr[0][frm_start_indx]     # reference pose
    
        last_average_acc = current_aver_acc
        for reso_indx in range(1, len(resoStrLst)):    # 1, 2, 3, 4
            # check the accuracy for this range    # if we can not find, use the last higher resolution
            curr_frame_index = frm_start_indx
            curr_accumulated_acc = 0.0
            while(curr_frame_index < min(max_frm_no, (frm_start_indx + jumping_frm_number))):
                curr_box = speaker_box_arr[reso_indx][curr_frame_index]
                
                acc = bb_intersection_over_union(ref_box, curr_box)     # acc with reference box
                curr_accumulated_acc += acc
                #print ("get_reso_selected curr_accumulated_acc: ", curr_accumulated_acc)
            
                curr_frame_index += 1
    
            average_acc = curr_accumulated_acc/(jumping_frm_number)  # jumping_frm_number include the start_frame_index
            
            if average_acc < min_acc_threshold:
                ans_reso_indx = reso_indx - 1
                
                break
            # print ("select_resolution average_acc: ", reso_indx, average_acc, min_acc_threshold)
            
            last_average_acc = average_acc
            
        return ans_reso_indx, last_average_acc


    def read_one_frame_arr(self, frame_path, curre_reso):
        # frame to numpy arr
        
        #prev_frame_arr = cv2.imread(prev_frame_path)
        #current_frame_arr = cv2.imread(curr_frame_path)
       
        width = int(curre_reso.split('x')[0])
        height = int(curre_reso.split('x')[1])
        frame_arr = io.imread(frame_path)
        frame_arr = cv2.resize(frame_arr, (width, height))

        return frame_arr
    
    
    def get_one_boundingbox(self, frame_arr, bounding_box):
        
        print("bounding_box: ", frame_arr.shape, bounding_box)
        top_left_x = int(bounding_box[0])
        top_left_y = int(bounding_box[1])
        btm_rt_x = int(bounding_box[2])
        btm_rt_y = int(bounding_box[3])
        
        box_arr = frame_arr[top_left_y:btm_rt_y, top_left_x: btm_rt_x]
        
        print("box_arr: ", box_arr.shape)
        
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
    
    

    def get_optical_flow_feature(self, video_frame_dir, frm_curr_indx, frm_lst_indx, curre_frm_box, last_frm_box, ans_reso_indx, last_reso_indx):
        
        curre_reso = resoStrLst[ans_reso_indx]
        curr_frame_path =  video_frame_dir + '/' + str(frm_curr_indx).zfill(5) + '.jpg'
        current_frame_arr = self.read_one_frame_arr(curr_frame_path, curre_reso)
        curr_box_arr = self.get_one_boundingbox(current_frame_arr, curre_frm_box)

        print("ans_reso_indx: ", ans_reso_indx, last_reso_indx)
        prev__reso = resoStrLst[last_reso_indx]
        prev_frame_path =  video_frame_dir + '/' + str(frm_lst_indx).zfill(5) + '.jpg'
        prev_frame_arr = self.read_one_frame_arr(prev_frame_path, prev__reso)
        prev_box_arr = self.get_one_boundingbox(prev_frame_arr, last_frm_box)

        print("curr_box_arr: ", curr_box_arr.shape, prev_box_arr.shape)
        
        optical_flow_arr = self.calculate_optical_flow(prev_box_arr, curr_box_arr, optical_flow_object_size)

        print("optical_flow_arr: ", optical_flow_arr.shape)

        
    def get_feature_data_instance_xy(self, file_dir, max_jump_number, min_acc_threshold):
        # get the relative speed
        # output feature, its data instances (x, y) 
        # x is the feature, y is the used (frame rate resolution)
        # speaker_box_arr numpy format:
        #reso/frame:   1   2  3 
        # 0            xx  xx
        # 1            xx
        # 2
        # 3
        # 4
        
        video_frame_dir =  file_dir   # +  '*_frames'  # 'car_traffic_' + str(video_id) + '_frames'
        
        for root, subdirs, files in os.walk(video_frame_dir):
            flag = False
            for frame_folder_name in subdirs:
                if '_frames' in frame_folder_name:
                    video_frame_dir = video_frame_dir + frame_folder_name
                    flag = True
                    break
            if flag:
                break
            
        
        print("video_frame_dir :", video_frame_dir)
        
        speaker_box_file = file_dir +  "single_speaker_box.npy"
        spf_file =  file_dir +  "single_spf.npy"
        speaker_box_arr = read_numpy(speaker_box_file)   # (5, 37977, 4) 
        spf_arr = read_numpy(spf_file)       # (5, 37977)
        
        acc_file = file_dir + "single_acc.npy"
        
        acc_arr = read_numpy(acc_file)  
        print("speaker_box_arr number: ", speaker_box_arr.shape, spf_arr.shape, acc_arr.shape)
        # start from 0, 1 frames
    
        
        reso_no = spf_arr.shape[0]
        frm_no = spf_arr.shape[1]
        
        
        ans_reso_indx = 0
        
        k = 25     # 
        frm_lst_indx = 1
        frm_curr_indx = k          # traverse from frame kth (index starts at 0)
        ema_absolute_curr = 0.0
        
        last_reso_indx = 0
        # initially used resolution
        last_frm_box = speaker_box_arr[ans_reso_indx][frm_lst_indx]   # initialize
          
        curr_detect_time = spf_arr[ans_reso_indx][0]        # first frame time
        delay_time_up = max(0, curr_detect_time - 1.0/PLAYOUT_RATE)      # delay time so far
        
        
        delay_up_arr = blist()
    
        delay_up_arr.append(delay_time_up)
                  
        data_one_video_instances_xy = blist()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        while(frm_curr_indx < frm_no):
            
            # calculate the relative feature
            
            curre_frm_box = speaker_box_arr[ans_reso_indx][frm_curr_indx]
            
            cur_absolute_speed = self.get_absolute_speed(last_frm_box, frm_lst_indx, curre_frm_box, frm_curr_indx)
             
            ema_absolute_curr = self.get_ema_absolute_speed(ema_absolute_curr, cur_absolute_speed)
             
            normalized_object_ratio = self.get_object_size_change(curre_frm_box, last_frm_box)   # get_object_size(curre_frm_box)
            #print ("normalized_object_ratio: " , normalized_object_ratio)
        
            optical_flow_next_pt = self.get_optical_flow_feature(video_frame_dir, frm_curr_indx, frm_lst_indx, curre_frm_box, last_frm_box, ans_reso_indx, last_reso_indx)
    
    
            xx
            feature_x = np.hstack((ema_absolute_curr, normalized_object_ratio))
            #print ("feature_x: " , feature_x)
            
            jumping_frm_number, current_aver_acc = self.estimate_frame_rate(speaker_box_arr, acc_arr, frm_curr_indx, ans_reso_indx, max_jump_number, min_acc_threshold)
            
            
            ans_reso_indx, average_acc = self.get_reso_selected(speaker_box_arr, current_aver_acc, frm_curr_indx, jumping_frm_number, min_acc_threshold)
            
            #print ("jumping_frm_number:feature_x,  ", jumping_frm_number, feature_x, ans_reso_indx)
            
            
            y_out = np.asarray([jumping_frm_number, ans_reso_indx])
            
            data_one_video_instances_xy.append(np.append(feature_x, y_out))
    
            curr_detect_time += spf_arr[ans_reso_indx][frm_curr_indx]
    
            last_frm_box = curre_frm_box
            
            frm_lst_indx = frm_curr_indx
            frm_curr_indx += jumping_frm_number  # 1  # jumping_frm_number   # 1    # jumping_frm_number  #
            
            last_reso_indx = ans_reso_indx
            #EMA_absolute_prev = EMA_absolute_curr   # update EMA
        
        data_one_video_instances_xy = np.asarray(data_one_video_instances_xy)
        print ("data_one_video_instances_xy:size,  ", data_one_video_instances_xy.shape)
        return data_one_video_instances_xy        
            
    
    
    
    def get_dataset_instances(self, input_dir, min_acc_threshold, max_jump_number):
       
        
        # dir_whole_data_instances_out = input_dir + "data_instance_xy_each_frm/"
        
        dir_whole_data_instances_out = input_dir + "all_data_instance_xy/"
        
    
            
        if not os.path.exists(dir_whole_data_instances_out):
            os.mkdir(dir_whole_data_instances_out)
                
        dir_whole_data_instances_out = dir_whole_data_instances_out + "minAcc_" + str(min_acc_threshold) + "/" 
        if not os.path.exists(dir_whole_data_instances_out):
            os.mkdir(dir_whole_data_instances_out)
                
        arr_whole_data_instances_xy = None
        
        
        for i, file_dir in enumerate(self.per_video_dir_lst[0:22]):      # use 22 video first
            
            file_dir = file_dir + '/'
            
            
            data_one_video_instance_xy = self.get_feature_data_instance_xy(file_dir, max_jump_number, min_acc_threshold)
            
            
            if i == 0:
                arr_whole_data_instances_xy = data_one_video_instance_xy
            else:
                arr_whole_data_instances_xy = np.vstack((arr_whole_data_instances_xy, data_one_video_instance_xy))
                
                
            # mkdir 
            #out_pickle_dir = file_dir + "data_instance_xy_each_frm/"     # data_instance_xy_each_frm
            out_pickle_dir = file_dir + "data_instance_xy/"
            if not os.path.exists(out_pickle_dir):
                os.mkdir(out_pickle_dir)
            
            out_pickle_dir = out_pickle_dir + "minAcc_" + str(min_acc_threshold) + "/" 
            if not os.path.exists(out_pickle_dir):
                os.mkdir(out_pickle_dir)
                         
    
            out_data_instance_xy_pickle = out_pickle_dir + "data_instance_xy.pkl" 
            write_pickle_data(data_one_video_instance_xy, out_data_instance_xy_pickle)
    
        
        whole_out_data_instance_xy_pickle = dir_whole_data_instances_out + "whole_data_instance_xy.pkl" 
        write_pickle_data(arr_whole_data_instances_xy, whole_out_data_instance_xy_pickle)
        
        
    
            
if __name__== "__main__": 
        
    speakerDataGenerateObj = speakerDataGenerate(input_dir)
        
    speakerDataGenerateObj.get_dataset_instances(input_dir, min_acc_threshold, max_jump_number)
    
    
        
        
    
    
    
    
    