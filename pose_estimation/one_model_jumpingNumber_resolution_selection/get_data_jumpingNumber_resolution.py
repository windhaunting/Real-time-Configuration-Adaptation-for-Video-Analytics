#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:56:58 2020

@author: fubao
"""


 #fixed a configuraiton with most expensive configuration then pick jumping frame number
 # then find the resolution that still satisfy the resolution
 
# predict jumping number and resolution at the same time.



import sys
import os
import math
import numpy as np

from blist import blist

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


'''
{{0, "Nose"}, //t
{1, "Neck"}, //f is not included in coco. How do you get the Neck keypoints
{2, "RShoulder"}, //t
{3, "RElbow"}, //t
{4, "RWrist"}, //t
{5, "LShoulder"}, //t
{6, "LElbow"}, //t
{7, "LWrist"}, //t
{8, "RHip"}, //t
{9, "RKnee"}, //t
{10, "RAnkle"}, //t
{11, "LHip"}, //t
{12, "LKnee"}, //t
{13, "LAnkle"}, //t
{14, "REye"}, //t
{15, "LEye"}, //t
{16, "REar"}, //t
{17, "LEar"}, //t
{18, "Bkg"}},
'''

# the keypoint printout in csv is actually tensorflow coco format

'''
0	nose
1	leftEye
2	rightEye
3	leftEar
4	rightEar
5	leftShoulder
6	rightShoulder
7	leftElbow
8	rightElbow
9	leftWrist
10	rightWrist
11	leftHip
12	rightHip
13	leftKnee
14	rightKnee
15	leftAnkle
16	rightAnkle
'''


# use EMA EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
# =  Price(current)  x Multiplier)   + (1-Multiplier) * EMA(prev) 

#resolution_dict = {int(r.split('x')[1]): i for i, r in enumerate(resoStrLst_OpenPose)}

#print ("resolution_dict : ", resolution_dict)


ALPHA = 0.8     #  The coefficient α represents the degree of weighting decrease, a constant smoothing factor between 0 and 1. A higher α discounts older observations faster.

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


max_jump_number = 25  # float('inf')  # 10  # float('inf')  # 25
min_acc_threshold = 0.92  # 0.92


class samplingResoDataGenerate(object):
    def __init__(self):
        pass


    def get_data_numpy(self, data_pose_keypoint_dir, data_pickle_dir, intervalFlag, all_config_flag):
        # read pickle data (30, 13390, 17, 3) (30, 13390,) (30, 13365,)
        # output: output all or  most expensive configuration's  numpy  (5, 13390, 17, 3) (5, 13390,) (5, 13365,)
        
        config_est_frm_arr = read_poseEst_conf_frm_more_dim(data_pickle_dir)
        
        acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir, intervalFlag)
    
        #print ('acc_frame_arr', config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        if all_config_flag:      # output all configuration 5x6 30 configs
            return config_est_frm_arr, acc_frame_arr, spf_frame_arr
        
        config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)
        
        print ("id_config_dict: ", id_config_dict)
  
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
    
    def get_estimated_keypoint_by_overall_speed(self, kp_pose_arr, cur_absolute_speed):
        average_speed = [s*(1/PLAYOUT_RATE) for s in cur_absolute_speed]
        
        new_kp_pose_arr = kp_pose_arr + average_speed * kp_pose_arr
        
        return new_kp_pose_arr
    
    def get_estimated_arm_leg_relative_speed(self, kp_pose_arr, arr_ema_relative_speed):
        index_critical_points = [7, 8, 9, 10, 13, 14, 15, 16]
        """
        dict_part_id = {'nose' : 0, 'leftEye': 1, 'rightEye': 2, 'leftEar': 3, 'rightEar': 4, 'leftShoulder': 5,
                 'rightShoulder': 6, 'leftElbow' : 7, 'rightElbow': 8, 'leftWrist': 9, 
                 'rightWrist': 10, 'leftHip': 11, 'rightHip':12, 'leftKnee': 13, 'rightKnee': 14,
                 'leftAnkle': 15, 'rightAnkle': 16}
        """
        KP_NUM = kp_pose_arr.shape[0]
        
        all_keypoint_relative_speed = np.zeros((KP_NUM, 3))
        #for i in range(0, len(index_critical_points)):
            
        all_keypoint_relative_speed[index_critical_points] = arr_ema_relative_speed
        
        average_all_keypoint_relative_speed = [s*(1/PLAYOUT_RATE) for s in all_keypoint_relative_speed]
        #print ("all_keypoint_speed: ",  all_keypoint_relative_speed)
        new_kp_pose_arr = kp_pose_arr + average_all_keypoint_relative_speed * kp_pose_arr
        
        return new_kp_pose_arr
    
    
    def estimate_jumping_number(self, config_est_frm_arr_all, acc_frame_arr_all, ans_reso_indx,  start_frame_index, arr_ema_absolute_speed, arr_ema_relative_speed, max_jump_number = 25,  min_acc_threshold = 0.95):
        #input: from a starting frame index, we use the current start frame to replace the several later frame
        # and calculate the accuracy, until we get an accuracy smaller than min_acc_threshold
        # ouput: the no. of the several frames jumped, and the replaced detection pose 
        # consider outlier and noisy data, we use max_jump_number here
        
        # get start frame's pose estimation result
        # print("estimate_jumping_number config_est_frm_arr_all: ", ans_reso_indx, config_est_frm_arr_all.shape, config_est_frm_arr_all[0].shape)
        
        frame_length = config_est_frm_arr_all[ans_reso_indx].shape[0]
        
        curr_frame_index = start_frame_index + 1
        ref_pose = config_est_frm_arr_all[0][start_frame_index]     # reference pose ground truth
        #print("ref_pose shape: ", ref_pose.shape)
        
        cnt_frames = 1      # include the first frame detected by DNN itself, how many jumping frame number
        # the acc for start_frame_index
        curr_accumulated_acc = acc_frame_arr_all[ans_reso_indx][start_frame_index]  # acc_frame_arr[start_frame_index]  # considered as 1.0 now
        average_acc = curr_accumulated_acc
        # print ("curr_accumulated_acc: ", curr_accumulated_acc)
        
        last_oks = 0
        next_frm_kp_arr = ref_pose
        
        while(curr_frame_index < frame_length and average_acc >= min_acc_threshold):
            # get the oks similarity acc
            curr_pose = config_est_frm_arr_all[ans_reso_indx][curr_frame_index]
            oks = computeOKS_1to1(next_frm_kp_arr, curr_pose, sigmas = None)     # oks with reference pose
            
            next_frm_kp_arr = self.get_estimated_keypoint_by_overall_speed(next_frm_kp_arr, arr_ema_absolute_speed)
            next_frm_kp_arr = self.get_estimated_arm_leg_relative_speed(next_frm_kp_arr, arr_ema_relative_speed)
            
            curr_accumulated_acc += oks
            #print ("oks: ", oks)
            average_acc = curr_accumulated_acc/(cnt_frames+1)
            
            curr_frame_index += 1
            cnt_frames += 1
            last_oks = oks
        
        cnt_frames = min(cnt_frames, max_jump_number)
        seg_detected_pos_frm_list = self.get_detected_pos_frm_arr(ref_pose, cnt_frames)
        if cnt_frames == 1:
            segment_average_acc = curr_accumulated_acc
        else:
            segment_average_acc = (curr_accumulated_acc - last_oks)/(cnt_frames-1)
        
        # print ("finallllllloks: ", curr_accumulated_acc, last_oks, (curr_accumulated_acc - last_oks), cnt_frames, segment_average_acc)
        return cnt_frames, seg_detected_pos_frm_list, segment_average_acc
    
    
    def get_detected_pos_frm_arr(self, start_frm_pos_arr, jumping_frame_num):
        # input: 17*3, one frame pos result
        # ouput: the array frame pos result with all frame detection number; jumping frame no need to detect actually
        detected_pos_frm_list = blist()
        for i in range(jumping_frame_num):
            detected_pos_frm_list.append(start_frm_pos_arr)
        #np.vstack([start_frm_pos_arr] * jumping_frame_num)
        #detected_pos_frm_arr = np.asarray(detected_pos_frm_arr)
        #print("detected_pos_frm_arr shape: ", start_frm_pos_arr, jumping_frame_num, detected_pos_frm_arr.shape)
        return detected_pos_frm_list
    
    
    def get_feature_x(self, arr_ema_speed):
        # from estimated speed to get mean and variance and add to the feature
        
        feature_vect_speed = arr_ema_speed[:, :2]
        #print("arr_ema_speed: ", feature_vect_speed)
        
        # get mean of the keypoints 
        feature_vect_mean = np.mean(feature_vect_speed, axis = 0)
        feature_vect_var = np.var(feature_vect_speed, axis = 0)
        #print("arr_ema_speed: ", feature_vect_mean, feature_vect_var, np.asarray([feature_vect_mean, feature_vect_var]))
        # get the jumping number y
        
        feature_x = np.hstack((feature_vect_speed.flatten(), feature_vect_mean, feature_vect_var))
                    #print("feature_x: ", feature_x)
        return feature_x


    def select_resolution(self, config_est_frm_arr_all, acc_frame_arr_all, spf_frame_arr_all, start_frame_index, jumping_frm_number, current_aver_acc, min_acc_threshold):
        #after getting the jumping number, pick the resolution above the min_acc and minimize delay
        # mininum delay
        # current_aver_acc is the acc achieved with frame jumping number selection
        
        # 5 resolutions to select
        # print ('select_resolution acc_frame_arr', config_est_frm_arr_all.shape, acc_frame_arr_all.shape, spf_frame_arr_all.shape)

        # count_jumping_frames = 1
        
        ans_reso_indx = 0   # ground truth
        
        ref_pose = config_est_frm_arr_all[0][start_frame_index]     # reference pose

        last_average_acc = current_aver_acc
        for reso_indx in range(1, 5):    # 1, 2, 3, 4
            # check the accuracy for this range    # if we can not find, use the last higher resolution
            curr_frame_index = start_frame_index
            curr_accumulated_acc = 0.0
            while(curr_frame_index < (start_frame_index + jumping_frm_number)):
                curr_pose = config_est_frm_arr_all[reso_indx][curr_frame_index]
                
                oks = computeOKS_1to1(ref_pose, curr_pose, sigmas = None)     # oks with reference pose
                curr_accumulated_acc += oks
                #print ("oks: ", oks)
            
                curr_frame_index += 1

            average_acc = curr_accumulated_acc/(jumping_frm_number)  # jumping_frm_number include the start_frame_index
            
            if average_acc < min_acc_threshold:
                ans_reso_indx = reso_indx - 1
                
                break
            # print ("select_resolution average_acc: ", reso_indx, average_acc, min_acc_threshold)
            
            last_average_acc = average_acc
        
        return ans_reso_indx, last_average_acc
    
    
    def get_absolute_speed(self, current_frm_est, prev_frm_est, count_jumping_frames, arr_ema_absolute_speed):
        
        time_interval = count_jumping_frames    #  # time interval unit is in frames, not in second (1.0/PLAYOUT_RATE)*count_jumping_frames
        arr_vec_diff = current_frm_est - prev_frm_est

        arr_abs_vec_diff =  arr_vec_diff  # np.absolute(arr_vec_diff)
        #print("abs_vec_diff: ", arr_abs_vec_diff)
        arr_speed_vec = arr_abs_vec_diff/time_interval      # current speed
        arr_ema_absolute_speed = arr_speed_vec * ALPHA + (1.0-ALPHA) * arr_ema_absolute_speed
        
        return arr_ema_absolute_speed
            
    def get_relative_speed_to_body_center(self, current_frm_est, prev_frm_est, count_jumping_frames, arr_ema_relative_speed):
        # get relative speed vector feature
        # body center use 4 points formedrectangle center, (left shoulder, right shoulder, left hip, right hip), 2 points are enough
        # get 8 points left elblow, right elbow, left wrist, right wrist, left knee, right knee, left ankle, right ankle
        #  interval_frm is for EMA current speed difference, how many previous frames are used to calculate
        
        #print ("get_relative_speed_to_body_center: current_frm_est: ", current_frm_est.shape, current_frm_est[0] )
        time_interval = count_jumping_frames   # #  # time interval unit is in frames, not in second  (1.0/PLAYOUT_RATE)*count_jumping_frames
        
        arr_relative_speed = np.zeros((8, 3))
        
        index_center_rectangle_points = [6, 12]    # use two point to get rectangle  
        
        index_critical_points = [7, 8, 9, 10, 13, 14, 15, 16]
        
        
        # get center point coordinate of current frame and previous frame
        curr_xy1 = current_frm_est[index_center_rectangle_points[0]]     # current frame's left shoulder points
        curr_xy2 = current_frm_est[index_center_rectangle_points[1]]
        curr_center_vector = [(curr_xy1[0] + curr_xy2[0])/2.0, (curr_xy1[1] + curr_xy2[1])/2.0, 2.0]
        
        prev_xy1 = prev_frm_est[index_center_rectangle_points[0]]     # current frame's left shoulder points
        prev_xy2 = prev_frm_est[index_center_rectangle_points[1]]
        prev_center_vector = [(prev_xy1[0] + prev_xy2[0])/2.0, (prev_xy1[1] + prev_xy2[1])/2.0, 2.0]
     
        #print ("get_relative_speed_to_body_center: center_vector_current: ", curr_center_vector,  prev_center_vector)
        
        
        # get relative speed
        
        for i, indx_pt in enumerate(index_critical_points):
            # get the index points
            curr_frm_relative_vec = current_frm_est[indx_pt] - curr_center_vector
            
            prev_frm_relative_vec = prev_frm_est[indx_pt] - prev_center_vector
            
            relative_vec_dist = curr_frm_relative_vec - prev_frm_relative_vec
            
            #print ("get_relative_speed_to_body_center: center_vector_current: ", relative_vec_dist)
            
            arr_relative_speed[i] = relative_vec_dist / time_interval
            
        arr_ema_relative_speed = arr_relative_speed * ALPHA + (1.0-ALPHA) * arr_ema_relative_speed
        
        return arr_relative_speed

    def get_object_size_x(self, current_frm_est): # get object size ratio with 3
        # left shoulder $ls_i$, right shoulder $rs_i$, left hip $lh_i$
        
        # the pose estimation point is already a ratio to the resolution size, so the image size is always 1.0
        # get the center point area
        left_shoulder_pt = current_frm_est[5]
        rt_shoulder_pt = current_frm_est[6]
        left_hip_pt = current_frm_est[11]
        object_size_ratio = math.sqrt(((rt_shoulder_pt - left_shoulder_pt)[0])**2 + ((rt_shoulder_pt - left_shoulder_pt)[1])**2) * math.sqrt(((left_hip_pt - left_shoulder_pt)[0]**2 + ((left_hip_pt - left_shoulder_pt)[1])**2))/1.0
        
        
        #print ("object_size_ratio: ", object_size_ratio)
              
        #xxxxx 
        return object_size_ratio    #  x, y-axis
    
    
    def get_object_size_change(self, current_frm_est, prev_frm_est):
        # object size change
        left_shoulder_pt = current_frm_est[5]
        rt_shoulder_pt = current_frm_est[6]
        left_hip_pt = current_frm_est[11]
        current_object_size_ratio = math.sqrt(((rt_shoulder_pt - left_shoulder_pt)[0])**2 + ((rt_shoulder_pt - left_shoulder_pt)[1])**2) * math.sqrt(((left_hip_pt - left_shoulder_pt)[0]**2 + ((left_hip_pt - left_shoulder_pt)[1])**2))/1.0
        
        
        # last frame ratio
        left_shoulder_pt = prev_frm_est[5]
        rt_shoulder_pt = prev_frm_est[6]
        left_hip_pt = prev_frm_est[11]
        prev_object_size_ratio = math.sqrt(((rt_shoulder_pt - left_shoulder_pt)[0])**2 + ((rt_shoulder_pt - left_shoulder_pt)[1])**2) * math.sqrt(((left_hip_pt - left_shoulder_pt)[0]**2 + ((left_hip_pt - left_shoulder_pt)[1])**2))/1.0
        
        return current_object_size_ratio - prev_object_size_ratio
        
    
    def get_data_instances(self, config_est_frm_arr_all, acc_frame_arr_all, spf_frame_arr_all, max_jump_number = 10, min_acc_threshold = 0.95, speed_type = 'ema', all_kp_flag = 1, interval_frm = 10):
        # get estimated speed    
        # interval_frm for calculating the current estimated speed
        # all_kp_flag: use all points or 8 critical keypoints  (arm, wrist, knee, ankle) 
        
        # use most expensive resolution here
        #config_est_frm_arr = config_est_frm_arr_all[0]          #  most expensive used here
        #acc_frame_arr = acc_frame_arr_all[0]
        # spf_frame_arr = spf_frame_arr_all[0]
        
        detected_est_frm_arr = blist()       # detected pose result, each is 17x3,  because of jumping number, each segment has several frame replaced as the detection
        
        # add the first one frame for the first time speed
        ans_reso_indx = 0  # # use highest resolution first
        detected_est_frm_arr += config_est_frm_arr_all[ans_reso_indx][0:(interval_frm+1)]
        
        #print ("detected_est_frm_arr: ", config_est_frm_arr_all.shape, len(detected_est_frm_arr), detected_est_frm_arr[0])
        
        #aaa
        segment_acc_arr = blist()       # each segment accuracy of pose result,
        segment_acc_arr.append(acc_frame_arr_all[ans_reso_indx][0])
        
        # delay
        #print ("detected_est_frm_arr indx 1: ", config_est_frm_arr_all.shape)
    
        delay_up_arr = blist()
        # current delay for this 
        curr_detect_time = spf_frame_arr_all[ans_reso_indx][0]        # first frame time
        delay_time_up = max(0, curr_detect_time - 1.0/PLAYOUT_RATE)      # delay time so far
        delay_up_arr.append(delay_time_up)
        
        
        list_estimated_speed_2_jumpinNumberResolution = blist()           # speed => jumping_frame_number and resolution
        
        FRM_NO =  spf_frame_arr_all[ans_reso_indx].shape[0]                # total frame no for the whole video
        
        #print("FRM_NO: ", FRM_NO, spf_frame_arr.shape, curr_detect_time)
                
        if all_kp_flag == 1:        # all keypoints
            arr_ema_absolute_speed = np.zeros((NUM_KEYPOINT, 3))
            arr_ema_relative_speed  = np.zeros((8, 3))    # 8 points to calculate relative speed
            
        count_jumping_frames = interval_frm
        if speed_type == 'ema':
            # Price(current)  x Multiplier)   + (1-Multiplier) * EMA(prev) 
            # Vi= ALPHA * (M_i) + (1-ALPHA)* V_{i-1}
            # current speed use previous current frame - previous frame as estimation now
            # start from 2nd frame
            current_indx = interval_frm          # 1  starting index = 0  left frame no = 1

            while(current_indx < FRM_NO):
                #get current speed, use current frame estimation point -  previous estimation point, and divide by 
                # frame time interval
                # vector speed
                current_frm_est = config_est_frm_arr_all[ans_reso_indx][current_indx]   # detected_est_frm_arr[current_indx] #  current frame is detected, so we use
                
                prev_used_indx =  max(current_indx - count_jumping_frames, 0)  # max(current_indx - 1, 0)
                
                #print("detected_est_frm_arr: ", len(detected_est_frm_arr), current_indx, prev_used_indx, count_jumping_frames)
                prev_frm_est = detected_est_frm_arr[prev_used_indx]
                
                #print("current_frm_est: ", current_frm_est)
                #print("prev_frm_est: ", prev_frm_est)
                
                # calculate current speed vector
                if all_kp_flag == 1:        # use all the 17 keypoints  # absolute
                    #print("vec_diff: ", vec_diff)
                    
                    # get absolute speed 
                    arr_ema_absolute_speed = self.get_absolute_speed(current_frm_est, prev_frm_est, count_jumping_frames, arr_ema_absolute_speed)

                    # get relative speed 
                    arr_ema_relative_speed = self.get_relative_speed_to_body_center(current_frm_est, prev_frm_est, count_jumping_frames, arr_ema_relative_speed)
                    
                    feature_x_absolute = self.get_feature_x(arr_ema_absolute_speed)

                    feature_x_relative = self.get_feature_x(arr_ema_relative_speed)
                    
                    feature_x_object_size = self.get_object_size_change(current_frm_est, prev_frm_est)   # self.get_object_size_x(current_frm_est)      
                    
                    feature_x = np.hstack((feature_x_absolute, feature_x_relative, feature_x_object_size))

                    #print("feature_x_relative: ", feature_x_absolute.shape, feature_x_relative.shape, feature_x.shape)
                    
                    
                    start_frame_index = current_indx
                    count_jumping_frames, seg_detected_pos_frm_list, average_acc = self.estimate_jumping_number(config_est_frm_arr_all, acc_frame_arr_all, ans_reso_indx, start_frame_index, arr_ema_absolute_speed, arr_ema_relative_speed, max_jump_number, min_acc_threshold)
                    #print("count_jumping_frames: ", count_jumping_frames, average_acc)
                    
                    
                    # get the resolution 
                    ans_reso_indx, average_acc_resolution = self.select_resolution(config_est_frm_arr_all, acc_frame_arr_all, spf_frame_arr_all, start_frame_index, count_jumping_frames, average_acc, min_acc_threshold)                    
                    
                    # list_estimated_speed_2_jump_number.append([feature_x, end_frame_index])
                    #print("data_one_instance: ", data_one_instance)
                        
                    if count_jumping_frames > max_jump_number:
                        count_jumping_frames = max_jump_number
                    y_out = np.asarray([count_jumping_frames, ans_reso_indx])
                    
                    data_one_instance_jumpingNumberReso = np.append(feature_x, y_out)
                    
                    list_estimated_speed_2_jumpinNumberResolution.append(data_one_instance_jumpingNumberReso)
                    
                    # add the accuracy of each segment
                    segment_acc_arr.append(average_acc_resolution)   # (average_acc)
                    
                    # get the current delay
                    curr_detect_time += spf_frame_arr_all[ans_reso_indx][current_indx]
                    # count_jumping_frames
                    # print("current_indx + count_jumping_frames: ", current_indx, curr_detect_time-(current_indx + count_jumping_frames-1)*1.0/PLAYOUT_RATE)
                    delay_time_up = max(0, delay_time_up + curr_detect_time - (current_indx + count_jumping_frames -1)*1.0/PLAYOUT_RATE)
                    delay_up_arr.append(delay_time_up)
                    
                    # update the detected result of pose 
                    detected_est_frm_arr += seg_detected_pos_frm_list
                    
                    current_indx +=  count_jumping_frames  # 1 # count_jumping_frames  # update next segment start index
                    #print("ddddddddetected_est_frm_arr: ", len(detected_est_frm_arr), len(seg_detected_pos_frm_list), count_jumping_frames, current_indx)
                        
                    #print ("delay_time_up", average_acc, count_jumping_frames, delay_time_up, ans_reso_indx)

    
        #print("list_estimated_speed_2_jump_number: ", len(list_estimated_speed_2_jumpinNumberResolution))
        arr_estimated_speed_2_jumpingNumber_reso = np.asarray(list_estimated_speed_2_jumpinNumberResolution)
        
        # print("arr_estimated_speed_2_jumpingNumber_reso: ", arr_estimated_speed_2_jumpingNumber_reso.shape, arr_estimated_speed_2_jumpingNumber_reso[0])

        detected_est_frm_arr = np.asarray(detected_est_frm_arr)
        segment_acc_arr = np.asarray(segment_acc_arr)
        delay_up_arr = np.asarray(delay_up_arr)
        print ("delay_up_arr", np.mean(segment_acc_arr), delay_up_arr[-1])
        
        return arr_estimated_speed_2_jumpingNumber_reso, detected_est_frm_arr, segment_acc_arr, delay_up_arr 
    

               
    def getDataExamples(self):
        # input video 
        #video_dir_lst = ['output_001-dancing-10mins/', 'output_006-cardio_condition-20mins/', 'output_008-Marathon-20mins/']   
            
    
        global dataDir3
        dataDir3 = "../" + dataDir3
        #all_arr_estimated_speed_2_jump_number = blist()  # all video
        all_arr_estimated_speed_2_resolution = None
        for i, video_dir in enumerate(video_dir_lst[0:35]): # [3:4]):    # [2:3]:   #[1:2]:  # [1:2]:  #[0:1]:        #[1:2]:
            data_pose_keypoint_dir = dataDir3 + video_dir
            
            data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
            #data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result_each_frm/'
            intervalFlag = 'frame'
            all_config_flag = False
            config_est_frm_arr, acc_frame_arr, spf_frame_arr = self.get_data_numpy(data_pose_keypoint_dir, data_pickle_dir, intervalFlag, all_config_flag)
            
            speed_type = 'ema'
            interval_frm = 10        # 1
            all_kp_flag = 1
           
            
            #print("config_est_frm_arr shape:  ", i, type(config_est_frm_arr), acc_frame_arr.shape, spf_frame_arr.shape)
            arr_estimated_speed_2_jumpingNumber_reso, detected_est_frm_arr, segment_acc_arr, delay_up_arr = self.get_data_instances(config_est_frm_arr, acc_frame_arr, spf_frame_arr, max_jump_number, min_acc_threshold, speed_type, all_kp_flag, interval_frm)
            
            print("all_arr_estimated_speed_2_resolution: ", arr_estimated_speed_2_jumpingNumber_reso.shape)

            if i == 0:
                all_arr_estimated_speed_2_resolution = arr_estimated_speed_2_jumpingNumber_reso
            else:
                all_arr_estimated_speed_2_resolution = np.vstack((all_arr_estimated_speed_2_resolution, arr_estimated_speed_2_jumpingNumber_reso))
            
            #print("eeeeeeall_arr_estimated_speed_2_jump_number: ", arr_estimated_speed_2_jump_number.shape,  all_arr_estimated_speed_2_jump_number.shape)
            #write_sub_dir_1 = dataDir3 + video_dir + "jumping_number_result_each_frm/" 
            write_sub_dir_1 = dataDir3 + video_dir + "jumping_number_result/" 

            if not os.path.exists(write_sub_dir_1):
                os.mkdir(write_sub_dir_1)
            write_sub_dir_2 =  write_sub_dir_1 + "jumpingNumber_resolution_selection/"
            if not os.path.exists(write_sub_dir_2):
                os.mkdir(write_sub_dir_2)
            
            out_pickle_dir = write_sub_dir_2 + "intervalFrm-" + str(interval_frm) + "_speedType-" + str(speed_type) + "_minAcc-" + str(min_acc_threshold) + "/"
            
            if not os.path.exists(out_pickle_dir):
                os.mkdir(out_pickle_dir)
                
     
            out_data_pickle_file = out_pickle_dir + "data_instance_speed_jumpingNumber_resolution_objectSizeRatio_xy.pkl" 
            write_pickle_data(arr_estimated_speed_2_jumpingNumber_reso, out_data_pickle_file)
            
            out_pose_est_frm_pickle = out_pickle_dir + "pose_est_frm.pkl" 
            write_pickle_data(detected_est_frm_arr, out_pose_est_frm_pickle)
            
            out_seg_acc_frm_pickle = out_pickle_dir + "segment_acc.pkl"
            write_pickle_data(segment_acc_arr, out_seg_acc_frm_pickle)
    
            out_delay_up_pickle = out_pickle_dir + "delay_time_up.pkl" 
            write_pickle_data(delay_up_arr, out_delay_up_pickle)
    
        
        #output_dir = dataDir3 + "dynamic_jumpingNumber_resolution_selection_output_each_frm/"
        output_dir = dataDir3 + "dynamic_jumpingNumber_resolution_selection_output/"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        subDir = output_dir + "intervalFrm-" + str(interval_frm) + "_speedType-" + str(speed_type) + "_minAcc-" + str(min_acc_threshold) + "/"
            
        if not os.path.exists(subDir):
            os.mkdir(subDir)
        self.write_all_instance_data(subDir, all_arr_estimated_speed_2_resolution)
             
    
    def write_all_instance_data(self, output_dir, all_arr_estimated_speed_2_jump_number):
        # write all the current video into the output
        out_data_pickle_file = output_dir + "all_data_instance_speed_JumpingNumber_resolution_objectSizeRatio_xy.pkl" 
        write_pickle_data(all_arr_estimated_speed_2_jump_number, out_data_pickle_file)
        
if __name__== "__main__": 
    
    data_obj = samplingResoDataGenerate()
    data_obj.getDataExamples()
    
    
    