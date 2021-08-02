#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:09:27 2020

@author: fubao
"""



# get the data for modeling jumping frames

import sys
import os

import numpy as np

from blist import blist

from data_file_process import write_pickle_data

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from classifierForSwitchConfig.common_classifier import read_poseEst_conf_frm_more_dim
from classifierForSwitchConfig.common_classifier import readProfilingResultNumpy

from profiling.common_prof import dataDir3
from profiling.common_prof import PLAYOUT_RATE
from profiling.common_prof import NUM_KEYPOINT   
from profiling.common_prof import resoStrLst_OpenPose

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


ALPHA = 0.8     # EMA
max_jump_number =  25  # float('inf')  #
min_acc_threshold = 0.92  # 0.92

video_dir_lst =  ['output_001_dance/', 'output_002_dance/', \
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
            
class DataGenerate(object):
    def __init__(self):
        pass


    def get_data_numpy_most_expensive(self, data_pose_keypoint_dir, data_pickle_dir):
        # read pickle data
        # output: most expensive configuration's  numpy  (13390, 17, 3) (13390,) (13365,)
        
        config_est_frm_arr = read_poseEst_conf_frm_more_dim(data_pickle_dir)
        
        intervalFlag = 'frame'
        acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir, intervalFlag)
    
        #print ('acc_frame_arr', config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        #config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)
        
        #print ("config_id_dict: ", len(config_id_dict), config_id_dict)
        
        # use most expensive configuration resolution to test now
        config_est_frm_arr = config_est_frm_arr[0]          # 
        acc_frame_arr = acc_frame_arr[0]
        spf_frame_arr = spf_frame_arr[0]
        print ('acc_frame_arr', config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        #estimat_frm_arr_more_dim = np.apply_along_axis(getPersonEstimation, 0, config_est_frm_arr)
        return config_est_frm_arr, acc_frame_arr, spf_frame_arr
    
    def get_data_numpy(self, data_pose_keypoint_dir, data_pickle_dir):
        # read pickle data
        # output: most expensive configuration's  numpy  (13390, 17, 3) (13390,) (13365,)
        
        config_est_frm_arr = read_poseEst_conf_frm_more_dim(data_pickle_dir)
        
        intervalFlag = 'frame'
        acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir, intervalFlag)
    
        #print ('acc_frame_arr', config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)
        
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
        
        config_est_frm_arr = config_est_frm_arr[resolution_id_max_frame_rate]          # 
        acc_frame_arr = acc_frame_arr[resolution_id_max_frame_rate]
        spf_frame_arr = spf_frame_arr[resolution_id_max_frame_rate]
        #print ('222acc_frame_arr', config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        #estimat_frm_arr_more_dim = np.apply_along_axis(getPersonEstimation, 0, config_est_frm_arr)
        return config_est_frm_arr, acc_frame_arr, spf_frame_arr
    

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
    

    def get_absolute_speed(self, current_frm_est, prev_frm_est, interval_frm, arr_ema_absolute_speed):
        
        time_interval = (1.0/PLAYOUT_RATE)*interval_frm
        arr_vec_diff = current_frm_est - prev_frm_est

        arr_abs_vec_diff = np.absolute(arr_vec_diff)
        #print("abs_vec_diff: ", arr_abs_vec_diff)
        arr_speed_vec = arr_abs_vec_diff/time_interval      #current speed
        arr_ema_absolute_speed = arr_speed_vec * ALPHA + (1.0-ALPHA) * arr_ema_absolute_speed
        
        return arr_ema_absolute_speed
            
    def get_relative_speed_to_body_center(self, current_frm_est, prev_frm_est, interval_frm, arr_ema_relative_speed):
        # get relative speed vector feature
        # body center use 4 points formedrectangle center, (left shoulder, right shoulder, left hip, right hip), 2 points are enough
        # get 8 points left elblow, right elbow, left wrist, right wrist, left knee, right knee, left ankle, right ankle
        #  interval_frm is for EMA current speed difference, how many previous frames are used to calculate
        
        #print ("get_relative_speed_to_body_center: current_frm_est: ", current_frm_est.shape, current_frm_est[0] )
        time_interval = (1.0/PLAYOUT_RATE)*interval_frm
        
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

    
    def estimate_jumping_number(self, config_est_frm_arr, acc_frame_arr, start_frame_index, max_jump_number = 500,  min_acc_threshold = 0.95):
        #input: from a starting frame index, we use the current start frame to replace the several later frame
        # and calculate the accuracy, until we get an accuracy smaller than min_acc_threshold
        # ouput: the no. of the several frames jumped, and the replaced detection pose 
        # consider outlier and noisy data, we use max_jump_number here
        
        # get start frame's pose estimation result
        #print("config_est_frm_arr: ", config_est_frm_arr.shape)
        frame_length = config_est_frm_arr.shape[0]
        
        curr_frame_index = start_frame_index + 1
        ref_pose = config_est_frm_arr[start_frame_index]     # reference pose
        #print("ref_pose shape: ", ref_pose.shape)
        
        cnt_frames = 1      # include the first frame detected by DNN itself
        # the acc for start_frame_index
        curr_accumulated_acc = acc_frame_arr[start_frame_index]  # acc_frame_arr[start_frame_index]  # considered as 1.0 now
        average_acc = curr_accumulated_acc
        # print ("curr_accumulated_acc: ", curr_accumulated_acc)
        
        last_oks = 0
        while(curr_frame_index < frame_length and average_acc >= min_acc_threshold):
            # get the oks similarity acc
            curr_pose = config_est_frm_arr[curr_frame_index]
            oks = computeOKS_1to1(ref_pose, curr_pose, sigmas = None)     # oks with reference pose
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
    

    def get_data_instances(self, config_est_frm_arr, acc_frame_arr, spf_frame_arr, max_jump_number = 25, min_acc_threshold = 0.95, speed_type = 'ema', all_kp_flag = 1, interval_frm = 1):
        # get estimated speed    
        # interval_frm for calculating the current estimated speed
        # all_kp_flag: use all points or 8 critical keypoints  (arm, wrist, knee, ankle) 
        
        detected_est_frm_arr = blist()       # detected pose result, each is 17x3,  because of jumping number, each segment has several frame replaced as the detection
        
        
        # add the first one frame for the first time speed
        detected_est_frm_arr.append(config_est_frm_arr[0]) 
          
        segment_acc_arr = blist()       # each segment accuracy pose result,
        segment_acc_arr.append(acc_frame_arr[0])
        
        # delay
        print ("detected_est_frm_arr indx 1: ", config_est_frm_arr.shape)
    
        delay_up_arr = blist()
        # current delay for this 
        curr_detect_time = spf_frame_arr[0]        # first frame time
        delay_time_up = max(0, curr_detect_time - 1.0/PLAYOUT_RATE)      # delay time so far
        delay_up_arr.append(delay_time_up)
        
        
        list_estimated_speed_2_jump_number = blist()          # data x => y
        FRM_NO =  spf_frame_arr.shape[0]          # total frame no for the whole video
        
        #print("FRM_NO: ", FRM_NO, spf_frame_arr.shape, curr_detect_time)
        
        # ALPHA = 0.8
        #time_interval = 1.0/PLAYOUT_RATE*interval_frm
        
        if all_kp_flag == 1:        # all keypoints
            arr_ema_absolute_speed = np.zeros((NUM_KEYPOINT, 3))
            arr_ema_relative_speed  = np.zeros((8, 3))    # 8 points to calculate relative speed


        if speed_type == 'ema':
            # Price(current)  x Multiplier)   + (1-Multiplier) * EMA(prev) 
            # Vi= ALPHA * (M_i) + (1-ALPHA)* V_{i-1}
            # current speed use previous current frame - previous frame as estimation now
            # start from 2nd frame
            current_indx = 1          # starting index = 0  left frame no = 1

            while(current_indx < FRM_NO):
                #get current speed, use current frame estimation point -  previous estimation point, and divide by 
                # frame time interval
                # vector speed
                current_frm_est = config_est_frm_arr[current_indx]  # detected_est_frm_arr[current_indx] #  current frame is detected, so we use
                
                prev_used_indx = max(current_indx - interval_frm, 0)
                
                #print("detected_est_frm_arr: ", len(detected_est_frm_arr), current_indx, prev_used_indx)
                prev_frm_est = detected_est_frm_arr[prev_used_indx]
                
                #print("current_frm_est: ", current_frm_est)
                #print("prev_frm_est: ", prev_frm_est)
                
                # calculate current speed vector
                if all_kp_flag == 1:        # use all the 17 keypoints  # absolute
                    
                    arr_ema_absolute_speed = self.get_absolute_speed(current_frm_est, prev_frm_est, interval_frm, arr_ema_absolute_speed)

                    # get relative speed 
                    arr_ema_relative_speed = self.get_relative_speed_to_body_center(current_frm_est, prev_frm_est, interval_frm, arr_ema_relative_speed)
                    
                    feature_x_absolute = self.get_feature_x(arr_ema_absolute_speed)

                    feature_x_relative = self.get_feature_x(arr_ema_relative_speed)
                    feature_x = np.hstack((feature_x_absolute, feature_x_relative))
                    
                    
                    start_frame_index = current_indx
                    count_jumping_frames, seg_detected_pos_frm_list, average_acc = self.estimate_jumping_number(config_est_frm_arr, acc_frame_arr, start_frame_index, max_jump_number, min_acc_threshold)
                    #print("end_frame_index: ", end_frame_index)
                    
                    data_one_instance = np.append(feature_x, count_jumping_frames)
                    # list_estimated_speed_2_jump_number.append([feature_x, end_frame_index])
                    #print("data_one_instance: ", data_one_instance)
                    list_estimated_speed_2_jump_number.append(data_one_instance)
        
        
                    # add the accuracy of each segment
                    segment_acc_arr.append(average_acc)
                    
                    # get the current delay
                    curr_detect_time += spf_frame_arr[current_indx]
                    # count_jumping_frames
                    # print("current_indx + count_jumping_frames: ", current_indx, curr_detect_time-(current_indx + count_jumping_frames-1)*1.0/PLAYOUT_RATE)
                    delay_time_up = max(0, delay_time_up + curr_detect_time - (current_indx + count_jumping_frames -1)*1.0/PLAYOUT_RATE)
                    delay_up_arr.append(delay_time_up)
                    
                    # update the detected result of pose 
                    detected_est_frm_arr += seg_detected_pos_frm_list
                    
                    current_indx += count_jumping_frames  # update next segment start index
                    #print("ddddddddetected_est_frm_arr: ", len(detected_est_frm_arr), len(seg_detected_pos_frm_list), count_jumping_frames, current_indx)
                        
                    
                    #print ("delay_time_up", average_acc, count_jumping_frames, delay_time_up)
                    
                #if current_indx > 300:  # debug only
                #    break
                
            
        #print("list_estimated_speed_2_jump_number: ", len(list_estimated_speed_2_jump_number), len(detected_est_frm_arr))
        arr_estimated_speed_2_jump_number = np.asarray(list_estimated_speed_2_jump_number)
        detected_est_frm_arr = np.asarray(detected_est_frm_arr)
        segment_acc_arr = np.asarray(segment_acc_arr)
        delay_up_arr = np.asarray(delay_up_arr)
        
        return arr_estimated_speed_2_jump_number, detected_est_frm_arr, segment_acc_arr, delay_up_arr
            
    
    def get_feature_x(self, arr_ema_speed):
        # from estimated speed
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
    
    def getDataExamples(self):
        # input video 
        #video_dir_lst = ['output_001-dancing-10mins/', 'output_006-cardio_condition-20mins/', 'output_008-Marathon-20mins/']   
        
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
            
        #all_arr_estimated_speed_2_jump_number = blist()  # all video
        all_arr_estimated_speed_2_jump_number = None
        for i, video_dir in enumerate(video_dir_lst):  # [3:4]:    # [2:3]:   #[1:2]:  # [1:2]:  #[0:1]:        #[1:2]:
            data_pose_keypoint_dir =  dataDir3 + video_dir
            
            data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
            config_est_frm_arr, acc_frame_arr, spf_frame_arr = self.get_data_numpy_most_expensive(data_pose_keypoint_dir, data_pickle_dir)
            
            speed_type = 'ema'
            interval_frm = 1
            all_kp_flag = 1
            
            
            #print("config_est_frm_arr shape:  ", i, type(config_est_frm_arr), acc_frame_arr.shape, spf_frame_arr.shape)
            arr_estimated_speed_2_jump_number, detected_est_frm_arr, segment_acc_arr, delay_up_arr = self.get_data_instances(config_est_frm_arr, acc_frame_arr, spf_frame_arr, max_jump_number, min_acc_threshold, speed_type, all_kp_flag, interval_frm)
            
            
            if i == 0:
                all_arr_estimated_speed_2_jump_number = arr_estimated_speed_2_jump_number
            else:
                all_arr_estimated_speed_2_jump_number = np.vstack((all_arr_estimated_speed_2_jump_number, arr_estimated_speed_2_jump_number))
                
            #print("eeeeeeall_arr_estimated_speed_2_jump_number: ", arr_estimated_speed_2_jump_number.shape,  all_arr_estimated_speed_2_jump_number.shape)
            out_pickle_dir =  dataDir3 + video_dir + "/jumping_number_result_each_frm/"

            #out_pickle_dir =  dataDir3 + video_dir + "/jumping_number_result/"
            if not os.path.exists(out_pickle_dir):
                os.mkdir(out_pickle_dir)
            
            next_dir = out_pickle_dir + 'jumping_number_prediction/'
            if not os.path.exists(next_dir):
                os.mkdir(next_dir)
                
            subDir = next_dir + "intervalFrm-" + str(interval_frm) + "_speedType-" + str(speed_type) + "_minAcc-" + str(min_acc_threshold) + "/"
            
            if not os.path.exists(subDir):
                os.mkdir(subDir)
                
            out_data_pickle_file = subDir + "data_instance_xy.pkl" 
            write_pickle_data(arr_estimated_speed_2_jump_number, out_data_pickle_file)
            
            #out_pose_est_frm_pickle = subDir + "pose_est_frm.pkl" 
            #write_pickle_data(detected_est_frm_arr, out_pose_est_frm_pickle)
            
            #out_seg_acc_frm_pickle = subDir + "segment_acc.pkl"
            #write_pickle_data(segment_acc_arr, out_seg_acc_frm_pickle)
            
            #out_delay_up_pickle = subDir + "delay_time_up.pkl" 
            #write_pickle_data(delay_up_arr, out_delay_up_pickle)
            
        
        # write all data instance file
        #print("all_arr_estimated_speed_2_jump_number: ", all_arr_estimated_speed_2_jump_number.shape)
        #output_dir = dataDir3 + "dynamic_jumping_frame_output/"
        #if not os.path.exists(subDir):
        #    os.mkdir(subDir)
        #self.write_all_instance_data(output_dir, all_arr_estimated_speed_2_jump_number)
                                    
            
    def write_all_instance_data(self, output_dir, all_arr_estimated_speed_2_jump_number):
        # write all the current video into the output
        out_data_pickle_file = output_dir + "all_data_instance_xy.pkl" 
        write_pickle_data(all_arr_estimated_speed_2_jump_number, out_data_pickle_file)
            
            
            
      
if __name__== "__main__": 
    
    data_obj = DataGenerate()
    data_obj.getDataExamples()









