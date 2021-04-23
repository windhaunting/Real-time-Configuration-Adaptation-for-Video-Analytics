# -*- coding: utf-8 -*-


# online detection

import sys
import os
import cv2
import time

import numpy as np
from glob import glob



from get_data_jumpingNumber_resolution import samplingResoDataGenerate
from get_data_jumpingNumber_resolution import video_dir_lst
from get_data_jumpingNumber_resolution import min_acc_threshold
from get_data_jumpingNumber_resolution import max_jump_number

from prediction_jumpingNumber_resolution import *

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from data_file_process import write_pickle_data
from data_file_process import read_pickle_data
from common_video_process import drawHuman

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/../..')

from classifierForSwitchConfig.common_classifier import read_poseEst_conf_frm_more_dim
from classifierForSwitchConfig.common_classifier import readProfilingResultNumpy

from profiling.common_prof import dataDir3
from profiling.common_prof import PLAYOUT_RATE
from profiling.common_prof import NUM_KEYPOINT   
from profiling.common_prof import resoStrLst_OpenPose

from profiling.common_prof import computeOKS_1to1
from profiling.writeIntoPickleConfigFrameAccSPFPoseEst import read_config_name_from_file

dataDir3 = "../" + dataDir3


class VideoApply(object):
    def __init__(self):
        pass
    
    
    def read_video_frm(self, predicted_video_frm_dir):
        
        imagePathLst = sorted(glob(predicted_video_frm_dir + "*.jpg"))  # , key=lambda filePath: int(filePath.split('/')[-1][filePath.split('/')[-1].find(start)+len(start):filePath.split('/')[-1].rfind(end)]))          # [:75]   5 minutes = 75 segments

        print ("imagePathLst: ", len(imagePathLst))
        
        return imagePathLst
    
    
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
    
    def get_prediction_acc_delay(self, predicted_video_dir, min_acc):
        # get the predicted video's accuracy and predicted_video_dird delay
        #  predicted_video_id as the testing data
        # predicted_video_dir:  such as output_021_dance
        # predicted_out_file is the prediction jumping number and delay
        
        interval_frm = 1
        
        data_pose_keypoint_dir = dataDir3 + predicted_video_dir

        data_pickle_dir = dataDir3 + predicted_video_dir + 'frames_pickle_result/'
        ResoDataGenerateObj = samplingResoDataGenerate()
        intervalFlag = 'frame'
        all_config_flag = False
        config_est_frm_arr, acc_frame_arr, spf_frame_arr = ResoDataGenerateObj.get_data_numpy(data_pose_keypoint_dir, data_pickle_dir, intervalFlag, all_config_flag)
     
        print ("get_prediction_acc_delay config_est_frm_arr: ", config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        # get jumping frm number prediction first
        output_pickle_dir = dataDir3 + predicted_video_dir + "jumping_number_result/" 

        model_dir = output_pickle_dir + "jumpingNumber_resolution_selection/intervalFrm-10_speedType-ema_minAcc-" + str(min_acc) + "/"    
        test_x_instances_file = model_dir + "data_instance_speed_jumpingNumber_resolution_objectSizeRatio_xy.pkl"
        X_test = read_pickle_data(test_x_instances_file)[:, :-2]     # fetch X, excluding the target

        print ("X_test shape: ", X_test.shape)
        pca = PCA(n_components=min(n_components, X_test.shape[1])).fit(X_test)
        
        model_file = model_dir + "model_regression.joblib_exclusive_output_001_dance.pkl"       # with other videos
        model = joblib.load(model_file)

        ModelRegressionObj = ModelRegression()
        
                
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
            #feature_x_object_size = ResoDataGenerateObj.get_object_size_x(current_frm_est)      
            feature_x_object_size = ResoDataGenerateObj.get_object_size_change(current_frm_est, prev_frm_est)   # self.get_object_size_x(current_frm_est)      

            feature_x = np.hstack((feature_x_absolute, feature_x_relative, feature_x_object_size))
                                
            predicted_y = ModelRegressionObj.test_on_data_y_unknown(model, feature_x, pca)

            #print ("get_prediction_acc_delay: ", predicted_y)
            
            jumping_frm_number = int(predicted_y[0][0])
            
            reso_curr = int(predicted_y[0][1])
                        
            
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
        
        detect_out_result_dir = model_dir + "video_applied_detection_result/"
        if not os.path.exists(detect_out_result_dir):
            os.mkdir(detect_out_result_dir)

        arr_acc_segment_file = detect_out_result_dir + "arr_acc_segment_.pkl"
        arr_delay_up_to_segment_file = detect_out_result_dir + "arr_delay_up_to_segment_.pkl"
        write_pickle_data(acc_arr, arr_acc_segment_file)
        write_pickle_data(delay_arr, arr_delay_up_to_segment_file)
     
        return 
    
    def get_estimated_keypoint_by_overall_speed(self, kp_pose_arr, cur_absolute_speed):
        average_speed = [s for s in cur_absolute_speed]   #  [s*(1/PLAYOUT_RATE) for s in cur_absolute_speed]
        
        new_kp_pose_arr = kp_pose_arr + average_speed  # kp_pose_arr + average_speed * kp_pose_arr       # 
        
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
        
        average_all_keypoint_relative_speed = [s for s in all_keypoint_relative_speed]   # [s*(1/PLAYOUT_RATE) for s in all_keypoint_relative_speed]
        #print ("all_keypoint_speed: ",  all_keypoint_relative_speed)
        new_kp_pose_arr =  kp_pose_arr + average_all_keypoint_relative_speed  #  kp_pose_arr + average_all_keypoint_relative_speed * kp_pose_arr
        
        return new_kp_pose_arr
        

    def draw_keypoint_pose_write(self, img_path_lst, img_out_parent_dir, config_est_frm_arr, frm_curr_indx, ans_reso_indx, jumping_frm_number, cur_absolute_speed, arr_ema_relative_speed):
        
        
        resize_wh = resoStrLst_OpenPose[ans_reso_indx]  # resize width x height string
        
        resize_w = int(resize_wh.split("x")[0])
        resize_h = int(resize_wh.split("x")[1])
        
        kp_pose_arr = config_est_frm_arr[ans_reso_indx][frm_curr_indx] 

        #print ("keyPoint_pose_arr: ", kp_pose_arr.shape, kp_pose_arr)
        #print ("cur_absolute_speed: ", cur_absolute_speed.shape, cur_absolute_speed)
        
        next_frm_kp_arr = kp_pose_arr
        start_frm_indx = frm_curr_indx
        
        while (start_frm_indx <= frm_curr_indx + jumping_frm_number):
            
            img_path = img_path_lst[start_frm_indx]
            #img_path = self.get_image_name_id(frm_curr_indx)
            img_name = img_path.split('/')[-1]
            #print ("img_path: ", img_path)
            img = cv2.imread(img_path)
                   
            #print ("frm_curr_indx img resize w h: ", frm_curr_indx, img.shape, resize_w, resize_h)
            
            img = cv2.resize(img, (resize_w, resize_h))
            
            # update next_frm_kp_arr according to estimated moving speed
            
            next_frm_kp_arr = self.get_estimated_keypoint_by_overall_speed(next_frm_kp_arr, cur_absolute_speed)
            next_frm_kp_arr = self.get_estimated_arm_leg_relative_speed(next_frm_kp_arr, arr_ema_relative_speed)
            img = drawHuman(img, next_frm_kp_arr)
            
            cv2.imwrite(img_out_parent_dir + img_name, img)
        
            start_frm_indx += 1
            #print ("next_frm_kp_arr next_frm_kp_arr: ", next_frm_kp_arr)
             
    def get_online_video_analytics_accuracy_spf(self, config_est_frm_arr, spf_frame_arr, frm_curr_indx, ans_reso_indx, jumping_frm_number, cur_absolute_speed, arr_ema_relative_speed):
        
        
        resize_wh = resoStrLst_OpenPose[ans_reso_indx]  # resize width x height string
        
        resize_w = int(resize_wh.split("x")[0])
        resize_h = int(resize_wh.split("x")[1])
        
        kp_pose_arr = config_est_frm_arr[ans_reso_indx][frm_curr_indx] 

        next_frm_kp_arr = kp_pose_arr
        start_frm_indx = frm_curr_indx
        
        average_speed = [s/jumping_frm_number for s in cur_absolute_speed]
        
        gt_start_box = config_est_frm_arr[0][frm_curr_indx] 

        acc_accumulate = computeOKS_1to1(gt_start_box, next_frm_kp_arr)  # 0
        time_accumualate = spf_frame_arr[ans_reso_indx][frm_curr_indx]   # initialized
        calculated_frm_num = 1

        start_time = time.time()
        while (start_frm_indx <= frm_curr_indx + jumping_frm_number):
            
            if start_frm_indx >= config_est_frm_arr.shape[1]:
                break
            #img_path = img_path_lst[start_frm_indx]
            #img_path = self.get_image_name_id(frm_curr_indx)
            #img_name = img_path.split('/')[-1]
           
            #print ("resize img: ", img.shape)
            start_frm_indx += 1

            next_frm_kp_arr = self.get_estimated_keypoint_by_overall_speed(next_frm_kp_arr, cur_absolute_speed)
            next_frm_kp_arr = self.get_estimated_arm_leg_relative_speed(next_frm_kp_arr, arr_ema_relative_speed)
            
            #print ("next_frm_kp_arr cnt: ", config_est_frm_arr[0][frm_curr_indx].shape, next_frm_kp_arr.shape)
            acc_accumulate += computeOKS_1to1(config_est_frm_arr[0][frm_curr_indx] , next_frm_kp_arr)
            #print ("bbbbbbacc_accumulate: ", acc_accumulate)
            calculated_frm_num += 1
            
            
        end_time = time.time()
        time_accumualate += end_time - start_time
        
        #print ("acc_accumulate: ", acc_accumulate, jumping_frm_number)
        
        return acc_accumulate, time_accumualate, calculated_frm_num
    
            
    def show_pose_estimation_detection_result(self, predicted_video_dir, min_acc, max_jump_number, imagePathLst, write_detection_pose_flag):
        #detect speaker after applying the predictive model for configuration adaptation
        #not real apply to video; use profiling results to simulate 
        # get bounding box of the test video in different resolutions (i.e profiling result of the video)
        # then draw on the frame when doing video analytics
        if write_detection_pose_flag:
            img_path = imagePathLst[0]   # get a random image path
            img_out_parent_dir = '/'.join(img_path.split('/')[:-2]) + "/" + img_path.split('/')[-2] + "_boundingbox_out/"
            
            if not os.path.exists(img_out_parent_dir):
                os.mkdir(img_out_parent_dir)
                
        #print ("img_out_parent_dir: ", img_out_parent_dir)
        
        interval_frm = 10
        
        data_pose_keypoint_dir = dataDir3 + predicted_video_dir

        data_pickle_dir = dataDir3 + predicted_video_dir + 'frames_pickle_result/'
        ResoDataGenerateObj = samplingResoDataGenerate()
        intervalFlag = 'frame'
        all_config_flag = False
        config_est_frm_arr, acc_frame_arr, spf_frame_arr = ResoDataGenerateObj.get_data_numpy(data_pose_keypoint_dir, data_pickle_dir, intervalFlag, all_config_flag)
     
        print ("get_prediction_acc_delay config_est_frm_arr: ", config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        output_pickle_dir = dataDir3 + predicted_video_dir + "jumping_number_result/" 
        subDir = output_pickle_dir + "jumpingNumber_resolution_selection/intervalFrm-10_speedType-ema_minAcc-" + str(min_acc) + "/"
        
        
        #model_file = subDir + "model_regression.joblib" + "_exclusive_" + str(predicted_video_dir[:-1])  + ".pkl"       # with other videos
                
        # read the model 
        model_dir = dataDir3 + "dynamic_jumpingNumber_resolution_selection_output/intervalFrm-10_speedType-ema_minAcc-" + str(min_acc) +"/"
        model_file = model_dir + "model_regression_jumping_frm_all.joblib.pkl"    
        
        
        test_x_instances_file = subDir + "data_instance_speed_jumpingNumber_resolution_objectSizeRatio_xy.pkl"
        X_test = read_pickle_data(test_x_instances_file)[:, :-2]     # fetch X, excluding the target

        print ("X_test shape: ", X_test.shape)
        pca = PCA(n_components=min(n_components, X_test.shape[1])).fit(X_test)
        
        model = joblib.load(model_file)

        ModelRegressionObj = ModelRegression()
        # get estimated accuracy and delay
        # start from 2nd frame 
        reso_curr = 0         # current resolution
        prev_indx = 0
        current_indx = 10      # 10 frames as initial window
        FRM_NO =  spf_frame_arr[reso_curr].shape[0]                # total frame no for the whole video

        # use predicted result to apply to this new video and get delay and accuracy
        acc_accumulate = 0.0
        time_spf_accumulate = 0.0
        calculated_frm_num = 0
        
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
            
            feature_x_object_size = ResoDataGenerateObj.get_object_size_x(current_frm_est)      
                    
            feature_x = np.hstack((feature_x_absolute, feature_x_relative, feature_x_object_size))
                    
            #print ("feature_x: ", feature_x.shape)
            predicted_y = ModelRegressionObj.test_on_data_y_unknown(model, feature_x, pca)
            
            
            jumping_frm_number = int(predicted_y[0][0])
            
            reso_curr = int(predicted_y[0][1])
            
            #print ("get_prediction_acc_delay: ", jumping_frm_number, reso_curr)
            
            if jumping_frm_number > max_jump_number:
                jumping_frm_number = max_jump_number
                
            #print ("arr_ema_relative_speed ", arr_ema_relative_speed.shape, arr_ema_relative_speed)
            
            # get accuracy of this segment
            #acc_seg = self.get_accuracy_segment(current_indx, jumping_frm_number, reso_curr, config_est_frm_arr)
            
            # get delay up to this segment
            #up_to_delay = max(0, spf_frame_arr[reso_curr][current_indx] - (1.0/PLAYOUT_RATE) * jumping_frm_number)
            #delay_arr.append(up_to_delay)
            
            #acc_arr.append(acc_seg)
            if write_detection_pose_flag:
                self.draw_keypoint_pose_write(imagePathLst, img_out_parent_dir, config_est_frm_arr, current_indx, reso_curr, jumping_frm_number, arr_ema_absolute_speed, arr_ema_relative_speed)
            
            acc_tmp, time_spf_tmp, calc_frm_num_tmp = self.get_online_video_analytics_accuracy_spf(config_est_frm_arr, spf_frame_arr, current_indx, reso_curr, jumping_frm_number, arr_ema_absolute_speed, arr_ema_relative_speed)
            acc_accumulate += acc_tmp
            time_spf_accumulate += time_spf_tmp
            calculated_frm_num += calc_frm_num_tmp
            
            
            prev_indx = current_indx            # update prev_indx as current index
            
            current_indx += jumping_frm_number         # not jumping_frm_number + 1
            segment_indx += 1
            
        acc_average = acc_accumulate/(calculated_frm_num)
        time_spf = time_spf_accumulate/calculated_frm_num
     
        print ("acc_average, time_spf: ", acc_average, time_spf)
        
        return acc_average, time_spf
  
    def get_acc_spf_under_different_minThreshold(self):
    
        min_acc_threshold_lst = [0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
        acc_lst = []
        SPF_spent_lst = []
        
        max_jump_number = 25 
        for min_acc_thres in min_acc_threshold_lst[0:1]:
                
            acc_average = 0.0
            spf_average = 0.0
            analyzed_video_lst = video_dir_lst[4:5]  # 10
            for predicted_video_dir in analyzed_video_lst:
                    
                predicted_video_frm_dir = dataDir3 + "_".join(predicted_video_dir[:-1].split("_")[1:]) + "_frames/"
                
                print ("predicted_video_frm_dir: ", predicted_video_frm_dir)  # ../input_output/speaker_video_dataset/sample_03_frames/
                write_detection_box_flag = True             # False
                
                if write_detection_box_flag:
                    imagePathLst = videoApplyObj.read_video_frm(predicted_video_frm_dir)
                else:
                    imagePathLst = []
                
                acc, spf = videoApplyObj.show_pose_estimation_detection_result(predicted_video_dir, min_acc_thres, max_jump_number, imagePathLst, write_detection_box_flag)
    
                acc_average += acc
                spf_average += spf
                
            
            acc_lst.append(acc_average/len(analyzed_video_lst))
                
            SPF_spent_lst.append(spf_average/len(analyzed_video_lst))
                
        print("acc_lst, SPF_spent_lst: ", acc_lst, SPF_spent_lst)
          
    
if __name__=="__main__":
    videoApplyObj = VideoApply()
    #videoApplyObj.extract_video_frames()
    
    for predicted_video_dir in video_dir_lst[0:1]:
        
        videoApplyObj.get_prediction_acc_delay(predicted_video_dir, min_acc_threshold)
        
    
    #videoApplyObj.get_acc_spf_under_different_minThreshold()
    