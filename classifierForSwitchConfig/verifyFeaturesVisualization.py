#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:12:55 2019

@author: fubao
"""

# verify the features and the frame to check the display result!

# read the pose estimation's feature in a frame!

import PIL
import re
import cv2
import sys
import math
import os
import pickle
import numpy as np
import time
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
from blist import blist
from collections import defaultdict

from common_classifier import paddingZeroToInter
from common_classifier import read_poseEst_conf_frm
\
from common_plot import plotScatterLineOneFig
from common_plot import plotOneScatterLine
from common_plot import plotOneBar
from common_plot import plotTwoLinesOneFigure


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir3
from profiling.common_prof import PLAYOUT_RATE


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


def verifyDetectedKeyPointTruePosition(data_pickle_dir, data_video_frm_dir, out_data_video_frm):
    arr_est_frm = read_poseEst_conf_frm(data_pickle_dir)
    
    #get one estimation result
    est_res = arr_est_frm[0, 0]       # first frame
    
    imgPath = data_video_frm_dir + "000001.jpg"
    im = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    height, width, layers = im.shape
    lst_points = [(int(float(t[0])*width), int(float(t[1])*height)) for t in re.findall(r'(0(?:\.\d*)?), (0(?:\.\d*)?), ([0123])', est_res)]
    
    print ("lst_points: ", lst_points)
    for pt in lst_points[0:1]:
        cv2.circle(im, pt, 5, (0,255,0), -1)
    img_name = '000001_out.jpg'
    writeStatus = cv2.imwrite(out_data_video_frm + img_name, im)
    if writeStatus is True:
        print("image written")
    else:
        print("problem")    
    

def plotLineInImage(im, point_lst, startX, startY, scaleFactor):
    '''
    startX
    startY
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 0.6

    stX1 = startX
    stY1 = startY
    cv2.putText(im, point_lst[0], (stX1, stY1), font, fontScale, (255, 0, 0), 1, cv2.LINE_AA)
    for i in range(1, len(point_lst)):     # cols
        
        stX2 = stX1 + 100
        stY2 = stY1 - int(scaleFactor*(float(point_lst[i]) - float(point_lst[i-1])))
        cv2.line(im,(stX1, stY1), (stX2, stY2),(0,255,0), 5)
        cv2.putText(im, point_lst[i-1], (stX1, stY1), font, fontScale, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(im, point_lst[i], (stX2, stY2), font, fontScale, (255, 0, 0), 1, cv2.LINE_AA)
        stX1 = stX2
        stY1 = stY2
        startY += 70    
    
    
def readFeatureValues(data_video_frm_dir, x_input_arr, y_out_arr, id_config_dict, acc_frame_arr, spf_frame_arr, arr_feature_frameIndex, out_data_video_frm):
    '''
    look at the feature value
    x_input_arr: feature arr
    arr_feature_frameIndex,   e.g '../input_output/one_person_diy_video_dataset/005_dance_frames/000026.jpg'
    '''

    #read image
    # 005_dance_frames
    #x_input_arr = x_input_arr.reshape((x_input_arr.shape[0], -1))
    rows = x_input_arr.shape[0]
    cols = x_input_arr.shape[1]
    
    last_start_frm_index = int(arr_feature_frameIndex[0].split('/')[-1].split('.')[0])
    curr_start_frm_index = last_start_frm_index
    
    last_im = None
    
    buffer_reso = []      # 
    buffer_frmRate = []
    buffer_acc = []
    buffer_spf = []
    
    max_len_buffer = 10
    
    #correpsonding feature_index 
    lst_absolute_speed_features_index = [18, 20, 30, 32]        #(speed, angle)
    lst_relative_speed_features_index1 = [34+0*2, 34+1*2, 34+2*2, 34+3*2]        #[9, 10, 15, 16, 11] to left hip
    lst_relative_speed_features_index2 = [42+0*2, 42+1*2, 42+2*2, 42+3*2]        #  the right hip.
    lst_relative_speed_features_index3 = [50+0*2, 50+1*2, 50+2*2, 50+3*2]   #  to the left shoulder.
    lst_relative_speed_features_index4 = [58+0*2, 58+1*2, 58+2*2, 58+3*2]   #  to the left shoulder.
    
    lst_relative_distance_index1 = [66+0*2, 66+1*2, 66+2*2, 66+3*2] 
    lst_relative_distance_index2 = [74+0*2, 74+1*2, 74+2*2, 74+3*2] 
    lst_relative_distance_index3 = [82+0*2, 82+1*2, 82+2*2, 82+3*2] 
    lst_relative_distance_index4 = [90+0*2, 90+1*2, 90+2*2, 90+3*2] 
    
    config_id_dict = defaultdict()
    
    for k, v in id_config_dict.items():
        config_id_dict[v] = k
    
    print("config_id_dict: ", config_id_dict) 

    for i in range(0, rows):
        #print("x_input_arr : ", i, )
        # current frame start_path

        curr_start_frm_path= arr_feature_frameIndex[i]
        #img_name = paddingZeroToInter(int(i+1)) + '.jpg'
        #img_path = data_video_frm_dir + img_name
        

        if i == 0:
            curr_start_frm_path= arr_feature_frameIndex[i]

            isFile = os.path.isfile(curr_start_frm_path) 
            print("isFile: ", isFile) 
            print("img_name  shape: ", curr_start_frm_path)
            im = cv2.imread(curr_start_frm_path, cv2.IMREAD_COLOR)
        
            if im is None:
                print ("img read failure" )
                
            height, width, layers = im.shape
            video_name = out_data_video_frm + 'out_video.mp4'        
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
            video = cv2.VideoWriter(video_name, fourcc, 25.0, (width, height))
            
        
        curr_start_frm_index = int(arr_feature_frameIndex[i].split('/')[-1].split('.')[0])
        k = 0
        
        config= id_config_dict[y_out_arr[i]]              # predicted config
        reso = config.split('-')[0].split('x')[1]
        frm_rate = config.split('-')[1]
            
        buffer_reso.append(reso)
        if len(buffer_reso) > max_len_buffer:
            buffer_reso.pop(0)
                
        buffer_frmRate.append(frm_rate)
        if len(buffer_frmRate) > max_len_buffer:
            buffer_frmRate.pop(0)
        
        used_conf_id = y_out_arr[i]   # config_id_dict[config]
        
        acc = str(round(float(acc_frame_arr[used_conf_id][(i+2)*PLAYOUT_RATE]), 3))         # i + 2 not i or i +1 , because of y_out_ predict next in the function "getOnePersonFeatureInputOutputAll001"
        buffer_acc.append(acc)
        if len(buffer_acc) > max_len_buffer:
            buffer_acc.pop(0)    
        #print ("frm_rate: ", frm_rate, acc,  used_conf_id, acc_frame_arr[:, (i+2)*PLAYOUT_RATE])
        
        spf = str(round(float(spf_frame_arr[used_conf_id][(i+2)*PLAYOUT_RATE]), 3))         # i + 2 not i or i +1 , because of y_out_ predict next in the function "getOnePersonFeatureInputOutputAll001"
        buffer_spf.append(spf)
        if len(buffer_spf) > max_len_buffer:
            buffer_spf.pop(0)            
        
        while (k < (curr_start_frm_index - last_start_frm_index)):
            
            start_frm_path_indx = curr_start_frm_index + k
            img_name = paddingZeroToInter(start_frm_path_indx) + '.jpg'
            img_path = data_video_frm_dir + img_name
            im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
            if im is None:
                print ("img read failure22" )
                
            font = cv2.FONT_HERSHEY_SIMPLEX 
            fontScale = 0.6
            startX = 10
            startY = 40
                        
            cv2.putText(im, 'Abso', (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
            startY += 40
            #lst_observePoints = [9, 10, 15, 16]  # leftWrist, rightWrist, leftAnkle, rightAnkle            
            for j in lst_absolute_speed_features_index:     # cols
                strVal = str(round(float(x_input_arr[i-1][j]), 3))          
                cv2.putText(im, strVal, (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                startY += 40
              
                
            startX += 120
            startY = 40
            cv2.putText(im, 'RelvSpd 1', (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
            startY += 40
            
            for j in lst_relative_speed_features_index1:     # cols
                strVal = str(round(float(x_input_arr[i-1][j]), 3))          
                cv2.putText(im, strVal, (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                startY += 40     
                
            ''' 
            startX += 120
            startY = 40
            cv2.putText(im, 'RelvSpd 2', (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
            startY += 40
            
            for j in lst_relative_speed_features_index2:     # cols
                strVal = str(round(float(x_input_arr[i-1][j]), 3))          
                cv2.putText(im, strVal, (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                startY += 40 
                
            
            startX += 120
            startY = 40
            cv2.putText(im, 'RelvSpd 3', (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
            startY += 40
            
            for j in lst_relative_speed_features_index3:     # cols
                strVal = str(round(float(x_input_arr[i-1][j]), 3))          
                cv2.putText(im, strVal, (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                startY += 40 
                
            startX += 120
            startY = 40
            cv2.putText(im, 'RelvSpd 4', (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
            startY += 40
            
            for j in lst_relative_speed_features_index4:     # cols
                strVal = str(round(float(x_input_arr[i-1][j]), 3))          
                cv2.putText(im, strVal, (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                startY += 40 
                
               
            startX += 120
            startY = 40
            cv2.putText(im, 'RelvDist 1', (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
            startY += 40
            
            for j in lst_relative_distance_index1:     # cols
                strVal = str(round(float(x_input_arr[i-1][j]), 3))          
                cv2.putText(im, strVal, (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                startY += 40 
                
            startX += 120
            startY = 40
            cv2.putText(im, 'RelvDist 2', (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
            startY += 40
            
            for j in lst_relative_distance_index2:     # cols
                strVal = str(round(float(x_input_arr[i-1][j]), 3))          
                cv2.putText(im, strVal, (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                startY += 40 
                
            startX += 120
            startY = 40
            cv2.putText(im, 'RelvDist 3', (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
            startY += 40
            
            for j in lst_relative_distance_index3:     # cols
                strVal = str(round(float(x_input_arr[i-1][j]), 3))          
                cv2.putText(im, strVal, (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                startY += 40 

            startX += 120
            startY = 40
            cv2.putText(im, 'RelvDist 4', (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
            startY += 40
            
            for j in lst_relative_distance_index4:     # cols
                strVal = str(round(float(x_input_arr[i-1][j]), 3))          
                cv2.putText(im, strVal, (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                startY += 40                 
            
            
            startX += 120
            startY = 40
            cv2.putText(im, 'Blurry', (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
            startY += 80
            
            strVal = str(round(float(x_input_arr[i-1][135]), 3))          
            cv2.putText(im, strVal, (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
                
            '''
                
            '''                                      
            startX = 200
            startY = 40
            for res in buffer_reso:
                cv2.putText(im, res + ',' , (startX, startY), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                startX += 70
            
            startX = 200
            startY += 60
            for frmRt in buffer_frmRate:
                cv2.putText(im, frmRt+ ',' , (startX, startY), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                startX += 70
            '''
            startX = 20
            startY = 200
            cv2.putText(im, 'Reso', (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
            
            startX = 100
            startY = 200
            scaleFactor = 0.13
            plotLineInImage(im, buffer_reso[:-1], startX, startY, scaleFactor)
            
            startX = 20
            startY = 300
            cv2.putText(im, 'Frm_Rt', (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
       
            startX = 100
            startY = 300
            scaleFactor = 3
            plotLineInImage(im, buffer_frmRate[:-1], startX, startY, scaleFactor)
            
            
            startX = 20
            startY = 400
            cv2.putText(im, 'OKS', (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
       
            startX = 100
            startY = 400
            scaleFactor = 1000
            #print("XXXX  buffer_acc: ", buffer_acc)

            plotLineInImage(im, buffer_acc[:-1], startX, startY, scaleFactor)
            
            startX = 20
            startY = 500
            cv2.putText(im, 'SPF', (startX, startY), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
       
            startX = 100
            startY = 500
            scaleFactor = 1000
            #print("XXXX  buffer_acc: ", buffer_acc)

            plotLineInImage(im, buffer_spf[:-1], startX, startY, scaleFactor)
            
            video.write(im)
            
            
            # write the extra frame curr_start_frm_index + k
            
            '''
            img_name = curr_start_frm_path.split('/')[-1]
            print("image img_nameimg_name", type(img_name), img_name)
            writeStatus = cv2.imwrite(out_data_video_frm + img_name, im)
            if writeStatus is True:
                print("image written")
            else:
                print("problem")        
                
            print("out_data_video_frm + img_name: ", out_data_video_frm + img_name)
            
            print("image arr_feature_frameIndex[i]", arr_feature_frameIndex[i])
            '''
            #print ("k: ", k)
            k += 1
            
        
        #last_im = im              
        last_start_frm_index = curr_start_frm_index
        
        #print("image curr_start_frm_index", curr_start_frm_index)
        
        
        if i == 7*60: # 7*60:
            break   # debug only
    
    print("readFeatureValues finished ")
    cv2.destroyAllWindows()
    video.release()

def exectuteVisualization():
    '''
    '''
    
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/']
        
    
        
    input_video_frms_dir = ['001_dance_frames/', '002_dance_frames/', \
                        '003_dance_frames/', '004_dance_frames/',  \
                        '005_dance_frames/', '006_yoga_frames/', \
                        '007_yoga_frames/', '008_cardio_frames/',
                        '009_cardio_frames/', '010_cardio_frames/',
                        '011_dance_frames/', '012_dance_frames/',
                        '013_dance_frames/', '014_dance_frames/',
                        '015_dance_frames/', '016_dance_frames/',
                        '017_dance_frames/', '018_dance_frames/',
                        '019_dance_frames/', '020_dance_frames/',
                        '021_dance_frames/']

    
    feature_calculation_flag = 'most_expensive_config'
    
    for i, video_dir in enumerate(video_dir_lst):
        
        if i  != 4:                    # check the 005_video only
            continue
        
        '''
        data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
        data_video_frm_dir = dataDir3 + '005_dance_frames/'
        out_data_video_frm = dataDir3 + video_dir + 'verify_frm_features/'
        verifyDetectedKeyPointTruePosition(data_pickle_dir, data_video_frm_dir, out_data_video_frm)
        
        '''
        
        data_pose_keypoint_dir =  dataDir3 + video_dir
        
        data_frame_path_dir = dataDir3 + input_video_frms_dir[i]
        
        history_frame_num = 1       #1
        max_frame_example_used = 12000  # 20000 #8025   # 10000
        
        data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
        minAccuracy = 0.95
    
        if feature_calculation_flag == 'most_expensive_config':
            from data_proc_feature_analysize_01 import getOnePersonFeatureInputOutputAll001

        x_input_arr, y_out_arr, id_config_dict, acc_frame_arr, spf_frame_arr = getOnePersonFeatureInputOutputAll001(data_pose_keypoint_dir, data_pickle_dir, data_frame_path_dir, history_frame_num, max_frame_example_used, minAccuracy)
     
        
        print("x_input_arr y_out_arr shape: ", x_input_arr.shape, y_out_arr.shape)
    
        
        out_data_video_frm = data_pose_keypoint_dir + 'verify_frm_features/'
        if not os.path.exists(out_data_video_frm):
            os.mkdir(out_data_video_frm)
        
        arr_feature_frameIndex = x_input_arr[:, 0]
        x_input_arr = x_input_arr[:, 1:]
        readFeatureValues(data_frame_path_dir, x_input_arr, y_out_arr, id_config_dict, acc_frame_arr, spf_frame_arr, arr_feature_frameIndex, out_data_video_frm)
        
    
if __name__== "__main__": 
    
    
    
    exectuteVisualization()
    
    
    
    
    
    
    
    
    
    
    