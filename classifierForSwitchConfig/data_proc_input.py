#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:33:14 2019

@author: fubao
"""

# input_data_transfer; read pose estimation result and transfer
# into the basic format of numpy


#each image is a 5 x 18 points array

import sys
import os
import csv
import numpy as np
import pandas as pd

from glob import glob
from blist import blist


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir2
from profiling.common_prof import frameRates

def getNewconfig(reso, model):
    '''
    get new config from all available frames
    the config file name has only frame rate 25;  1 frame do not have frame rate 
    so we get more models from frames
    '''
    config_lst = blist()
    for frmRt in frameRates:
        config_lst.append(reso + '-' + str(frmRt) + '-' + model)
        
    return config_lst


def read_config_name_from_file(data_pose_keypoint_dir):
    '''
    read config info and order based on resolution*frame rate and then order them in descending order
    and make it a dictionary
    '''
    config_lst = blist()
    # get config_id
    
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    #resoFrmRate = blist()
    for fileCnt, filePath in enumerate(filePathLst):
        # get the resolution, frame rate, model
        # input_output/diy_video_dataset/output_006-cardio_condition-20mins/frames_config_result/1120x832_25_cmu_frame_result.tsv
        filename = filePath.split('/')[-1]
        #print ("filename: ", filename)
        reso = filename.split('_')[0]
        #res_right = reso.split('x')[1]
        #frm_rate = filename.split('_')[1]
        
        model = filename.split('_')[2]
                
        #print ("reso: ", reso)
        
        config_lst += getNewconfig(reso, model)     # add more configs
        
        #resoFrmRate.append(res_frame_multiply)  random.sort(key=lambda e: e[1])

        
    #model_resoFrm_dict = dict(zip(config_lst, resoFrmRate))
    #sort by resolution*frame_rate  e.g. 720px25
    config_lst.sort(key = lambda ele: int(ele.split('-')[0].split('x')[1])* int(ele.split('-')[1]), reverse=True)
    config_id_dict = dict(zip(config_lst,range(0, len(config_lst))))
        
    id_config_dict = dict(zip(range(0, len(config_lst)), config_lst))

    #print ("model_resoFrm_dict: ", id_config_dict, len(id_config_dict), config_id_dict)
    
    pickle_dir = data_pose_keypoint_dir 
    with open(pickle_dir + 'config_to_id.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Config_name", "Id_in_order"])
        for key, value in config_id_dict.items():
            writer.writerow([key, value])

    
def readPoseEstimationKeyPoint(data_pose_keypoint_dir):
    '''
    read into numpy
    each is te 5 x 18 dimension values calculated from key point estimation results of multi persons
              frame_id
    config_Id  ...
    
    
    '''
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    confg_frm_spf_arr
    for fileCnt, filePath in enumerate(filePathLst):
        print ("file ",  file)
        #parse each images 
        df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)         # det-> detection
        

        print ("numy shape: ", confg_frm_spf_arr.shape, filePath)
        
        for index, row in df_det.iterrows():  
            #print ("index, row: ", index, row)
            reso = row['Resolution']
            #frm_rate = row['Frame_rate']
            model = row['Model']
            time_spf = row['Time_SPF']
            acc = row['Acc']
            frm_id = int(row['Image_path'].split('/')[-1].split('.')[0])
            
        
        


    
def readPoseEstimationAccSPFPickle(data_pickle_dir):
    '''
    Each pickle
    x = 1
    '''
    
    acc_frame_arr = np.load(data_pickle_dir + file_lst[0])
    spf_frame_arr = np.load(data_pickle_dir + file_lst[1])
    #acc_seg_arr = np.load(data_pickle_dir + file_lst[2])
    #spf_seg_arr = np.load(data_pickle_dir + file_lst[3])
    
    print ("acc_frame_arr ", type(acc_frame_arr), acc_frame_arr)
    
    return acc_frame_arr, spf_frame_arr
    

if __name__== "__main__": 
        
    data_pose_keypoint_dir =  dataDir2 + 'output_006-cardio_condition-20mins/'
    
    #read_config_name_from_file(data_pose_keypoint_dir)
    
    readPoseEstimationKeyPoint(data_pose_keypoint_dir)