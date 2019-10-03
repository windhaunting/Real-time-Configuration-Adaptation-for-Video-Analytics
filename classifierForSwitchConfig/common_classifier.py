#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:12:00 2019

@author: fubao
"""

# common file for classification of switching config


import sys
import os
import csv

import numpy as np

from glob import glob
from blist import blist

from sklearn.preprocessing import OneHotEncoder

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir3
from profiling.common_prof import frameRates




def load_data_all_features(data_examples_dir, xfile, yfile):
    '''
    data the data for traing and test with all the features, 
    without feature selection
    '''
    
    x_input_arr = np.load(data_examples_dir + xfile)
    y_out_arr = np.load(data_examples_dir + yfile).astype(int)
    
    print ("y_out_arr:",x_input_arr.shape, y_out_arr.shape)

    #y_out_arr = y_out_arr.reshape(-1, 1)
    #print ("y_out_arr before:", np.unique(y_out_arr),  y_out_arr.shape, y_out_arr[:2])

    # reshape to 2-dimen for input
    x_input_arr = x_input_arr.reshape((x_input_arr.shape[0], -1))
    
    #output config to one hot encoder for classification
    #onehot_encoder = OneHotEncoder(sparse=False)
    #y_out_arr = onehot_encoder.fit_transform(y_out_arr)
    
    print ("y_out_arr after:", y_out_arr.shape)
    #return 
    return x_input_arr, y_out_arr          # debug only 1000 first




def feature_selection(data_examples_dir, history_frame_num, max_frame_example_used):
    '''
    transfer input output into a csv and run it in weka 
    '''                 
        
    x_input_arr = np.load(data_examples_dir + "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl")
    y_out_arr = np.load(data_examples_dir + "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl")
    #x_input_arr = x_input_arr.reshape((x_input_arr.shape[0], x_input_arr.shape[1]*x_input_arr.shape[2]))
    y_out_arr = y_out_arr.reshape((y_out_arr.shape[0], 1))

    print ("x_input_arr y_out_arr shape:",x_input_arr.shape, y_out_arr.shape)
    
    data_example_arr= np.hstack((x_input_arr, y_out_arr))
    header_lst = ','.join([str(i) for i in range(0, x_input_arr.shape[1])]) + ',Config'
    np.savetxt( data_examples_dir + "data_example_history_frms1.csv", data_example_arr, delimiter=",", header = header_lst)
    
 

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


def read_config_name_from_file(data_pose_keypoint_dir, write_flag):
    '''
    read config info and order based on resolution*frame rate and then order them in descending order
    and make it a dictionary
    '''
    config_lst = blist()
    # get config_id
    
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*result*.tsv"))  # must read ground truth file(the most expensive config) first
    #resoFrmRate = blist()
    for fileCnt, filePath in enumerate(filePathLst):
        #if '1120x832' in filePath and 'cmu' in filePath:        # neglect the most expensive config as ground truth for caluclating accuracy and resource cost
        #    continue
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
    
    if write_flag:
        pickle_dir = data_pose_keypoint_dir 
        with open(pickle_dir + 'config_to_id.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Config_name", "Id_in_order"])
            for key, value in config_id_dict.items():
                writer.writerow([key, value])
    
    return config_id_dict, id_config_dict



def read_poseEst_conf_frm(data_pickle_dir):
    '''
    read profiling conf's pose of each frame from pickle 
    the pickle file is created from the file "writeIntoPickleConfigFrameAccSPFPoseEst.py"
    
    '''
    
    confg_est_frm_arr = np.load(data_pickle_dir + 'config_estimation_frm.pkl')
    #acc_seg_arr = np.load(data_pickle_dir + file_lst[2])
    #spf_seg_arr = np.load(data_pickle_dir + file_lst[3])
    
    print ("confg_est_frm_arr ", type(confg_est_frm_arr))
    
    return confg_est_frm_arr


def readProfilingResultNumpy(data_pickle_dir):
    '''
    read profiling from pickle
    the pickle file is created from the file "writeIntoPickle.py"
    
    '''
    
    acc_frame_arr = np.load(data_pickle_dir + 'config_acc_frm.pkl')
    spf_frame_arr = np.load(data_pickle_dir + 'config_spf_frm.pkl')
    #acc_seg_arr = np.load(data_pickle_dir + file_lst[2])
    #spf_seg_arr = np.load(data_pickle_dir + file_lst[3])
    
    print ("acc_frame_arr ", type(acc_frame_arr), acc_frame_arr)
    
    return acc_frame_arr, spf_frame_arr


def read_config_name_resolution_frmRate(data_pose_keypoint_dir, write_flag):
    '''
    read config info and order based on resolution and then order them in descending order
    and make it a dictionary
    
    read config info and order based on frame_rate and then order them in descending order
    and make it a dictionary
    
    '''
    
    config_lst = blist()
    # get config_id
    
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    #resoFrmRate = blist()
    for fileCnt, filePath in enumerate(filePathLst):
        #if '1120x832' in filePath and 'cmu' in filePath:        # neglect the most expensive config as ground truth for caluclating accuracy and resource cost
        #    continue
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
        
    
    config_tuple_lst = blist()
    for config in config_lst:
        reso = int(config.split('-')[0].split('x')[1])
        frm_rate = int(config.split('-')[1])
        model = config.split('-')[2]

        config_tuple_lst.append((reso, frm_rate, model))
    
    print ("config_tuple_lst: ", config_tuple_lst)

    config_tuple_lst.sort(key = lambda ele: ele[0], reverse=True)
    
    config_tuple_id_dict = dict(zip(config_tuple_lst,range(0, len(config_tuple_lst))))

    id_config_tuple_dict = dict(zip(range(0, len(config_tuple_lst)), config_tuple_lst))

    print ("config_tuple_dict: ", id_config_tuple_dict)

    if write_flag:
        pickle_dir = data_pose_keypoint_dir 
        with open(pickle_dir + 'config_tuple_to_id.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Config_name", "Id_in_order"])
            for key, value in config_tuple_id_dict.items():
                writer.writerow([key, value])





def read_config_name_resolution_only(data_pose_keypoint_dir, write_flag):
    '''
    read config info and order based on resolution and then order them in descending order
    and make it a dictionary
    
    read config info and order based on frame_rate and then order them in descending order
    and make it a dictionary
    
    '''
    
    config_lst = set()
    # get config_id
    
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    #resoFrmRate = blist()
    for fileCnt, filePath in enumerate(filePathLst):
        #if '1120x832' in filePath and 'cmu' in filePath:        # neglect the most expensive config as ground truth for caluclating accuracy and resource cost
        #    continue
        # get the resolution, frame rate, model
        # input_output/diy_video_dataset/output_006-cardio_condition-20mins/frames_config_result/1120x832_25_cmu_frame_result.tsv
        filename = filePath.split('/')[-1]
        #print ("filename: ", filename)
        reso = filename.split('_')[0]
        #res_right = reso.split('x')[1]
        #frm_rate = filename.split('_')[1]
        
        model = filename.split('_')[2]
                
        #print ("reso: ", reso)
        
        #config_lst += getNewconfig(reso, model)     # add more configs
        config_lst.add((int(reso.split('x')[1])))
    
   
    print ("config_lst: ", config_lst)

    config_lst = list(config_lst)
    config_lst.sort(reverse=True)
    
    config_reso_id_dict = dict(zip(config_lst,range(0, len(config_lst))))

    id_config_reso_dict = dict(zip(range(0, len(config_lst)), config_lst))

    print ("config_reso_id_dict: ", config_reso_id_dict)

    if write_flag:
        pickle_dir = data_pose_keypoint_dir 
        with open(pickle_dir + 'config_to_id_resoOnly.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Config_name", "Id_in_order"])
            for key, value in config_reso_id_dict.items():
                writer.writerow([key, value])
                
    return config_reso_id_dict, id_config_reso_dict





if __name__== "__main__": 

    #data_pose_keypoint_dir =  dataDir2 + 'output_006-cardio_condition-20mins/'

    #read_config_name_from_file(data_pose_keypoint_dir, True)
    #read_config_name_resolution_frmRate(data_pose_keypoint_dir, True) 
    #read_config_name_resolution_only(data_pose_keypoint_dir, True)
    
    
    # feature selection
    #video_dir_lst = ['output_001-dancing-10mins/', 'output_006-cardio_condition-20mins/', 'output_008-Marathon-20mins/']   
    
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/']
    
    for video_dir in video_dir_lst[0:1]: 
        history_frame_num = 1  #1          # 
        max_frame_example_used =  8000 # 20000 #8025   # 8000
        
        data_examples_dir =  dataDir3 + video_dir + 'data_examples_files/'
        #data_examples_dir =  dataDir2 + 'output_006-cardio_condition-20mins/' + 'data_examples_files_resoFR_tuple/'
        #data_examples_dir =  dataDir2 + 'output_006-cardio_condition-20mins/' + 'data_examples_files_resolutionOnly/'
        
        feature_selection(data_examples_dir, history_frame_num, max_frame_example_used)