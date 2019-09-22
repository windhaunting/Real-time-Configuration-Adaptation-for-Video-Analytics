#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:31:00 2019

@author: fubao
"""




#scp -P 5122 -r ~/workDir/video_analytics_pose_estimation/profiling/*.py fubao@ipanema.ecs.umass.edu:/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/videoAnalytics_poseEstimation/profiling/


# combine the functions of segmentProcess.py and writeIntoPickle.py two files together


import os
import pandas as pd
import numpy as np

import sys
import csv

import pickle

from blist import blist

from glob import glob

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur)

from common_prof import dataDir2
from common_prof import frameRates
from common_prof import computeOKSAP


'''
def modelToInt(model):
    if model == 'cmu':
        return 0
    elif model == 'a':
        return 2
    else: 
        return 1

'''


def getNewconfig(reso, model):
    '''
    get new config from all available frames
    the config file name has only frame rate 25;  1 frame does not have frame rate 
    so we get more models from frames
    '''
    config_lst = blist()
    for frmRt in frameRates:
        config_lst.append(reso + '-' + str(frmRt) + '-' + model)
        
    return config_lst


def getconfigSPFEachFrm(reso, model, time_spf, config_id_dict, index, confg_frm_spf_arr):
    '''
    frame-by-frame consideration
    get new config's tiem_spf for the current frame ,  because the input is for PLAYOUT rate 25
    it's frame-by-frame, so the acc and spf are all the same
    it actually considers only frame_rate which is 25
    '''
    for frmRt in frameRates:
        #frmInter = math.ceil(PLAYOUT_RATE/frmRt)          # frame rate sampling frames in interval, +1 every other
        
        new_spf = time_spf         # time_spf/frmInter is not correct
        
        config = reso + '-' + str(frmRt) + '-' + model.split('_')[0]
        #config = (int(reso.split('x')[1]), int(frmRt), modelToInt(model.split('_')[0]))   #
        
        #config = int(reso.split('x')[1])   #
        
        cfg_id = config_id_dict[config]
        
        confg_frm_spf_arr[cfg_id, index] = new_spf
        #confg_frm_acc_arr[cfg_id, index] = acc
        #print ("cfg_id: ", cfg_id)
    #print ("confg_frm_acc_arr 2222: ",  confg_frm_acc_arr)
    return confg_frm_spf_arr


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




def write_config_frm_poseEst_result(data_pose_keypoint_dir, data_pickle_dir):
    '''
    frame-by-frame consideration
    get config's estimation result based on each config and frame
    and the spf result 
    '''
    
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, True)
    return

    filePathLst = sorted(glob(data_pose_keypoint_dir + "*estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    config_num = len(config_id_dict)  # len(config_id_dict)
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection
    frame_num = len(df_det)     #  maybe some frame_id is missing, only consider all frames that could be parsed from a video
    #create a numpy array
    confg_frm_est_arr = np.zeros((config_num, frame_num), dtype=object) # array of estimation result with config vs frame_Id
    
    confg_frm_spf_arr = np.zeros((config_num, frame_num)) # array of time_spf with config vs frame_Id
    
    print ("config_id_dict: ", config_id_dict,  len(config_id_dict), frame_num)
    
    for fileCnt, filePath in enumerate(filePathLst):
        df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)         # det-> detection
        

        print ("numy shape: ", confg_frm_est_arr.shape, filePath)
        
        for index, row in df_det.iterrows():  
            #print ("index, row: ", index, row)
            reso = row['Resolution']
            #frm_rate = row['Frame_rate']
            model = row['Model'].split('_')[0]
            #frm_id = int(row['Image_path'].split('/')[-1].split('.')[0])
            est_res = row['Estimation_result']
            time_spf = row['Time_SPF']
            #print ("est_ressssss: ", est_res)
            #get the config index
            config_lst = getNewconfig(reso, model)     # add more configs
            for config in config_lst:
                id_conf = config_id_dict[config]  
                confg_frm_est_arr[id_conf, index] = est_res    
                
            
            #print ("confg_frm_est_arr: ", str(confg_frm_est_arr[id_conf,index]))
            confg_frm_spf_arr = getconfigSPFEachFrm(reso, model, time_spf, config_id_dict, index, confg_frm_spf_arr)
            
            #break    # test only
        
        #break    # test only

    print ("confg_frm_est_arr: ", confg_frm_est_arr.shape, confg_frm_est_arr[5][0], confg_frm_spf_arr[0][0])            

    with open(data_pickle_dir + 'config_estimation_frm.pkl','wb') as fs:
        pickle.dump(confg_frm_est_arr, fs)
      
    #out_frm_spf_pickle_file = pickle_dir + "spf_frame.pkl"      # spf for config vs each frame
    with open(data_pickle_dir + 'config_spf_frm.pkl','wb') as fs:
        pickle.dump(confg_frm_spf_arr, fs)

    return



def readConfigFrmEstFile(data_pickle_dir):
    '''
    read confg_frm_est_arr from the output of write_config_frm_poseEst_result
    '''
    
    #pickle_dir = dataDir2 + 'output_006-cardio_condition-20mins/' + 'pickle_files/'
    
    pickleFile = data_pickle_dir + 'config_estimation_frm.pkl'
    confg_frm_est_arr = np.load(pickleFile)

    #print ("confg_frm_est_arr: ", (confg_frm_est_arr[:, 0]))
    #return
    return confg_frm_est_arr


def apply_acc_fun(arrs):
    
    #print ("commmmmmm: ", type((arrs[0])), arrs[0])
    #print ("commmmmmm22: ", str(arrs[0]) == 'nan')
    #print ("commmmmmm33: ", np.isnan(arrs[0]))

    if str(arrs[0]) == 'nan' and str(arrs[1]) == 'nan':
        return 1.0
    elif str(arrs[0]) == 'nan' and str(arrs[1]) != 'nan':
        return 0.0
    elif str(arrs[0]) != 'nan' and str(arrs[1]) == 'nan':
        return 0.0
    acc = computeOKSAP(str(arrs[0]), str(arrs[1]), '')
    return acc
    
def calculate_config_frm_acc(ests_arr, gts_arr):
    '''
    each input it's the array of all frames
    '''
    
    #print ("ests_arr shape: ", ests_arr.shape, gts_arr.shape)
    #combine together
    combine_arr = np.vstack((ests_arr, gts_arr))
    
    #print ("combine_arr shape: ", combine_arr.shape)
    acc_arr = np.apply_along_axis(apply_acc_fun, 0, combine_arr)
    
    print ("acc_arr: ", acc_arr.shape, acc_arr)
    
    return acc_arr
    
def write_config_frm_acc_result(confg_frm_est_arr, data_pickle_out_dir):
    '''
    frame-by-frame consideration
    get each config acc for each frame, frame rate is actually 25.
    '''
    
    # select the ground truth used config id, here it's a fixed number
    # so id is 0, that is it's the first line
    
    gts_arr = confg_frm_est_arr[0]  #ground truth for each frame    1120*83-25-cmu
    
    config_frm_acc_arr = np.apply_along_axis(calculate_config_frm_acc, 1, confg_frm_est_arr, gts_arr)
    
    print ("config_frm_acc_arr1 final: ", config_frm_acc_arr.shape, config_frm_acc_arr[0])
    print ("config_frm_acc_arr2 final: ", config_frm_acc_arr)
    
    out_frm_acc_pickle_file = data_pickle_out_dir + "config_acc_frame.pkl"      # acc for config vs each frame
    
    with open(out_frm_acc_pickle_file,'wb') as fs:
        pickle.dump(config_frm_acc_arr, fs)   
    

def readConfigFrmSPfFile(data_pickle_dir):
    '''
    read confg_frm_est_arr from the output of write_config_frm_poseEst_result
    '''
    
    #pickle_dir = dataDir2 + 'output_006-cardio_condition-20mins/' + 'pickle_files/'
    
    pickleFile = data_pickle_dir + 'config_spf_frm.pkl'
    confg_frm_spf_arr = np.load(pickleFile)

    #print ("confg_frm_est_arr: ", (confg_frm_est_arr[:, 0]))
    #return
    return confg_frm_spf_arr


def executeWriteIntoPickle():
    
    video_dir_lst = ['output_001-dancing-10mins/', 'output_002-video_soccer-20mins/', 
                     'output_003-bike_race-10mins/', 'output_006-cardio_condition-20mins/',
                     'output_008-Marathon-20mins/'
                     ]
    
    for vd_dir in video_dir_lst[0:1]:
        
        data_pickle_dir = dataDir2 +  vd_dir + 'frames_pickle_result/'
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)
            
            
        #write_config_frm_poseEst_result(dataDir2 +  vd_dir, data_pickle_dir)
        '''
        confg_frm_est_arr = readConfigFrmEstFile(data_pickle_dir)
        #write_config_frm_acc_result(confg_frm_est_arr, data_pickle_dir)

        #write segment interval 1, 2,..., n seconds
        segment = 1     # 1 second
        confg_frm_est_arr = readConfigFrmEstFile(data_pickle_dir)
        confg_frm_spf_arr = readConfigFrmSPfFile(data_pickle_dir)
        
        write_config_seg_acc_spf_result(confg_frm_est_arr, confg_frm_spf_arr)
        '''

if __name__== "__main__":
    
    executeWriteIntoPickle()