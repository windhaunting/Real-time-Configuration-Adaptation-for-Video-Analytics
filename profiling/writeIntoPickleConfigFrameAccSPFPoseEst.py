#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:31:00 2019
@author: fubao
"""




#scp -P 5122 -r ~/workDir/video_analytics_pose_estimation/profiling/*.py fubao@ipanema.ecs.umass.edu:/home/fubao/workDir/ResearchProjects/IOTVideoAnalysis/videoAnalytics_poseEstimation/profiling/


# combine the functions of segmentProcess.py and writeIntoPickle.py two files together

import re
import os
import math
import pandas as pd
import numpy as np
import time

import sys
import csv

import pickle

from blist import blist
from collections import defaultdict
from glob import glob

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur)

#sfrom common_prof import dataDir2
from common_prof import  dataDir3
from common_prof import frameRates
from common_prof import PLAYOUT_RATE
from common_prof import computeOKSAP
from common_prof import computeOKSFromOrigin


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
        frmInter = math.ceil(PLAYOUT_RATE/frmRt)          # frame rate sampling frames in interval, +1 every other
        
        new_spf = time_spf/frmInter         # time_spf is not correct
        
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




def write_config_frm_poseEst_result(data_pose_keypoint_dir, data_pickle_dir, start_frm_index):
    '''
    frame-by-frame consideration
    get config's estimation result based on each config and frame
    and the spf result 
    start_frm_index is the new start frame index that for getting the pickle matrix
    '''
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)
    

    filePathLst = sorted(glob(data_pose_keypoint_dir + "*estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    config_num = len(config_id_dict)  # len(config_id_dict)
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection
    frame_num = len(df_det)-start_frm_index     #  maybe some frame_id is missing, only consider all frames that could be parsed from a video
    #create a numpy array
    confg_frm_est_arr = np.zeros((config_num, frame_num), dtype=object) # array of estimation result with config vs frame_Id
    
    confg_frm_spf_arr = np.zeros((config_num, frame_num)) # array of time_spf with config vs frame_Id
    
    print ("config_id_dict: ", config_id_dict,  len(config_id_dict), frame_num)
    
    for fileCnt, filePath in enumerate(filePathLst):
        df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)             # det-> detection
        
        print ("numy shape: ", confg_frm_est_arr.shape, filePath)
        start_index = 0
        for index, row in df_det.iloc[start_frm_index:].iterrows():  
            
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
                confg_frm_est_arr[id_conf, start_index] = est_res    
                
            
            #print ("confg_frm_est_arr: ", str(confg_frm_est_arr[id_conf,index]))
            confg_frm_spf_arr = getconfigSPFEachFrm(reso, model, time_spf, config_id_dict, start_index, confg_frm_spf_arr)
            
            
            start_index += 1
            #break    # test only
        
        #break    # test only

    print ("confg_frm_est_arr: ", confg_frm_est_arr.shape, confg_frm_est_arr[5][0], confg_frm_spf_arr[0][0])            

    if start_frm_index == 0:
        with open(data_pickle_dir + 'config_estimation_frm.pkl','wb') as fs:
            pickle.dump(confg_frm_est_arr, fs)
          
        #out_frm_spf_pickle_file = pickle_dir + "spf_frame.pkl"      # spf for config vs each frame
        with open(data_pickle_dir + 'config_spf_frm.pkl','wb') as fs:
            pickle.dump(confg_frm_spf_arr, fs)
            
        return data_pickle_dir
    else:
        data_pickle_out_parent_dir = "/".join(data_pose_keypoint_dir.split("/")[:-2]) + "/" + data_pose_keypoint_dir.split("/")[-2] + "-start-"+ str(start_frm_index)
        
        if not os.path.exists(data_pickle_out_parent_dir):
            os.mkdir(data_pickle_out_parent_dir)
            
        data_pickle_out_dir = data_pickle_out_parent_dir + "/frames_pickle_result/"
        if not os.path.exists(data_pickle_out_dir):
            os.mkdir(data_pickle_out_dir)    
        
        with open(data_pickle_out_dir + 'config_estimation_frm.pkl','wb') as fs:
            pickle.dump(confg_frm_est_arr, fs)
          
        #out_frm_spf_pickle_file = pickle_dir + "spf_frame.pkl"      # spf for config vs each frame
        with open(data_pickle_out_dir + 'config_spf_frm.pkl','wb') as fs:
            pickle.dump(confg_frm_spf_arr, fs)
            
    return data_pickle_out_dir



def readConfigFrmEstFile(data_pickle_dir):
    '''
    read confg_frm_est_arr from the output of write_config_frm_poseEst_result
    '''
    
    #pickle_dir = dataDir2 + 'output_006-cardio_condition-20mins/' + 'pickle_files/'
    
    pickleFile = data_pickle_dir + 'config_estimation_frm.pkl'
    confg_frm_est_arr = np.load(pickleFile, allow_pickle=True)

    #print ("confg_frm_est_arr: ", (confg_frm_est_arr[:, 0]))
    #return
    return confg_frm_est_arr

def getPredictedPoseEst(conf_arr_origin, lst_indx):
    #use the predicted actual to replace the undetected frame
    #the maximum is PLAYOUT_RATE
    # input : estimation 1x25   config x 25 frames estimation
    # output: 1x25 estimation      
    conf_arr_new = np.zeros(conf_arr_origin.shape, dtype=object)
    #print ("conf_arr_origin dimen: ", conf_arr_origin)
    
    current_pose = conf_arr_origin[0]
    for ind in range(0, conf_arr_origin.shape[0]):
        
        conf_arr_new[ind] = current_pose
        if (ind+1) in lst_indx:
            current_pose = conf_arr_origin[ind+1]
            
    return conf_arr_new
    
    
def write_config_frm_acc_result(data_pose_keypoint_dir, confg_frm_est_arr, data_pickle_out_dir):
    #frame-by-frame consideration
    #get each config acc for each frame, use actual pose estimation to replace the undetected frames 
    # when calculating low frame rate
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)


    frm_extracted_index_dict = defaultdict()
    
    for frmRt in frameRates:
        frmInter = math.ceil(PLAYOUT_RATE/frmRt)          # frame rate sampling frames in interval, +1 every other
        lst_indx = range(0, PLAYOUT_RATE, frmInter)
        
        print ("lst_indx: ", lst_indx, len(lst_indx))
        frm_extracted_index_dict[frmRt] = lst_indx

    # row [0] is ground truth
    gts_arr = confg_frm_est_arr[0]  #ground truth for each frame    1120*83-25-cmu
    print ("confg_frm_est_arr dimen: ",confg_frm_est_arr.shape)
        
    confg_frm_acc_arr = np.zeros((confg_frm_est_arr.shape[0], confg_frm_est_arr.shape[1]-PLAYOUT_RATE))
    for row_ind in range(0, confg_frm_est_arr.shape[0]):
        config = id_config_dict[row_ind]
        print('config, row_ind: ', config, row_ind)
        frmRt = int(config.split('-')[1])
        lst_indx = frm_extracted_index_dict[frmRt]
        for col_ind in range(0, confg_frm_est_arr.shape[1]-PLAYOUT_RATE):
            #print('config, col_ind: ', config, col_ind, confg_frm_est_arr[row_ind][col_ind])
            # construct 25 frames result
            conf_arr_origin = confg_frm_est_arr[row_ind][col_ind:col_ind+PLAYOUT_RATE]
            conf_arr_new = getPredictedPoseEst(conf_arr_origin, lst_indx)
            #print ("conf_arr_new dimen: ",lst_indx, conf_arr_new)
            
            gts_arr = confg_frm_est_arr[0][col_ind:col_ind+PLAYOUT_RATE] 
            
            #calculate the accuracy
            acc = np.mean(calculate_config_frm_acc(conf_arr_new, gts_arr))
            #print ("acc_arr: ",config, acc_arr.shape, acc_arr)
            confg_frm_acc_arr[row_ind][col_ind] = acc
            

    print('config, ddd: ', config, col_ind, confg_frm_acc_arr.shape)

    out_frm_acc_pickle_file = data_pickle_out_dir + "config_acc_frm.pkl"      # acc for config vs each frame
    
    with open(out_frm_acc_pickle_file,'wb') as fs:
        pickle.dump(confg_frm_acc_arr, fs)   
        
        
def write_config_frm_acc_oks_result_interval(data_pose_keypoint_dir, confg_frm_est_arr, data_pickle_out_dir, interval, metric):
    #1second interval consideration to calculate oks or acc
    #get each config acc for each frame, use actual pose estimation to replace the undetected frames 
    # when calculating low frame rate
    # metric, us oks or acc
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)


    frm_extracted_index_dict = defaultdict()
    
    for frmRt in frameRates: 
        frmInter = math.ceil(PLAYOUT_RATE/frmRt)          # frame rate sampling frames in interval, +1 every other
        lst_indx = range(0, PLAYOUT_RATE, frmInter)
        
        print ("lst_indx: ", lst_indx, len(lst_indx))
        frm_extracted_index_dict[frmRt] = lst_indx

    # row [0] is ground truth
    gts_arr = confg_frm_est_arr[0]  #ground truth for each frame    1120*83-25-cmu
    print ("confg_frm_est_arr dimen: ",confg_frm_est_arr.shape)
        
    confg_frm_acc_arr = np.zeros((confg_frm_est_arr.shape[0], confg_frm_est_arr.shape[1]-PLAYOUT_RATE)) # oks or acc
    for row_ind in range(0, confg_frm_est_arr.shape[0]):
        config = id_config_dict[row_ind]
        print('config, row_ind: ', config, row_ind)
        frmRt = int(config.split('-')[1])
        lst_indx = frm_extracted_index_dict[frmRt]
        for col_ind in range(0, confg_frm_est_arr.shape[1]-PLAYOUT_RATE, PLAYOUT_RATE):
            #print('config, col_ind: ', config, col_ind, confg_frm_est_arr[row_ind][col_ind])
            # construct 25 frames result
            conf_arr_origin = confg_frm_est_arr[row_ind][col_ind:col_ind+PLAYOUT_RATE]
            conf_arr_new = getPredictedPoseEst(conf_arr_origin, lst_indx)
            #print ("conf_arr_new dimen: ",lst_indx, conf_arr_new)
            
            gts_arr = confg_frm_est_arr[0][col_ind:col_ind+PLAYOUT_RATE] 
            
            #calculate the accuracy
            acc = np.mean(calculate_config_frm_acc(conf_arr_new, gts_arr, metric))
            #print ("acc_arr: ",config, acc_arr.shape, acc_arr)
            confg_frm_acc_arr[row_ind][col_ind] = acc
            

    print('config, ddd: ', config, col_ind, confg_frm_acc_arr.shape)
    
    out_frm_acc_pickle_file = data_pickle_out_dir + "config_" + metric + "_interval_" + str(interval) +"sec.pkl"      # acc for config vs each frame
        
    with open(out_frm_acc_pickle_file,'wb') as fs:
        pickle.dump(confg_frm_acc_arr, fs)   
        
        

def getOnePersonEstimation(ests_arr):
    '''
   use only one person select the confidence score higher
    '''
    #print ("ests_arr ests_arr: ", ests_arr)
    
    strLst = re.findall(r'],\d.\d+', ests_arr)
    person_lst = [re.findall(r'\d.\d+', st) for st in strLst]
    
    ind = np.argmax(person_lst)
    #print ("ind: ", person_lst, ind)
    
    return ests_arr.split(';')[ind]


def apply_acc_fun(arrs, metric):
    
    #print ("commmmmmm: ", type((arrs[0])), arrs[0])
    #print ("commmmmmm22: ", str(arrs[0]) == 'nan')
    #print ("commmmmmm33: ", np.isnan(arrs[0]))

    if str(arrs[0]) == 'nan' and str(arrs[1]) == 'nan':
        return 1.0
    elif str(arrs[0]) == 'nan' and str(arrs[1]) != 'nan':
        return 0.0
    elif str(arrs[0]) != 'nan' and str(arrs[1]) == 'nan':
        return 0.0
    elif str(arrs[0]) == '' or str(arrs[1]) == '':
        return 0.0
    elif str(arrs[0]) == '0' or str(arrs[1]) == '0':        # no pose estimation detected for the low config
        return 0.0
    
    arr0 = getOnePersonEstimation(arrs[0])
    arr1 = getOnePersonEstimation(arrs[1])
    
    if metric == 'acc':
        acc = computeOKSAP(arr0, arr1, '')
    elif metric == 'oks':
        acc = computeOKSFromOrigin(arr0, arr1, '')
        
    return acc
    
def calculate_config_frm_acc(ests_arr, gts_arr, metric):
    #each input it's the array of all frames

    
    #print ("ests_arr shape: ", ests_arr.shape, gts_arr.shape)
    #combine together
    combine_arr = np.vstack((ests_arr, gts_arr))
    
    #print ("combine_arr shape: ", combine_arr.shape)
    acc_arr = np.apply_along_axis(apply_acc_fun, 0, combine_arr, metric)
    
    #print ("acc_arr: ", acc_arr.shape, acc_arr)
    
    return acc_arr


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



def extendMoreDataSamplingWriteIntoPickle():
        video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/', \
                    'output_011_dance/', 'output_012_dance/']
        
        for vd_dir in video_dir_lst[1:2]:        # [3:4]:   # [0:1]:
        
            data_pickle_dir = dataDir3 +  vd_dir + 'frames_pickle_result/'
            if not os.path.exists(data_pickle_dir):
                os.mkdir(data_pickle_dir)
            
            data_pose_keypoint_dir = dataDir3 +  vd_dir
            
            start_frm_index = 280        # 10 frames
            while(start_frm_index < 1000):
                data_pickle_out_dir = write_config_frm_poseEst_result(data_pose_keypoint_dir, data_pickle_dir, start_frm_index)        
    
                confg_frm_est_arr = readConfigFrmEstFile(data_pickle_out_dir)
        
                interval = 1       # 1 sec
                metric = 'oks'
                write_config_frm_acc_oks_result_interval(data_pose_keypoint_dir, confg_frm_est_arr, data_pickle_out_dir, interval, metric)
               
                start_frm_index += 10
                
def executeWriteIntoPickleOnePeron():
    
    
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/', \
                    'output_011_dance/', 'output_012_dance/']
    
    
    for vd_dir in video_dir_lst[0:1]:        # [3:4]:   # [0:1]:
        
        data_pickle_dir = dataDir3 +  vd_dir + 'frames_pickle_result/'
        if not os.path.exists(data_pickle_dir):
            os.mkdir(data_pickle_dir)
            
        st_time = time.time()
        start_frm_index = 0        # 10 frames
        #data_pickle_dir = write_config_frm_poseEst_result(dataDir3 +  vd_dir, data_pickle_dir, start_frm_index)
        
        confg_frm_est_arr = readConfigFrmEstFile(data_pickle_dir)

        data_pose_keypoint_dir = dataDir3 +  vd_dir

        interval = 1       # 1 sec
        metric = 'oks'
        write_config_frm_acc_oks_result_interval(data_pose_keypoint_dir, confg_frm_est_arr, data_pickle_dir, interval, metric)
        
        #metric = 'acc'
        #write_config_frm_acc_oks_result_interval(data_pose_keypoint_dir, confg_frm_est_arr, data_pickle_dir, interval, metric)
        
        elapsed_time = time.time() - st_time
        print ("elapsed_time for each file: ", elapsed_time)

if __name__== "__main__":
    
    #executeWriteIntoPickleOnePeron()
    extendMoreDataSamplingWriteIntoPickle()