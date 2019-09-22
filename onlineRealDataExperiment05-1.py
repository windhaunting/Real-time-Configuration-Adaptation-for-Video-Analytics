#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:45:33 2019

@author: fubao
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:07:00 2019

@author: fubao
"""

'''
test the impact of profiling
consider the bounded accuracy, select best config to observe the lag

'''


import os
import pandas as pd
import copy
import random
import numpy as np
import copy
import operator
import math
import time

import numpy as np
import pickle

from blist import blist

from glob import glob
from collections import defaultdict
from plot import plotTwoDimensionScatter
from plot import plotTwoDimensionScatterLine
from plot import plotTwoDimensionMultiLines
from plot import plotUpsideDownTwoFigures
from plot import plotTwoSubplots
from plot import plotThreeSubplots

from common import retrieve_name
from common import cls_fifo
from common import getBufferedLag
from common import read_profile_data
from common import paddingZeroToInter

from common import dataDir2

from profiling.common_prof import PLAYOUT_RATE
from profiling.common_prof import frameRates

import matplotlib.backends.backend_pdf

from blist import blist

# actually online simulation
'''
simulate with a threshold of lag
and chek the accuracy can achieve?

different lag thresholds?

finally to get a plot with different average accuracy with different lags

Q1:  online simulation, does it keep within the lag in until the end of each segment?   
if not, we allow some fluctuation, why not set previous threshold higher? and  what's the point of threshold?

'''


    
def getNewconfig(reso, model):
    '''
    get new config from all available frames
    the config file name has only frame rate 25;  1 frame do not have frame rate 
    so we get more models from frames
    '''
    config_lst = blist()
    for frm in frameRates:
        config_lst.append(reso + '-' + str(frm) + '-' + model)
        
    return config_lst
    
 
def read_config_name_from_file(data_frame_dir):
    '''
    read config info and order based on resolution*frame rate and then order them in descending order
    '''
    
    
    config_lst = blist()
    # get config_id
    
    filePathLst = sorted(glob(data_frame_dir + "*frame_result*.tsv"))  # must read ground truth file(the most expensive config) first
    resoFrmRate = blist()
    for fileCnt, filePath in enumerate(filePathLst):
        # get the resolution, frame rate, model
        # input_output/diy_video_dataset/output_006-cardio_condition-20mins/frames_config_result/1120x832_25_cmu_frame_result.tsv
        filename = filePath.split('/')[-1]
        #print ("filename: ", filename)
        reso = filename.split('_')[0]
        res_right = reso.split('x')[1]
        frm_rate = filename.split('_')[1]
        
        model = filename.split('_')[2]
        
        
        #720 720 25 mobilenet
        #res_frame_multiply = int(res_right) * int(frm_rate)    # order of the resolution
        
        #print ("reso: ", reso, res_right, frm_rate, model)
        
        config_lst += getNewconfig(reso, model)     # add more configs
        
        #resoFrmRate.append(res_frame_multiply)  random.sort(key=lambda e: e[1])

        
    #model_resoFrm_dict = dict(zip(config_lst, resoFrmRate))
    #sort by resolution*frame_rate
    config_lst.sort(key = lambda ele: int(ele.split('-')[0].split('x')[1])* int(ele.split('-')[1]), reverse=True)
    config_id_dict = dict(zip(config_lst,range(1, len(config_lst)+1)))
        
    id_config_dict = dict(zip(range(1, len(config_lst)+1), config_lst))

    #print ("model_resoFrm_dict: ", id_config_dict, len(id_config_dict), config_id_dict)
    
    return config_id_dict, id_config_dict


def getconfigSPF(reso, model, time_spf, acc):
    '''
    get new config's tiem_spf for the current frame ,  because the input is for PLAYOUT rate 25
    
    '''
    config_spf = blist()
    config_acc = blist()
    for frm in frameRates:
        frmInter = math.ceil(PLAYOUT_RATE/frm)          # frame rate sampling frames in interval, +1 every other
        
        new_spf = time_spf/frmInter
        
        config_spf.append(new_spf)
        config_acc.append(acc)
    return config_spf, config_acc

def read_frame_result_config_numpy(data_frame_dir, out_frm_spf_pickle_file, out_frm_acc_pickle_file):
    '''
    read frame profiling result into numpy array
    
    
    config_id:  reso --frame -- model
    matrix 1: accuracy value
                image_id
    config_id     0.9
    
    matrix 2: detection  computation cost value  SPF
                imag_id
    config_id   0.05 s
    
    '''
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_frame_dir)
    
    #config_lst = blist()
    # 1120x832_25_cmu_frame_result
    filePathLst = sorted(glob(data_frame_dir + "*frame_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    config_num = len(config_id_dict)  # len(config_id_dict)
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection
    frame_num = len(df_det)+100     # because maybe some frame_id is missing
    #create a numpy array
    arr_confg_frm_spf = np.zeros((config_num, frame_num)) # array of time_spf with config vs frame_Id
    arr_confg_frm_acc = np.zeros((config_num, frame_num)) # array of acc with config vs frame_Id
    
    for fileCnt, filePath in enumerate(filePathLst):
        df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)         # det-> detection
        

        print ("numy shape: ", arr_confg_frm_spf.shape, filePath)
        
        for index, row in df_det.iterrows():  
            #print ("index, row: ", index, row)
            reso = row['Resolution']
            #frm_rate = row['Frame_rate']
            model = row['Model']
            time_spf = row['Time_SPF']
            acc = row['Acc']
            frm_id = int(row['Image_path'].split('/')[-1].split('.')[0])
            
            config_spf, config_acc = getconfigSPF(reso, model, time_spf, acc)
            
            #print ("config_lst: ", config_spf)
            #print ("id: ", config_id_dict[config_lst[1]])
            arr_confg_frm_spf[fileCnt*len(frameRates):fileCnt*len(frameRates)+len(frameRates), frm_id-1] = config_spf
           
            arr_confg_frm_acc[fileCnt*len(frameRates):fileCnt*len(frameRates)+len(frameRates), frm_id-1] = config_acc
           
            #print ("arr_confg_frm_spf: ", arr_confg_frm_spf, frm_id-1)
            
            #if index == 1:
            #    break   # debug only
                
        #if fileCnt == 1:
        #    break   # debug only

    print ("arr_confg_frm_spf: ", arr_confg_frm_spf)
    
    with open(out_frm_spf_pickle_file,'wb') as fs:
        pickle.dump(arr_confg_frm_spf, fs)
        
    with open(out_frm_acc_pickle_file,'wb') as fa:
        pickle.dump(arr_confg_frm_acc, fa)




def read_segment_result_config_numpy(data_frame_dir, out_seg_spf_pickle_file, out_seg_acc_pickle_file):
    '''
    read frame profiling result into numpy array
    
    
    config_id:  reso --frame -- model
    matrix 1: accuracy value
                image_id
    config_id     0.9
    
    matrix 2: detection  computation cost value  SPF
                imag_id
    config_id   0.05 s
    
    '''
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_frame_dir)
    
    #config_lst = blist()
    # 1120x832_25_cmu_frame_result
    filePathLst = sorted(glob(data_frame_dir + "*frame_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    config_num = len(config_id_dict)  # len(config_id_dict)
    seg_num = len(filePathLst)
    
    arr_confg_seg_spf = np.zeros((config_num, seg_num)) # array of time_spf with config vs frame_Id
    arr_confg_seg_acc = np.zeros((config_num, seg_num)) # array of acc with config vs frame_Id
    
    
    for fileCnt, filePath in enumerate(filePathLst):
        df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)         # det-> detection
        for index, row in df_det.iterrows():  
            #print ("index, row: ", index, row)
            reso = row['Resolution']
            #frm_rate = row['Frame_rate']
            model = row['Model']
            seg_spf = 1.0/row['Detection_speed_FPS']
            acc = row['Acc']        
            seg_id = int(row['Segment_no'])
            
            arr_confg_seg_spf[index, seg_id] = seg_spf
            arr_confg_seg_acc[index, seg_id] = acc

    print ("arr_confg_seg_spf: ", arr_confg_seg_spf)
    
    with open(out_seg_spf_pickle_file,'wb') as fs:
        pickle.dump(arr_confg_seg_spf, fs)
        
    with open(out_seg_acc_pickle_file,'wb') as fa:
        pickle.dump(arr_confg_seg_acc, fa)
    

def executeWriteIntoPickle():
    data_frame_dir = dataDir2 + 'output_006-cardio_condition-20mins/' + 'frames_config_result/'
    #read_frame_result_config_numpy(data_frame_dir)
    
    
    pickle_dir = dataDir2 + 'output_006-cardio_condition-20mins/' + "pickle_files/"
    if not os.path.exists(pickle_dir):
        os.mkdir(pickle_dir)
    '''
    out_frm_spf_pickle_file = pickle_dir + "spf_frame.pkl"      # spf for config vs each frame
    out_frm_acc_pickle_file = pickle_dir + "acc_frame.pkl"      # acc for config vs each frame
    read_frame_result_config_numpy(data_frame_dir, out_frm_spf_pickle_file, out_frm_acc_pickle_file)
    '''
    
    
    if not os.path.exists(pickle_dir):
        os.mkdir(pickle_dir)
    out_seg_spf_pickle_file = pickle_dir + "spf_seg.pkl"      # spf for config vs each frame
    out_seg_acc_pickle_file = pickle_dir + "acc_seg.pkl"      # acc for config vs each frame
    read_frame_result_config_numpy(data_frame_dir, out_seg_spf_pickle_file, out_seg_acc_pickle_file)
        
if __name__== "__main__": 
    executeWriteIntoPickle()
    

