#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 09:29:36 2019

@author: fubao
"""

import sys
import os
import csv
import pickle
import math

import numpy as np
import pandas as pd

from glob import glob
from blist import blist
from operator import itemgetter
from collections import defaultdict
from common_classifier import read_config_name_from_file
from common_classifier import read_poseEst_conf_frm
from common_classifier import readProfilingResultNumpy

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


#from profiling.common_prof import dataDir2
from profiling.common_prof import dataDir3

from profiling.common_prof import frameRates
from profiling.common_prof import PLAYOUT_RATE


# get global optimal solution
#f(a, b, d) = max_{j}(f(a, a+1, d_{i-1}) + Acc(j-1, j))


def getOptimalSolution_config(data_pickle_dir):
    
    confg_est_frm_arr = read_poseEst_conf_frm(data_pickle_dir)
    
    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)

    # select one person, i.e. no 0
    
    #get first 10 sec
    #confg_est_frm_arr = confg_est_frm_arr[:, 0:10]
    
    acc_frame_arr = acc_frame_arr[:, 0:20]
    spf_frame_arr = spf_frame_arr[:, 0:20]
    
    print ("acc_frame_arr.shape: ", acc_frame_arr.shape)
    personNo = 0
    
    MAX_DELAY = 1
    
    dict_acc = defaultdict(list)         # the key is interval (start, end, maximum available delay)m, value is the acc and config selected
    
    for col1 in range(acc_frame_arr.shape[1]):
        for col2 in range(col1, acc_frame_arr.shape[1]):
            maxAcc = -1
            for row in range(acc_frame_arr.shape[0]):
                procTime = float(spf_frame_arr[row, col1:col1+1])
                #print ("procTime: ", procTime)
                intervalTime = 1         # 1 sec fixed
                if (MAX_DELAY - (procTime - intervalTime)) < 0:
                    continue
                new_acc = acc_frame_arr[row, col1:col1+1] + dict_acc[(col1, col2, MAX_DELAY - (procTime - intervalTime))][0]
                     
                print ("new_acc: ", new_acc)
                if maxAcc <= new_acc:
                    maxAcc = new_acc
                    dict_acc[(col1, col2, MAX_DELAY-procTime)] = [maxAcc, row]
                
            
            print ("maxAcc: ", maxAcc, row)
    #dict_acc = sorted(dict_acc.items(), key=itemgetter(1))
    
    for k, v in dict_acc.items():
        if k[0] == 0 and k[1] == acc_frame_arr.shape[1]-1:
            if k[2] <= MAX_DELAY:
                print ("vvvv: ", k, v)
    #print ("dict_acc: ", dict_acc)
    
def execute_optimization():
    
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                'output_003_dance/', 'output_004_dance/',  \
                'output_005_dance/', 'output_006_yoga/', \
                'output_007_yoga/', 'output_008_cardio/', \
                'output_009_cardio/', 'output_010_cardio/']
    

    for video_dir in video_dir_lst[0:1]:  # [2:3]:     # [2:3]:   #[1:2]:      #[0:1]:     #[ #[1:2]:  #[1:2]:         #[0:1]:
        #data_examples_dir =  dataDir3 + video_dir + 'data_examples_files/'
        data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
        
        getOptimalSolution_config(data_pickle_dir)
        
        
        
if __name__== "__main__": 
    execute_optimization()
    