#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:38:49 2019

@author: fubao
"""

# https://www.kaggle.com/sanket30/cudnnlstm-lstm-99-accuracy


#  https://github.com/yjchoe/TFCudnnLSTM


# LSTM-based classifier with tensorflow



# use CPU only for LSTM
# reference: https://gist.github.com/siemanko/b18ce332bde37e156034e5d3f60f8a23

# https://github.com/aymericdamien/TensorFlow-Examples

import sys
import os
import csv
import pickle

import numpy as np
import pandas as pd

from glob import glob
from blist import blist


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir2
from profiling.common_prof import frameRates


CONFIG_NUM = 0              # need to be updated later in function "read_data_represent"
FRAME_NUM = 0               # ...
represent_dim = None        # ...

def read_data_represent(data_keypoint_represent_dir):
    '''
    # read each video's data, read the representation; 
    # one video to test now
    '''
    represent_file = data_keypoint_represent_dir + 'config_frames_representation_pkl'
    
    config_frm_represent_arr = np.load(represent_file)
    
    global CONFIG_NUM, FRAME_NUM, represent_dim
    CONFIG_NUM = config_frm_represent_arr.shape[0]
    FRAME_NUM = config_frm_represent_arr.shape[1]
    
    
    represent_dim = (config_frm_represent_arr.shape[2], config_frm_represent_arr.shape[3])
    
    return config_frm_represent_arr


def initlize_networks():
    
    # Training Parameters
    learning_rate = 0.001
    training_steps = 10000
    batch_size = 128
    display_step = 200
    
    # Network Parameters
    num_input = (CONFIG_NUM, represent_dim[0], represent_dim[1])    # (66, 5, 17) each image is a config_num

    print ("num_input: ", num_input)

    timesteps = 5       # 10, 15  #timesteps   #5 frames, 10 frames as a timestep to input
    num_hidden = 128    # hidden layer num of features
    
    
    
if __name__== "__main__": 
        
    data_keypoint_represent_dir =  dataDir2 + 'output_006-cardio_condition-20mins/' + 'representation_files/'
    read_data_represent(data_keypoint_represent_dir)
    
    initlize_networks()
    
