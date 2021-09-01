#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:28:11 2019

@author: fubao
"""


import sys
import math
import os
import pickle
import numpy as np
import time
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

from collections import defaultdict
#from sklearn import datasets 
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC 
from sklearn.externals import joblib


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir2
from classifierForSwitchConfig.common_classifier import read_config_name_from_file
from classifierForSwitchConfig.common_classifier import readProfilingResultNumpy
from classifierForSwitchConfig.common_classifier import read_poseEst_conf_frm

# test classification result
# e.g. use the trained model to do switching on a video and test the accuracy and the delay.

def load_model_classification(data_pickle_dir, model_path):
    '''
    use a video test on the classification model
    '''

    
    # load the model from disk
    loaded_model = pickle.load(open(model_path, 'rb'))
    
    #result = loaded_model.score(X_test, Y_test)
    print("loaded_model: ", loaded_model)
    
    confg_est_frm_arr = read_poseEst_conf_frm(data_pickle_dir)
    
    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)

    frms_num = confg_est_frm_arr.shape[1]
    
    #get the feature first
    
    for ind in range(0, frms_num):
        '''
        xue
        '''
        
        

if __name__== "__main__": 
    
    video_dir = 'output_001-dancing-10mins/'
    model_path = dataDir2 + video_dir + 'classifier_result/' + 'svm_out_model'

    data_pickle_dir = dataDir2 + video_dir + 'frames_pickle_result/'


    load_model_classification(data_pickle_dir, model_path)









    