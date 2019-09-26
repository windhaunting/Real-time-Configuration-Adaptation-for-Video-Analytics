#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:11:56 2019

@author: fubao
"""

# xgboost



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 00:28:45 2019

@author: fubao
"""


# random_forest_tree

#https://github.com/mahesh147/Random-Forest-Classifier/blob/master/random_forest_classifier.py


import sys
import os
import pickle

import numpy as np
import time
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common_classifier import load_data_all_features
from sklearn.metrics import confusion_matrix
from data_proc_features_03_01 import *


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from profiling.common_prof import dataDir2


def xgbBoostTrainTest(X, y):

    # Splitting the dataset into the Training set and Test set
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    print ("X_train X_test shape:", X_Train.shape, X_Test.shape)
   
    # Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    X_Test = sc_X.transform(X_Test)
    
    # Fitting the classifier into the Training set
    classifier = XGBClassifier()

    classifier.fit(X_Train,Y_Train)
    
    # Predicting the test set results
    
    Y_Pred = classifier.predict(X_Test)
    
    # Making the Confusion Matrix 
    accuracy = classifier.score(X_Test, Y_Test) 
    cm = confusion_matrix(Y_Test, Y_Pred)
    
    training_accuracy = classifier.score(X_Train, Y_Train) 
    print ("rftTrainTest predicted config: ", Y_Pred)
     
    print ("rftTrainTest training acc, cm: ", training_accuracy)

    print ("rftTrainTest testing acc, cm: ", accuracy,  cm)

def execute_get_feature_boundedAcc(video_dir):
    '''
    most expensive config's pose result to get feature
    '''
    data_pose_keypoint_dir =  dataDir2 + video_dir
        
    history_frame_num = 1  #1          # 
    max_frame_example_used =  8025 # 10000 #8025   # 8000
    data_pickle_dir = dataDir2 + video_dir + 'frames_pickle_result/'
    minAccuracy = 0.85

    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput01(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput02(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput03(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    x_input_arr, y_out_arr = getOnePersonFeatureInputOutput04(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
            
    x_input_arr = x_input_arr.reshape((x_input_arr.shape[0], -1))
            
    # add current config as a feature
    print ("combined before:",x_input_arr.shape, y_out_arr[history_frame_num:-1].shape)
    #current_config_arr = y_out_arr[history_frame_num:-1].reshape((y_out_arr[history_frame_num:-1].shape[0], -1))
    #x_input_arr = np.hstack((x_input_arr, current_config_arr))
            
    #y_out_arr = y_out_arr[history_frame_num+1:]
    
    print ("y_out_arr shape after:", x_input_arr.shape, y_out_arr.shape)
            
    #data_examples_arr = np.hstack((x_input_arr, y_out_arr))
            
        
    out_frm_examles_pickle_dir = data_pose_keypoint_dir + "data_examples_files/" 
    if not os.path.exists(out_frm_examles_pickle_dir):
        os.mkdir(out_frm_examles_pickle_dir)
                
    with open(out_frm_examles_pickle_dir + "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(x_input_arr, fs)
            
        
    with open(out_frm_examles_pickle_dir + "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(y_out_arr, fs)


def execute_get_feature_boundedAcc_minDelay(video_dir):
    '''
    most expensive config's pose result to get feature

    '''
    
    data_pose_keypoint_dir =  dataDir2 + video_dir

    history_frame_num = 1  #1          # 
    max_frame_example_used =  8025 # 20000 #8025   # 8000
    data_pickle_dir = dataDir2 + video_dir + 'frames_pickle_result/'
    minAccuracy = 0.85
    minDelayTreshold = 2        # 2 sec
    
    
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput01(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy, minDelayTreshold)
    
    x_input_arr, y_out_arr = getOnePersonFeatureInputOutput02(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy, minDelayTreshold)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput03(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy, minDelayTreshold)
    
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput04(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy, minDelayTreshold)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput05(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy, minDelayTreshold)
    
    #y_out_arr = getGroundTruthY(data_pickle_dir, max_frame_example_used, history_frame_num)
    x_input_arr = x_input_arr.reshape((x_input_arr.shape[0], -1))
    
    # add current config as a feature
    print ("combined before:",x_input_arr.shape, y_out_arr[history_frame_num:-1].shape)
    #current_config_arr = y_out_arr[history_frame_num:-1].reshape((y_out_arr[history_frame_num:-1].shape[0], -1))
    #x_input_arr = np.hstack((x_input_arr, current_config_arr))
    
    
    print ("y_out_arr shape after:", x_input_arr.shape, y_out_arr.shape)
    
    #data_examples_arr = np.hstack((x_input_arr, y_out_arr))
        
        
    out_frm_examles_pickle_dir = data_pose_keypoint_dir + "data_examples_files/" 
    if not os.path.exists(out_frm_examles_pickle_dir):
            os.mkdir(out_frm_examles_pickle_dir)
            
    with open(out_frm_examles_pickle_dir + "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(x_input_arr, fs)
        
    
    with open(out_frm_examles_pickle_dir + "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(y_out_arr, fs)
    
    
def executeTest_feature_most_expensive_config():
    '''
    execute classification, where features are calculated from the pose esimation result derived from the most expensive config
    '''
    video_dir_lst = ['output_001-dancing-10mins/', 'output_006-cardio_condition-20mins/', 'output_008-Marathon-20mins/'
                     ]    
    
    for video_dir in video_dir_lst:   # [2:3]:   # [1:2]:  # [0:1]:  #[1:2]:  #[1:2]:         #[0:1]:
        
        execute_get_feature_boundedAcc_minDelay(video_dir)
        
        data_examples_dir =  dataDir2 + video_dir + 'data_examples_files/'

        xfile = 'X_data_features_config-history-frms1-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
        yfile = 'Y_data_features_config-history-frms1-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
        
        #xfile = 'X_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
        #yfile = 'Y_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
        X,y= load_data_all_features(data_examples_dir, xfile, yfile)
    
        xgbBoostTrainTest(X,y)
   
    
def executeTest_feature_selected_config():
    '''
    '''
    data_examples_dir =  dataDir2 + 'output_006-cardio_condition-20mins/' + 'data_examples_files_feature_selected_config/'

    #xfile = 'X_data_features_config-history-frms25-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
    #yfile = 'Y_data_features_config-history-frms25-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
    
    xfile = 'X_data_features_config-history-frms1-sampleNum35765.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
    yfile = 'Y_data_features_config-history-frms1-sampleNum35765.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
    X,y= load_data_all_features(data_examples_dir, xfile, yfile)
    xgbBoostTrainTest(X,y)


if __name__== "__main__": 
    
    executeTest_feature_most_expensive_config()
    
    #executeTest_feature_selected_config()
