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
from blist import blist

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common_classifier import load_data_all_features
from sklearn.metrics import confusion_matrix


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from profiling.common_prof import dataDir3


def xgbBoostTrainTest(X, y):

    # Splitting the dataset into the Training set and Test set
    
    X_train, X_Test, y_train, Y_Test = train_test_split(X, y, test_size = 0.2, random_state = 0)
   
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print ("X_train X_test shape:", X_train.shape, X_Test.shape)
    
    # Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_Test = sc_X.fit_transform(X_Test)
    
    # Fitting the classifier into the Training set
    xgb_model = XGBClassifier()

    xgb_model.fit(X_train,y_train)
    
    # Predicting the test set results
    
    Y_Pred = xgb_model.predict(X_Test)
    
    # Making the Confusion Matrix 
    test_acc_score = xgb_model.score(X_Test, Y_Test) 
    cm = confusion_matrix(Y_Test, Y_Pred)
    
    train_acc_score = xgb_model.score(X_train, y_train) 
    print ("rftTrainTest predicted config: ", Y_Pred)
     
    print ("rftTrainTest training acc, cm: ", train_acc_score)

    print ("rftTrainTest testing acc, cm: ", test_acc_score,  cm)

    return xgb_model, train_acc_score, test_acc_score

def execute_get_feature_config_boundedAcc(history_frame_num, max_frame_example_used, video_dir, feature_calculation_flag):
    '''
    most expensive config's pose result to get feature
    '''
    data_pose_keypoint_dir =  dataDir3 + video_dir
        

    data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
    minAccuracy = 0.85

    if feature_calculation_flag == 'most_expensive_config':
        from data_proc_features_03 import getOnePersonFeatureInputOutputAll001
        
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput01(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput02(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput03(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput04(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)

    x_input_arr, y_out_arr = getOnePersonFeatureInputOutputAll001(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
 
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
            


def execute_get_feature_config_boundedAcc_minDelay(history_frame_num, max_frame_example_used, video_dir, feature_calculation_flag):
    '''
    most expensive config's pose result to get feature

    '''
    
    data_pose_keypoint_dir =  dataDir3 + video_dir

    data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
    minAccuracy = 0.9
    minDelayTreshold = 2        # 2 sec
    
    if feature_calculation_flag == 'most_expensive_config':
        from data_proc_features_03_02 import getOnePersonFeatureInputOutput01
    
    elif feature_calculation_flag == 'selected_config':
        from data_proc_features_06_01 import getOnePersonFeatureInputOutput01
        
    x_input_arr, y_out_arr = getOnePersonFeatureInputOutput01(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy, minDelayTreshold)
    
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput02(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy, minDelayTreshold)
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
    
    
def executeTest_feature_boundedAcc_minDelay():
    '''
    execute classification, where features are calculated from the pose esimation result derived from the most expensive config
    '''
    #video_dir_lst = ['output_001-dancing-10mins/', 'output_006-cardio_condition-20mins/', 'output_008-Marathon-20mins/']    
    
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                'output_003_dance/', 'output_004_dance/',  \
                'output_005_dance/', 'output_006_yoga/', \
                'output_007_yoga/', 'output_008_cardio/', \
                'output_009_cardio/', 'output_010_cardio/']
        
    for video_dir in video_dir_lst[0:1]:   # [2:3]:   # [1:2]:  # [0:1]:  #[1:2]:  #[1:2]:         #[0:1]:
        
                
        history_frame_num = 1  #1          # 
        max_frame_example_used =  8000 # 20000 #8025   # 8000
    
        execute_get_feature_config_boundedAcc_minDelay(history_frame_num, max_frame_example_used, video_dir, 'most_expensive_config')
        
        data_examples_dir =  dataDir3 + video_dir + 'data_examples_files/'
        
        xfile = "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl"
        yfile = "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl" #'Y_data_features_config-history-frms1-sampleNum20000.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
        
        #xfile = 'X_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
        #yfile = 'Y_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
        X,y= load_data_all_features(data_examples_dir, xfile, yfile)
    
        xgbBoostTrainTest(X,y)
   



def combineMultipleVideoDataTrainTest():
    '''
    combine mutlipel data example together to train and test
    '''
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/']
        

    X_lst = blist()
    y_lst = blist()
    for video_dir in video_dir_lst[0:8]:  # [2:3]:     # [2:3]:   #[1:2]:      #[0:1]:     #[ #[1:2]:  #[1:2]:         #[0:1]:
        data_examples_dir =  dataDir3 + video_dir + 'data_examples_files/'
            
        history_frame_num = 1  #1          # 
        max_frame_example_used =  10000 # 20000 #8025   # 8000
        execute_get_feature_config_boundedAcc(history_frame_num, max_frame_example_used, video_dir, 'most_expensive_config')
        #execute_get_feature_config_boundedAcc_minDelay(history_frame_num, max_frame_example_used, video_dir, 'most_expensive_config')

        xfile = "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl"
        yfile = "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl" #'Y_data_features_config-history-frms1-sampleNum20000.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
        
        #xfile = 'X_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
        #yfile = 'Y_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
        X,y= load_data_all_features(data_examples_dir, xfile, yfile)
        print("X: ", X.shape, y.shape)
        
        X_lst.append(X)
        y_lst.append(y.reshape(-1, 1))
    
    total_X = np.vstack(X_lst)
    total_y = np.vstack(y_lst)
    
    print("total_X: ", total_X.shape, total_y.shape)
    
    data_pose_keypoint_dir =  dataDir3 + video_dir
    
    data_plot_dir = dataDir3 + video_dir +'classifier_result/'
    if not os.path.exists(data_plot_dir):
        os.mkdir(data_plot_dir)

    svm_model, train_acc_score, test_acc_score = xgbBoostTrainTest(total_X, total_y)
            
    
if __name__== "__main__": 
    
    #executeTest_feature_boundedAcc_minDelay()
    combineMultipleVideoDataTrainTest()
