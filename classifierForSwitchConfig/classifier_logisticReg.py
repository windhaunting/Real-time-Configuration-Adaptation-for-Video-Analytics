#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:06:37 2019

@author: fubao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 00:28:45 2019

@author: fubao
"""


# logistic regression
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

import pickle
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from blist import blist

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support, classification_report, balanced_accuracy_score, precision_recall_curve, roc_curve, auc

from common_classifier import load_data_all_features
from common_classifier import calculateDifferenceSumFrmRate
from common_classifier import extract_specific_config_name_from_file

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from profiling.common_prof import dataDir3


    
def logisticTrainTest(data_classification_dir, X, y):
       
    
    X = X.reshape((X.shape[0], -1))
    y = y.reshape((-1, 1))
    print ("X y shape:", X.shape, y.shape)
    
    #video_frm_id_arr = X[:, :1]
    
    # remove the first two columns, which is the video id and frame_id
    #X = X[:, 1:]
    
    # add to video_id and frame_id back to know the instance
    #X = np.hstack((video_frm_id_arr, X))
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)
    print ("X_train X_test shape:", X_train.shape, X_test.shape)
   
    
    train_video_frm_id_arr = X_train[:, 0]
    test_video_frm_id_arr = X_test[:, 0]
    
    X_train = X_train[:, 1:]        # remove video_frm_id_arr to train
    X_test = X_test[:, 1:]
    # Feature Scaling
    
    #from imblearn.over_sampling import RandomOverSampler
    #ros = RandomOverSampler(random_state=0)
    #X_train, y_train = ros.fit_resample(X_train, y_train)
    #print ("X_train X_test shape:", X_train.shape, X_Test.shape)
    
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
        
    print ("xxx: ", X_train)
    
    #tuneParameter(X_train, y_train, X_test, y_test)
    #tuneParamterMaxFeature(X_train, y_train, X_test, y_test)
    #tuneParamterMinSamplesSplits(X_train, y_train, X_test, y_test)

    # Fitting the classifier into the Training set
    rf_model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    rf_model.fit(X_train,y_train)
    
    # Predicting the test set results
    
    y_train_used_pred = rf_model.predict(X_train)
    y_test_used_pred = rf_model.predict(X_test)
    
    test_acc_score = round(accuracy_score(y_test, y_test_used_pred), 3)  #svm_model.score(X_test, y_test) 
    F1_test_score = round(f1_score(y_test_used_pred, y_test, average='weighted'), 3) 
    train_acc_score = round(accuracy_score(y_train, rf_model.predict(X_train)), 3)  # accuracy_score(y_train, svm_model.predict(X_train))
    
    # creating a confusion matrix 
    cm = confusion_matrix(y_test, y_test_used_pred) 
    print ("rftTrainTest y predicted config: ", y_test_used_pred)
    print ("rftTrainTest training acc: ", train_acc_score)
    print ("rftTrainTest testing acc cm, f1-score: ", test_acc_score, cm, F1_test_score)

    
    data_classification_dir = dataDir3 + 'output_005_dance/classifier_result/'
    with open(data_classification_dir + "x_train.pkl", 'wb') as fs:
        pickle.dump(X_train, fs)
    
    with open(data_classification_dir + "y_train.pkl", 'wb') as fs:
        pickle.dump(y_train, fs)
   
    with open(data_classification_dir + "y_train_used_pred.pkl", 'wb') as fs:
        pickle.dump(y_train_used_pred, fs)
        
    
    with open(data_classification_dir + "x_test.pkl", 'wb') as fs:
        pickle.dump(X_test, fs)
    
    with open(data_classification_dir + "y_test.pkl", 'wb') as fs:
        pickle.dump(y_test, fs)
   
    with open(data_classification_dir + "y_test_used_pred.pkl", 'wb') as fs:
        pickle.dump(y_test_used_pred, fs)
                
    with open(data_classification_dir + "train_video_frm_id_arr.pkl", 'wb') as fs:
        pickle.dump(train_video_frm_id_arr, fs)
   
    with open(data_classification_dir + "test_video_frm_id_arr.pkl", 'wb') as fs:
        pickle.dump(test_video_frm_id_arr, fs)
        
        
    return rf_model, train_acc_score, test_acc_score, train_video_frm_id_arr, test_video_frm_id_arr, y_test_used_pred, y_test


def execute_get_feature_config_boundedAcc(data_pose_keypoint_dir, data_pickle_dir, data_frame_path_dir, history_frame_num, max_frame_example_used, feature_calculation_flag):
    '''
    most expensive config's pose result to get feature
    '''
        
    minAccuracy = 0.95

    if feature_calculation_flag == 'most_expensive_config':
        #from data_proc_features_03 import getOnePersonFeatureInputOutputAll001
        from data_proc_feature_analysize_01 import getOnePersonFeatureInputOutputAll001

    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput01(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput02(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput03(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput04(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr, _ = getOnePersonFeatureInputOutputAll001(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    x_input_arr, y_out_arr, id_config_dict, acc_frame_arr, spf_frame_arr, confg_est_frm_arr = getOnePersonFeatureInputOutputAll001(data_pose_keypoint_dir, data_pickle_dir, data_frame_path_dir, history_frame_num, max_frame_example_used, minAccuracy)
 
    x_input_arr = x_input_arr.reshape((x_input_arr.shape[0], -1))
            
    # add current config as a feature
    #print ("combined before:",x_input_arr.shape, y_out_arr.shape)
    #current_config_arr = y_out_arr[history_frame_num:-1].reshape((y_out_arr[history_frame_num:-1].shape[0], -1))
    #x_input_arr = np.hstack((x_input_arr, current_config_arr))
            
    #y_out_arr = y_out_arr[history_frame_num+1:]
    
    print ("y_out_arr shape after:", x_input_arr.shape, y_out_arr.shape)
            
    #data_examples_arr = np.hstack((x_input_arr, y_out_arr))
            
 
    return x_input_arr, y_out_arr, id_config_dict


def execute_get_feature_config_boundedDelay(data_pose_keypoint_dir, data_pickle_dir, data_frame_path_dir, history_frame_num, max_frame_example_used, feature_calculation_flag):
    '''
    most expensive config's pose result to get feature
    '''
        
    minDelayTreshold = 0
    
    if feature_calculation_flag == 'most_expensive_config':
        #from data_proc_features_03 import getOnePersonFeatureInputOutputAll001
        from data_proc_feature_analysize_01 import getOnePersonFeatureInputOutputAll001

    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput01(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput02(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput03(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput04(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr, _ = getOnePersonFeatureInputOutputAll001(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    x_input_arr, y_out_arr, id_config_dict, acc_frame_arr, spf_frame_arr, confg_est_frm_arr = getOnePersonFeatureInputOutputAll001(data_pose_keypoint_dir, data_pickle_dir, data_frame_path_dir, history_frame_num, max_frame_example_used, "", minDelayTreshold)
 
    x_input_arr = x_input_arr.reshape((x_input_arr.shape[0], -1))
            
    # add current config as a feature
    #print ("combined before:",x_input_arr.shape, y_out_arr.shape)
    #current_config_arr = y_out_arr[history_frame_num:-1].reshape((y_out_arr[history_frame_num:-1].shape[0], -1))
    #x_input_arr = np.hstack((x_input_arr, current_config_arr))
            
    #y_out_arr = y_out_arr[history_frame_num+1:]
    
    print ("y_out_arr shape after:", x_input_arr.shape, y_out_arr.shape)
            
    #data_examples_arr = np.hstack((x_input_arr, y_out_arr))
            
 
    return x_input_arr, y_out_arr, id_config_dict

def combineMultipleVideoDataTrainTest():
    '''
    combine mutlipel data example together to train and test
    '''
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/']
        

    input_video_frms_dir = ['001_dance_frames/', '002_dance_frames/', \
                        '003_dance_frames/', '004_dance_frames/',  \
                        '005_dance_frames/', '006_yoga_frames/', \
                        '007_yoga_frames/', '008_cardio_frames/',
                        '009_cardio_frames/', '010_cardio_frames/',
                        '011_dance_frames/', '012_dance_frames/',
                        '013_dance_frames/', '014_dance_frames/',
                        '015_dance_frames/', '016_dance_frames/',
                        '017_dance_frames/', '018_dance_frames/',
                        '019_dance_frames/', '020_dance_frames/',
                        '021_dance_frames/']
   
    # judge file exist or not
    data_classification_dir = dataDir3  +'test_classification_result/'

    history_frame_num = 1       #1          # 
    max_frame_example_used =  18000 # 20000 #8025   # 10000
        
    xfile = "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl"
    yfile = "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl"
    
    if os.path.exists(data_classification_dir+xfile) and os.path.exists(data_classification_dir+yfile):
        
        total_X,total_y= load_data_all_features(data_classification_dir, xfile, yfile)
        
        resolution_set = ["1120x832", "960x720", "640x480", "480x352", "320x240"]   # ["640x480"]  #["1120x832", "960x720", "640x480",  "480x352", "320x240"]   # for openPose models [720, 600, 480, 360, 240]   # [240] #     # [240]       # [720, 600, 480, 360, 240]    #   [720]     # [720, 600, 480, 360, 240]  #  [720]    # [720, 600, 480, 360, 240]            #  16: 9
        frame_set = [25, 15, 10, 5, 2, 1]     #  [25, 10, 5, 2, 1]    # [30],  [30, 10, 5, 2, 1] 
        model_set = ['cmu']   #, 'mobilenet_v2_small']
        data_pose_keypoint_dir = dataDir3 + video_dir_lst[4]
        lst_id_subconfig, id_config_dict = extract_specific_config_name_from_file(data_pose_keypoint_dir, resolution_set, frame_set, model_set)

    else:
        X_lst = blist()
        y_lst = blist()
        
        for i, video_dir in enumerate(video_dir_lst):  # [2:3]:     # [2:3]:   #[1:2]:      #[0:1]:     #[ #[1:2]:  #[1:2]:         #[0:1]:
            
            #if i != 4:                    # check the 005_video only
            #    continue
    
            data_pose_keypoint_dir =  dataDir3 + video_dir
            data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
            data_frame_path_dir = dataDir3 + input_video_frms_dir[i]
            
            out_frm_examles_pickle_dir =  dataDir3 + video_dir + 'data_examples_files/'
            
            history_frame_num = 1           #1          # 
            max_frame_example_used =  18000 # 20000 #8025   # 10000
            #x_input_arr, y_out_arr, id_config_dict = execute_get_feature_config_boundedAcc(data_pose_keypoint_dir, data_pickle_dir, data_frame_path_dir, history_frame_num, max_frame_example_used, 'most_expensive_config')
            
            x_input_arr, y_out_arr, id_config_dict = execute_get_feature_config_boundedDelay(data_pose_keypoint_dir, data_pickle_dir, data_frame_path_dir, history_frame_num, max_frame_example_used, 'most_expensive_config')
    
            if not os.path.exists(out_frm_examles_pickle_dir):
                os.mkdir(out_frm_examles_pickle_dir)
                        
            with open(out_frm_examles_pickle_dir + "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
                pickle.dump(x_input_arr, fs)
                    
            with open(out_frm_examles_pickle_dir + "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
                pickle.dump(y_out_arr, fs)
            
            #xfile = "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl"
            #yfile = "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl" #'Y_data_features_config-history-frms1-sampleNum20000.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
            
            #xfile = 'X_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
            #yfile = 'Y_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
            #x_input_arr,y_out_arr= load_data_all_features(data_examples_dir, xfile, yfile)
            
            #print("X y shape: ", y_out_arr.shape, y_out_arr.shape)
            
            
            X_lst.append(x_input_arr)
            y_lst.append(y_out_arr.reshape(-1, 1))
        
        
        total_X = np.vstack(X_lst)
        total_y = np.vstack(y_lst)
        
        print("total_X: ", total_X.shape, total_y.shape)
        
        data_pose_keypoint_dir =  dataDir3 + video_dir
        
        data_classification_dir = dataDir3 + 'test_classification_result/'
        if not os.path.exists(data_classification_dir):
            os.mkdir(data_classification_dir)

        with open(data_classification_dir + "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
            pickle.dump(total_X, fs)
                
        with open(data_classification_dir + "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
            pickle.dump(total_y, fs)    
    
    rf_model, train_acc_score, test_acc_score, train_video_frm_id_arr, test_video_frm_id_arr, y_pred, y_test = logisticTrainTest(data_classification_dir, total_X, total_y)
        
    overall_diffSum, sub_diffSum = calculateDifferenceSumFrmRate(y_test, y_pred, id_config_dict)
    print ("combineMultipleVideoDataTrainTest diffSum: ", overall_diffSum, sub_diffSum)
    
    return test_video_frm_id_arr, y_pred, y_test
    
    
if __name__== "__main__": 
    
    data_examples_dir =  dataDir3 + 'output_001_dance/' + 'data_examples_files/'

    #executeTest_feature_classification()
    combineMultipleVideoDataTrainTest()
    

