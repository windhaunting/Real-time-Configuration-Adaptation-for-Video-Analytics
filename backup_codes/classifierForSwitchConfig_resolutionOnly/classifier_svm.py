#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 00:02:58 2019

@author: fubao
"""

#classifier_svm


# https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/


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


from data_proc_features_01 import *


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from classifierForSwitchConfig.common_classifier import load_data_all_features
from classifierForSwitchConfig.common_classifier import read_config_name_from_file

from classifierForSwitchConfig.common_plot import plotScatterLineOneFig
from classifierForSwitchConfig.common_plot import plotOneScatterLine
from classifierForSwitchConfig.common_plot import plotOneBar





from profiling.common_prof import dataDir3




def executePlot(data_plot_dir, x_lst1, y_lst1, xlabel, ylabel, title_name):
    
    outputPlotPdf = data_plot_dir + 'classifier_result/' \
     + 'y_train_historgram.pdf'
    
    if not os.path.exists(data_plot_dir + 'classifier_result/'):
        os.mkdir(data_plot_dir + 'classifier_result/')
        
    plt = plotOneBar(list(x_lst1), list(y_lst1), xlabel, ylabel, title_name)
    
    plt.savefig(outputPlotPdf)
        

def getYoutputStaticVariable(y_train, id_config_dict, config_id_dict, data_plot_dir):
    '''
    get covariance of resolution and frame_rate
    '''
    uniques, counts = np.unique(y_train, return_counts=True)
    count_y_dict = dict(zip(uniques, counts))
    executePlot(data_plot_dir, uniques, counts, 'config_id', 'Number of samples', 'Training data classes distribution')
    
    #print ("count_y_dict: ", count_y_dict, type(unique), counts, len(unique))
    
    configs = [id_config_dict[u] for u in uniques]
    count_config_dict = dict(zip(configs, counts))
    
    print ("count_config_dict: ", len(count_config_dict), count_y_dict, count_config_dict)
    
    reso_dict = defaultdict(int)
    frmRt_dict = defaultdict(int)
    config_dict = defaultdict(int)
    for k, v in count_config_dict.items():
        reso = k.split('-')[0].split('x')[1]
        frmRt= k.split('-')[1]
        reso_dict[reso] += v
        frmRt_dict[frmRt] += v
        
        config_dict[reso +'-' + frmRt] += v
    
    
    print ("reso_dict: ", len(reso_dict), reso_dict)
    print ("frmRt_dict: ", len(frmRt_dict), frmRt_dict)
    print ("config_dict: ", len(config_dict), config_dict)
    
    #getEachProbability
    total_reso_counts = sum(reso_dict.values())
    total_frmRt_counts = sum(frmRt_dict.values())
    total_config_counts = sum(config_dict.values())
    
    prob_reso_dict = defaultdict(float)
    prob_frmRt_dict = defaultdict(float)
    prob_confg_dict = defaultdict(float)
    
    reso_lst = []
    for k, v in reso_dict.items():
        prob_reso_dict[int(k)] = v/total_reso_counts
        reso_lst.append(v)
        
    frmRt_lst = []
    for k, v in frmRt_dict.items():
        prob_frmRt_dict[int(k)] = v/total_frmRt_counts
        
    for k, v in config_dict.items():
        prob_confg_dict[k] = v/total_config_counts
        frmRt_lst.append(v)
        
    #print ("prob_reso_dict: ", len(prob_reso_dict), prob_reso_dict)
    #print ("prob_frmRt_dict: ", len(prob_frmRt_dict), prob_frmRt_dict)
    #print ("prob_confg_dict: ", len(prob_confg_dict), prob_confg_dict)
         
    EX = sum([k*prob_reso_dict[k] for k in prob_reso_dict])         # resolution
    EY = sum([k*prob_frmRt_dict[k] for k in prob_frmRt_dict])       # frame_rate
    
    VarX = sum([(k-EX)**2*prob_reso_dict[k] for k in prob_reso_dict])
    VarY = sum([(k-EY)**2*prob_frmRt_dict[k] for k in prob_frmRt_dict])
    
    EXY = 0.0
    for k, v in prob_confg_dict.items():

        reso = int(k.split('-')[0])
        frmRt = int(k.split('-')[1])
        
        EXY += reso*frmRt*prob_confg_dict[k]
        
    COVXY = EXY-EX*EY
    
    corre = COVXY/(math.sqrt(VarX)*math.sqrt(VarY))
    print ("EXEX: ", EX, EY, EXY, COVXY, VarX, VarY, corre)
    
def svmTrainTest(data_plot_dir, data_pose_keypoint_dir, X, y, kernel):
    '''
    '''
    
    print ("svmTrainTest y output datasample config: ", y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0) 
    
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)

    getYoutputStaticVariable(y_train, id_config_dict, config_id_dict, data_plot_dir)   
    
    
    #xxx
    #count_y_dict = dict(zip(unique, counts))
    #executePlot(data_plot_dir, unique, counts, 'config_id', 'Number of samples', 'Training data classes distribution')
    
    
    #print ("count_y_dict: ", count_y_dict, type(unique), counts, len(unique))
    
    #return

    scaler = StandardScaler()  # RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test =  scaler.fit_transform(X_test)
    
    print ("X_train X_test shape:",X_train.shape, X_test.shape)
    # training a linear SVM classifier 
    startTime = time.time()
     
    svm_model = SVC(kernel = kernel, C = 1000).fit(X_train, y_train) 
    #    svm_model = SVC(kernel = kernel, C = 1, gamma=1e-3).fit(X_train, y_train) 
    print ("elapsed training time: ", time.time() - startTime)
    
    startTime = time.time()
    y_pred = svm_model.predict(X_test)  # look at training 
    print ("elapsed test time: ", time.time() - startTime)
    # model accuracy for X_test   
    accuracy = svm_model.score(X_test, y_test) 
    F1_test_score = f1_score(y_pred, y_test, average='weighted') 
    train_acc_score = accuracy_score(y_train, svm_model.predict(X_train))
    # creating a confusion matrix 
    cm = confusion_matrix(y_test, y_pred) 
    print ("svmTrainTest y predicted config: ", y_pred)
    print ("svmTrainTest training acc: ", train_acc_score)
    print ("svmTrainTest testing acc cm, f1-score: ", accuracy, cm, F1_test_score)

    return svm_model

    
def svmCrossValidTrainTest(X,y, model_output_path):
        # Splitting data into train, validation and test set
    # 70% training set, 30% test set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    c = [10 ** (-3), 10 ** (-2), 10 ** (-1), 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
    g = [10 ** (-9), 10 ** (-7), 10 ** (-5), 10 ** (-3)]
    best_values = [0.0, 0.0, 0.0]  # respectively best success rate, best C and best gamma

    k_fold = KFold(n_splits=5)

    print('Performing 5-Fold validation')
    for i in c:
        plt.figure(figsize=(40, 20))
        for j in g:
            for id_train, id_test in k_fold.split(x_train):
                svc = SVC(kernel='rbf', C=i, gamma=j)
                score = svc.fit(x_train[id_train], y_train[id_train]).score(x_train[id_test], y_train[id_test])
                print('With C=' + str(i) + ' and gamma=' + str(j) + ' avg=' + str(score))
                if score > best_values[0]:
                    best_values = score, i, j

    print('Best accuracy=' + str(best_values[0]) + ' with C=' + str(best_values[1]) + ' and gamma=' + str(best_values[2]))

    # With the best C ang gamma evaluating k-fold on test set
    #print('Evaluating test set')
    svc = SVC(kernel='rbf', C=best_values[1], gamma=best_values[2])
    svc.fit(x_train, y_train)
    print('testing acc: ', svc.score(x_test, y_test))


def execute_get_feature_most_expensive_config_boundedAcc(video_dir):
    '''
    most expensive config's pose result to get feature
    '''
    data_pose_keypoint_dir =  dataDir3 + video_dir
        
    history_frame_num = 1  #1          # 
    max_frame_example_used =  8025 # 10000 #8025   # 8000
    data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
    minAccuracy = 0.85

    x_input_arr, y_out_arr = getOnePersonFeatureInputOutput01(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput02(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput03(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
    #x_input_arr, y_out_arr = getOnePersonFeatureInputOutput04(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy)
            
    x_input_arr = x_input_arr.reshape((x_input_arr.shape[0], -1))
            
    # add current config as a feature
    print ("combined before:",x_input_arr.shape, y_out_arr[history_frame_num:-1].shape)
    #current_config_arr = y_out_arr[history_frame_num:-1].reshape((y_out_arr[history_frame_num:-1].shape[0], -1))
    #x_input_arr = np.hstack((x_input_arr, current_config_arr))
            
    #y_out_arr = y_out_arr[history_frame_num+1:]
    
    print ("y_out_arr shape after:", x_input_arr.shape, y_out_arr.shape)
            
    #data_examples_arr = np.hstack((x_input_arr, y_out_arr))
            
        
    out_frm_examles_pickle_dir = data_pose_keypoint_dir + "data_examples_files_resolution_class_only/" 
    if not os.path.exists(out_frm_examles_pickle_dir):
        os.mkdir(out_frm_examles_pickle_dir)
                
    with open(out_frm_examles_pickle_dir + "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(x_input_arr, fs)
            
        
    with open(out_frm_examles_pickle_dir + "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(y_out_arr, fs)

def execute_get_feature_most_expensive_config_boundedAcc_minDelay(video_dir):
    '''
    most expensive config's pose result to get feature

    '''
    
    data_pose_keypoint_dir =  dataDir3 + video_dir

    history_frame_num = 1  #1          # 
    max_frame_example_used =  8025 # 20000 #8025   # 8000
    data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
    minAccuracy = 0.9
    minDelayTreshold = 2        # 2 sec
    
    
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
    
    #y_out_arr = y_out_arr[history_frame_num+1:]
    
    print ("y_out_arr shape after:", x_input_arr.shape, y_out_arr.shape)
    
    #data_examples_arr = np.hstack((x_input_arr, y_out_arr))
        
        
    out_frm_examles_pickle_dir = data_pose_keypoint_dir + "data_examples_files_resolution_class_only/" 
    if not os.path.exists(out_frm_examles_pickle_dir):
            os.mkdir(out_frm_examles_pickle_dir)
            
    with open(out_frm_examles_pickle_dir + "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(x_input_arr, fs)
        
    
    with open(out_frm_examles_pickle_dir + "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(y_out_arr, fs)
            
 

def execute_get_feature_selected_config_boundedAcc_minDelay(video_dir):
    '''
    use selected config's pose result to get feature
    import data_proc_feature_06_01.py

    '''
    
    data_pose_keypoint_dir =  dataDir3 + video_dir

    history_frame_num = 1  #1          # 
    max_frame_example_used =  10000 # 20000 #8025   # 8000
    data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
    minAccuracy = 0.85
    minDelayTreshold = 2        # 2 sec
    
    
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
    
    #y_out_arr = y_out_arr[history_frame_num+1:]
    
    print ("y_out_arr shape after:", x_input_arr.shape, y_out_arr.shape)
    
    #data_examples_arr = np.hstack((x_input_arr, y_out_arr))
        
        
    out_frm_examles_pickle_dir = data_pose_keypoint_dir + "data_examples_files_resolution_class_only/" 
    if not os.path.exists(out_frm_examles_pickle_dir):
            os.mkdir(out_frm_examles_pickle_dir)
            
    with open(out_frm_examles_pickle_dir + "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(x_input_arr, fs)
        
    
    with open(out_frm_examles_pickle_dir + "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl", 'wb') as fs:
        pickle.dump(y_out_arr, fs)
        


def executeTest_feature_classification():
    '''
    execute classification, where features are calculated from the pose esimation result derived from the most expensive config
    '''
    #video_dir_lst = ['output_001-dancing-10mins/', 'output_006-cardio_condition-20mins/', 'output_008-Marathon-20mins/']   
    
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                'output_003_dance/', 'output_004_dance/',  \
                'output_005_dance/', 'output_006_yoga/', \
                'output_007_yoga/', 'output_008_cardio/', \
                'output_009_cardio/', 'output_010_cardio/']
        
    
    for video_dir in video_dir_lst[1:2]:    # [2:3]:  #    # [2:3]:   #   #    # [2:3]:   #[1:2]:      #     #[0:1]:     #[ #[1:2]:  #[1:2]:         #[0:1]:
        
        #execute_get_feature_most_expensive_config_boundedAcc(video_dir)
        execute_get_feature_most_expensive_config_boundedAcc_minDelay(video_dir)
        #execute_get_feature_selected_config_boundedAcc_minDelay(video_dir)
        
        data_examples_dir =  dataDir3 + video_dir + 'data_examples_files_resolution_class_only/'
        
        xfile = 'X_data_features_config-history-frms1-sampleNum8025.pkl'  #'X_data_features_config-history-frms1-sampleNum20000.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
        yfile = 'Y_data_features_config-history-frms1-sampleNum8025.pkl' #'Y_data_features_config-history-frms1-sampleNum20000.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
        
        #xfile = 'X_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
        #yfile = 'Y_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
        X,y= load_data_all_features(data_examples_dir, xfile, yfile)
        
        
        data_pose_keypoint_dir =  dataDir3 + video_dir

        kernel =   'rbf'            #'poly'  #'sigmoid'  # 'rbf'    # linear
        svm_model = svmTrainTest(dataDir3 + video_dir, data_pose_keypoint_dir, X, y, kernel)
        
        
        model_output_path = dataDir3 + video_dir + 'classifier_result_resolution_class_only/' + 'svm_out_model'
        if not os.path.exists(dataDir3 + video_dir + 'classifier_result_resolution_class_only/'):
            os.mkdir(dataDir3 + video_dir + 'classifier_result_resolution_class_only/')
        
        #svmCrossValidTrainTest(X,y, model_output_path)
        
        save_model_flag = True
        if save_model_flag:
            pickle.dump(svm_model, open(model_output_path, 'wb'))
    



    
if __name__== "__main__": 

    
    executeTest_feature_classification()
