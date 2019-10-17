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
from blist import blist

from collections import defaultdict
from imblearn.over_sampling import SMOTE

#from sklearn import datasets 
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support, classification_report, balanced_accuracy_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC 
from sklearn.externals import joblib
from sklearn.decomposition import PCA

from common_classifier import load_data_all_features
from common_classifier import read_config_name_from_file
from common_classifier import feature_selection

from common_plot import plotScatterLineOneFig
from common_plot import plotOneScatterLine
from common_plot import plotOneBar
from common_plot import plotTwoLinesOneFigure


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir3




def executePlot(data_plot_dir, x_lst1, y_lst1, xlabel, ylabel, title_name):
    
    outputPlotPdf = data_plot_dir   + 'y_train_historgram.pdf'
    
    #print ("58 outputPlotPdf: ", outputPlotPdf)
    
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
    
    #print ("svmTrainTest y output datasample config: ", y)
    

    video_frm_id_arr = X[:, :1]
    
    # remove the first two columns, which is the video id and frame_id
    X = X[:, 1:]
    
    #feature selection with chi2
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    X = SelectKBest(chi2, k = 20).fit_transform(X, y)

    #print ("X_train data column 0 :", X[:, 0])
    #print ("X_train data column 1 :", X[:, 1], video_frm_id_arr.shape, X.shape)    
    
    # add to video_id and frame_id back to know the instance
    X = np.hstack((video_frm_id_arr, X))
    
    #print ("X_train data column 0 bbbb :", X[:, 0])
    #print ("X_train data column 1 bbbb :", X[:, 1])   
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = None, shuffle = True) 
    
    
    test_video_frm_id_arr = X_test[:, 0]
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)

    getYoutputStaticVariable(y_train, id_config_dict, config_id_dict, data_plot_dir)   
    
    #from imblearn.over_sampling import RandomOverSampler
    #ros = RandomOverSampler(random_state=0)
    #X_train, y_train = ros.fit_resample(X_train, y_train)
    
    #xxx
    #count_y_dict = dict(zip(unique, counts))
    #executePlot(data_plot_dir, unique, counts, 'config_id', 'Number of samples', 'Training data classes distribution')
    
    
    #print ("count_y_dict: ", count_y_dict, type(unique), counts, len(unique))
    
    #pca = PCA(n_components=15, svd_solver='full')
    #X_train = pca.fit_transform(X_train)
    #X_test = pca.fit_transform(X_test)
    
    
    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]

    scaler = StandardScaler()   #  RobustScaler()   # StandardScaler() 
    X_train = scaler.fit_transform(X_train)
    X_test =  scaler.fit_transform(X_test)
    
    
    print ("X_train X_test shape:", X_train.shape, X_test.shape)

    # training a linear SVM classifier 
    startTime = time.time()
     
    #svm_model = SVC(kernel = kernel, C = 1, class_weight='balanced').fit(X_train, y_train) 
    svm_model = SVC(kernel = kernel, C = 1).fit(X_train, y_train) 
    #    svm_model = SVC(kernel = kernel, C = 1, gamma=1e-3).fit(X_train, y_train) 
    print ("elapsed training time: ", time.time() - startTime)
    
    startTime = time.time()
    y_pred = svm_model.predict(X_test)  # look at training 
    print ("elapsed test time: ", time.time() - startTime)
    # model accuracy for X_test   
    test_acc_score = round(accuracy_score(y_test, y_pred), 3)  #svm_model.score(X_test, y_test) 
    F1_test_score = round(f1_score(y_pred, y_test, average='weighted'), 3) 
    train_acc_score = round(accuracy_score(y_train, svm_model.predict(X_train)), 3)  # accuracy_score(y_train, svm_model.predict(X_train))
    
    # creating a confusion matrix 
    cm = confusion_matrix(y_test, y_pred) 
    print ("svmTrainTest y predicted config: ", y_pred)
    print ("svmTrainTest training acc: ", train_acc_score)
    print ("svmTrainTest testing acc cm, f1-score: ", test_acc_score, cm, F1_test_score)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)

    print('svmTrainTest precision: {}'.format(precision))
    print('svmTrainTest recall: {}'.format(recall))
    print('svmTrainTest f1score: {}'.format(fscore))
    print('svmTrainTest support: {}'.format(support))

    return svm_model, train_acc_score, test_acc_score, test_video_frm_id_arr, y_pred, y_test



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
    svm_model = SVC(kernel='rbf', C=best_values[1], gamma=best_values[2])
    svm_model.fit(x_train, y_train)
    train_acc_score = svm_model.score(x_train, y_train)
    test_acc_score = svm_model.score(x_test, y_test)
    
    print('training acc: ', train_acc_score)
    print('testing acc: ', test_acc_score)
    
    return svm_model, train_acc_score, test_acc_score

def execute_get_feature_config_boundedAcc(history_frame_num, max_frame_example_used, video_dir, feature_calculation_flag):
    '''
    most expensive config's pose result to get feature
    '''
    data_pose_keypoint_dir =  dataDir3 + video_dir
        

    data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
    minAccuracy = 0.9

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
    use selected config's pose result to get feature
    import data_proc_feature_06_01.py

    '''

    data_pose_keypoint_dir =  dataDir3 + video_dir


    data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
    minAccuracy = 0.85
    minDelayTreshold = 2      # 2 sec
    
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
        
    for video_dir in video_dir_lst[2:3]:  # [2:3]:     # [2:3]:   #[1:2]:      #[0:1]:     #[ #[1:2]:  #[1:2]:         #[0:1]:
        
        history_frame_num = 1  #1          # 
        max_frame_example_used =  8000 # 20000 #8025   # 8000
    
        y_training_acc_lst = blist()
        y_testing_acc_lst = blist()
        max_frame_example_used_lst = [8000]  # range(8000, 9000, 1000)   # range(3000, 10000, 500)
        for max_frame_example_used in max_frame_example_used_lst:
            
            #execute_get_feature_config_boundedAcc(history_frame_num, max_frame_example_used, video_dir, 'most_expensive_config')
            execute_get_feature_config_boundedAcc_minDelay(history_frame_num, max_frame_example_used, video_dir, 'most_expensive_config')
            #execute_get_feature_config_boundedAcc_minDelay(history_frame_num, max_frame_example_used, video_dir, 'selected_config')
            
            data_examples_dir =  dataDir3 + video_dir + 'data_examples_files/'
            
    
            xfile = "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl"
            yfile = "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl" #'Y_data_features_config-history-frms1-sampleNum20000.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
            
            #xfile = 'X_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
            #yfile = 'Y_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
            X,y= load_data_all_features(data_examples_dir, xfile, yfile)
            
            
            data_pose_keypoint_dir =  dataDir3 + video_dir
    
            kernel =  'rbf'  # 'rbf' #'poly'  #'sigmoid'  # 'rbf'    # 'linear'
            data_plot_dir = dataDir3 + video_dir +'classifier_result/'
            if not os.path.exists(data_plot_dir):
                os.mkdir(data_plot_dir)
        
            svm_model, train_acc_score, test_acc_score = svmTrainTest(data_plot_dir, data_pose_keypoint_dir, X, y, kernel)
            
            #svm_model, train_acc_score, test_acc_score = svmCrossValidTrainTest(X, y, kernel)

            
            y_training_acc_lst.append(train_acc_score)
            y_testing_acc_lst.append(test_acc_score)
            
            model_output_path = data_plot_dir + 'max_frame_example_used' + str(max_frame_example_used) +'-svm_out_model'
            #svmCrossValidTrainTest(X,y, model_output_path)
            
            save_model_flag = False
            if save_model_flag:
                pickle.dump(svm_model, open(model_output_path, 'wb'))
    
    outputPlotPdf = data_plot_dir  + 'traing_test_accuracy.pdf'
    print ("max_frame_example_used_lst len: " ,len(max_frame_example_used_lst), len(y_training_acc_lst), len(y_testing_acc_lst))
    plt = plotTwoLinesOneFigure(max_frame_example_used_lst, y_training_acc_lst, y_testing_acc_lst, "Data example number",  "accuracy", "traing_test_accuracy")

    plt.savefig(outputPlotPdf)


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
        max_frame_example_used =  12000 # 20000 #8025   # 10000
        execute_get_feature_config_boundedAcc(history_frame_num, max_frame_example_used, video_dir, 'most_expensive_config')
        #execute_get_feature_config_boundedAcc_minDelay(history_frame_num, max_frame_example_used, video_dir, 'most_expensive_config')

        xfile = "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl"
        yfile = "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl" #'Y_data_features_config-history-frms1-sampleNum20000.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
        
        #xfile = 'X_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
        #yfile = 'Y_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
        X,y= load_data_all_features(data_examples_dir, xfile, yfile)
        
        print("X y shape: ", X.shape, y.shape)
        
        
        X_lst.append(X)
        y_lst.append(y.reshape(-1, 1))
    
    total_X = np.vstack(X_lst)
    total_y = np.vstack(y_lst)
    
    print("total_X: ", total_X.shape, total_y.shape)
    
    data_pose_keypoint_dir =  dataDir3 + video_dir
    
    kernel =  'rbf'  # 'rbf' #'poly'  #'sigmoid'  # 'rbf'    # 'linear'
    data_plot_dir = dataDir3 + video_dir +'classifier_result/'
    if not os.path.exists(data_plot_dir):
        os.mkdir(data_plot_dir)

    svm_model, train_acc_score, test_acc_score, video_frm_id_arr, y_pred, y_test = svmTrainTest(data_plot_dir, data_pose_keypoint_dir, total_X, total_y, kernel)
            
    
    return video_frm_id_arr, y_pred, y_test 
    
    
if __name__== "__main__": 
    
    data_examples_dir =  dataDir3 + 'output_001_dance/' + 'data_examples_files/'

    #executeTest_feature_classification()
    combineMultipleVideoDataTrainTest()