#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 00:28:45 2019

@author: fubao
"""


# random_forest_tree

#https://github.com/mahesh147/Random-Forest-Classifier/blob/master/random_forest_classifier.py

import pandas as pd
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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support, classification_report, balanced_accuracy_score, precision_recall_curve, roc_curve



current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from profiling.common_prof import dataDir3


def tuneParamterMinSamplesSplits(x_train, y_train, x_test, y_test):

    min_samples_splits = np.linspace(0.001, 1.0, 20, endpoint=True)
    min_samples_splits = list(range(2, 100, 5))

    train_results = []
    test_results = []
    for min_samples_split in min_samples_splits:
       rf = RandomForestClassifier(min_samples_split=min_samples_split)
       rf.fit(x_train, y_train)

       train_pred = rf.predict(x_train)
       acc_score = round(f1_score(y_train, train_pred, average='weighted'), 3)  #svm_model.score(X_test, y_test) 
       train_results.append(acc_score)
       
       y_pred = rf.predict(x_test)
       acc_score = round(f1_score(y_test, y_pred, average='weighted'), 3)  #svm_model.score(X_test, y_test) 
       test_results.append(acc_score)
       
    print ("train_results: ", len(train_results), len(test_results))
    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train Acc")
    line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test Acc")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy')
    #plt.xlabel('Tree depth')
    plt.xlabel('Min samples split')
    plt.show()


def tuneParamterMaxFeature(x_train, y_train, x_test, y_test):
    # https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d

    max_features = list(range(1,x_train.shape[1]))
    train_results = []
    test_results = []
    for max_feature in max_features:
       rf = RandomForestClassifier(max_features=max_feature)
       rf.fit(x_train, y_train)

       train_pred = rf.predict(x_train)
       acc_score = round(f1_score(y_train, train_pred, average='macro'), 3)  #svm_model.score(X_test, y_test) 
       train_results.append(acc_score)
       
       y_pred = rf.predict(x_test)
       acc_score = round(f1_score(y_test, y_pred, average='macro'), 3)  #svm_model.score(X_test, y_test) 
       test_results.append(acc_score)
    
    
    print ("train_results: ", len(train_results), len(test_results))
    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(max_features, train_results, 'b', label="Train Acc")
    line2, = plt.plot(max_features, test_results, 'r', label="Test Acc")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy')
    #plt.xlabel('Tree depth')
    plt.xlabel('Max feature')
    plt.show()


def tuneParameter(x_train, y_train, x_test, y_test):
    #tune a parameter in the rf
    # https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d
    
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]

    max_depths = np.linspace(1, 32, 32, endpoint=True)
    
    train_results = []
    test_results = []
    
    #for max_depth in max_depths:
    for estimator in n_estimators:
       #rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
       rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)

       rf.fit(x_train, y_train)
       
       train_pred = rf.predict(x_train)
       acc_score = round(f1_score(y_train, train_pred, average='macro'), 3)  #svm_model.score(X_test, y_test) 
       train_results.append(acc_score)
       
       y_pred = rf.predict(x_test)
       acc_score = round(f1_score(y_test, y_pred, average='macro'), 3)  #svm_model.score(X_test, y_test) 
       test_results.append(acc_score)
       
    print ("train_results: ", len(train_results), len(test_results))
    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(n_estimators, train_results, 'b', label="Train Acc")
    line2, = plt.plot(n_estimators, test_results, 'r', label="Test Acc")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy')
    #plt.xlabel('Tree depth')
    plt.xlabel('Tree Number')
    plt.show()

    
def generate_y_out_put(data_classification_dir, X, y):
    #generte the y output 
      
    with open(data_classification_dir + "all_video_x_feature.pkl", 'wb') as fs:
        pickle.dump(X, fs)
    
    
    with open(data_classification_dir + "all_video_y_gt.pkl", 'wb') as fs:
        pickle.dump(y, fs)
    
    
    with open(data_classification_dir + "all_video_frm_id_arr.pkl", 'wb') as fs:
        pickle.dump(X[:, 0], fs)
        
        
def rftTrainTest(data_classification_dir, X, y, total_x_video_frm_path_lst):
       
    
    X = X.reshape((X.shape[0], -1))
    y = y.reshape((-1, 1))
    print ("X y shape:", X.shape, y.shape)
    
    
    # add to video_id and frame_id back to know the instance
    X = np.hstack((total_x_video_frm_path_lst, X))
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.2, shuffle = True)
    #print ("X_train X_test shape:", X_train.shape, X_test.shape)
   
    
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
    #max_features = 20 # 'sqrt'   # 10 # 'auto' 'sqrt' int(math.sqrt(X_train.shape[1]))
    rf_model = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', max_features='auto', max_depth=13, min_samples_split= 8, n_jobs=-1)
    rf_model.fit(X_train,y_train)
    
    # Predicting the test set results
    
    y_train_used_pred = rf_model.predict(X_train)
    y_test_used_pred = rf_model.predict(X_test)
    
    test_acc_score = round(accuracy_score(y_test, y_test_used_pred), 3)  #svm_model.score(X_test, y_test) 
    F1_test_score = round(f1_score(y_test_used_pred, y_test, average='weighted'), 3) 
    train_acc_score = round(accuracy_score(y_train, y_train_used_pred), 3)  # accuracy_score(y_train, svm_model.predict(X_train))
    
    # creating a confusion matrix 
    cm = confusion_matrix(y_test, y_test_used_pred) 
    print ("rftTrainTest y predicted config: ", y_test_used_pred)
    print ("rftTrainTest training acc: ", train_acc_score)
    print ("rftTrainTest testing acc cm, f1-score: ", test_acc_score, cm, F1_test_score)

    
    #data_classification_dir = dataDir3 + 'output_005_dance/classifier_result/'
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


def execute_all_video_train_test():
    
    minAccuracy = 0.95

    from data_proc_feature_analysize_02 import get_all_video_x_y_manual_feature
    
    total_X, total_y, total_x_video_frm_path_lst = get_all_video_x_y_manual_feature()
    
    data_classification_dir = dataDir3 + 'test_classification_result/'
    if not os.path.exists(data_classification_dir):
        os.mkdir(data_classification_dir)

    data_classification_dir = data_classification_dir + 'min_accuracy-' + str(minAccuracy)+ '/'
    if not os.path.exists(data_classification_dir):
        os.mkdir(data_classification_dir)
            
    rf_model, train_acc_score, test_acc_score, train_video_frm_id_arr, test_video_frm_id_arr, y_pred, y_test = rftTrainTest(data_classification_dir, total_X, total_y, total_x_video_frm_path_lst)
        
    #overall_diffSum, sub_diffSum = calculateDifferenceSumFrmRate(y_test, y_pred, id_config_dict)
    #print ("combineMultipleVideoDataTrainTest diffSum: ", overall_diffSum, sub_diffSum)
    
    return test_video_frm_id_arr, y_pred, y_test


if __name__== "__main__": 
    
    #data_examples_dir =  dataDir3 + 'output_001_dance/' + 'data_examples_files/'

    #executeTest_feature_classification()
    
    execute_all_video_train_test()
    
    #combineAugmentedVideoDatasetTrainTest()