#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 00:02:58 2019

@author: fubao
"""

#classifier_svm


# https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/


import sys
import os
import pickle
import numpy as np
import time
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

#from sklearn import datasets 
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC 
from sklearn.externals import joblib

from common_classifier import load_data_all_features

from common_plot import plotScatterLineOneFig
from common_plot import plotOneScatterLine
from common_plot import plotOneBar

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir2




def executePlot(data_plot_dir, x_lst1, y_lst1, xlabel, ylabel, title_name):
    
    outputPlotPdf = data_plot_dir + 'classifier_result/' \
     + 'y_train_historgram.pdf'
    
    if not os.path.exists(data_plot_dir + 'classifier_result/'):
        os.mkdir(data_plot_dir + 'classifier_result/')
        
    plt = plotOneBar(list(x_lst1), list(y_lst1), xlabel, ylabel, title_name)
    
    plt.savefig(outputPlotPdf)
        
def svmTrainTest(data_plot_dir, X, y, kernel):
    '''
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0) 
    
    
    unique, counts = np.unique(y_train, return_counts=True)
    count_y_dict = dict(zip(unique, counts))
    executePlot(data_plot_dir, unique, counts, 'config_id', 'Number of samples', 'Training data classes distribution')
    print ("count_y_dict: ", count_y_dict, type(unique), counts)
    
    #return

    scaler = StandardScaler()  # RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test =  scaler.fit_transform(X_test)
    
    print ("X_train X_test shape:",X_train.shape, X_test.shape)
    # training a linear SVM classifier 
    startTime = time.time()
     
    svm_model = SVC(kernel = kernel, C = 1).fit(X_train, y_train) 
    #    svm_model = SVC(kernel = kernel, C = 1, gamma=1e-3).fit(X_train, y_train) 
    print ("elapsed training time: ", time.time() - startTime)
    
    startTime = time.time()
    y_pred = svm_model.predict(X_test)  # look at training 
    print ("elapsed test time: ", time.time() - startTime, y_pred)
    # model accuracy for X_test   
    accuracy = svm_model.score(X_test, y_test) 
    F1_test_score = f1_score(y_pred, y_test, average='weighted') 
    train_acc_score = accuracy_score(y_train, svm_model.predict(X_train))
    # creating a confusion matrix 
    cm = confusion_matrix(y_test, y_pred) 
        
    print ("svmTrainTest testing acc cm, f1-score: ", accuracy, cm, F1_test_score)

    print ("svmTrainTest training acc: ", train_acc_score)
    
    
def svmCrossValidTrainTest(X,y, model_output_path):
        # Splitting data into train, validation and test set
    # 70% training set, 30% test set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

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
    
    
def executeTest_feature_most_expensive_config():
    '''
    execute classification, where features are calculated from the pose esimation result derived from the most expensive config
    '''
    video_dir_lst = ['output_001-dancing-10mins/', 'output_006-cardio_condition-20mins/', 'output_008-Marathon-20mins/'
                     ]   
    
    for video_dir in video_dir_lst[1:2]: #[1:2]:  #[1:2]:         #[0:1]:
        
        data_examples_dir =  dataDir2 + video_dir + 'data_examples_files/'
    
        xfile = 'X_data_features_config-history-frms1-sampleNum8025.pkl'  #'X_data_features_config-history-frms1-sampleNum20000.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
        yfile = 'Y_data_features_config-history-frms1-sampleNum8025.pkl' #'Y_data_features_config-history-frms1-sampleNum20000.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
        
        #xfile = 'X_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
        #yfile = 'Y_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
        X,y= load_data_all_features(data_examples_dir, xfile, yfile)
        
        
        kernel =   'rbf' #'poly'  #'sigmoid'  # 'rbf'    # linear
        svmTrainTest(dataDir2 + video_dir, X, y, kernel)
        
        
        model_output_path = dataDir2 + video_dir + 'classifier_result/' + 'svm_out_model'
        #svmCrossValidTrainTest(X,y, model_output_path)
        
        
    
def executeTest_feature_selected_config():
    '''
    execute classification, where features are calculated from the pose esimation result derived from the current selected config
    '''
    data_examples_dir =  dataDir2 + 'output_006-cardio_condition-20mins/' + 'data_examples_files_feature_selected_config/'

    #xfile = 'X_data_features_config-history-frms25-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
    #yfile = 'Y_data_features_config-history-frms25-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
    
    xfile = 'X_data_features_config-history-frms1-sampleNum35765.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
    yfile = 'Y_data_features_config-history-frms1-sampleNum35765.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
    X,y= load_data_all_features(data_examples_dir, xfile, yfile)
    kernel = 'rbf'    # linear
    svmTrainTest(X, y, kernel)
    
if __name__== "__main__": 
    
    
    
    executeTest_feature_most_expensive_config()
    #executeTest_feature_selected_config()