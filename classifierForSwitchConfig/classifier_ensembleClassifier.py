#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:03:15 2019

@author: fubao
"""



# overcome the imbalanced dataset 
# not use oversampling or undersampling

# use bagging classifier 
# it's said that the skleran bagging classifier can not solve the imbalanced problem

# https://towardsdatascience.com/having-an-imbalanced-dataset-here-is-how-you-can-solve-it-1640568947eb



import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedBaggingClassifier

from common_classifier import load_data_all_features
from sklearn.metrics import confusion_matrix


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from profiling.common_prof import dataDir2






def emsembleClassifierTrainTest(X, y):

    # Splitting the dataset into the Training set and Test set
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    print ("X_train X_test shape:", X_Train.shape, X_Test.shape)
   
    # Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    X_Test = sc_X.transform(X_Test)
    
    # Fitting the classifier into the Training set
    classifier = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators = 200,
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)
    
    classifier.fit(X_Train,Y_Train)
    
    # Predicting the test set results
    
    Y_Pred = classifier.predict(X_Test)
    
    # Making the Confusion Matrix 
    accuracy = classifier.score(X_Test, Y_Test) 
    cm = confusion_matrix(Y_Test, Y_Pred)
    
    training_accuracy = classifier.score(X_Train, Y_Train) 

    print ("rftTrainTest testing acc, cm: ", accuracy,  cm)
     
    print ("rftTrainTest training acc, cm: ", training_accuracy)

    
def executeTest_feature_most_expensive_config():
    '''
    execute classification, where features are calculated from the pose esimation result derived from the most expensive config
    '''
    video_dir_lst = ['output_006-cardio_condition-20mins/', 'output_008-Marathon-20mins/'
                     ]   
    
    for video_dir in video_dir_lst[1:2]:  #[1:2]:         #[0:1]:
        
        data_examples_dir =  dataDir2 + video_dir + 'data_examples_files/'

        xfile = 'X_data_features_config-history-frms1-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
        yfile = 'Y_data_features_config-history-frms1-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
        
        #xfile = 'X_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
        #yfile = 'Y_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
        X,y= load_data_all_features(data_examples_dir, xfile, yfile)
    
        emsembleClassifierTrainTest(X,y)
   
    
def executeTest_feature_selected_config():
    '''
    '''
    data_examples_dir =  dataDir2 + 'output_006-cardio_condition-20mins/' + 'data_examples_files_feature_selected_config/'

    #xfile = 'X_data_features_config-history-frms25-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
    #yfile = 'Y_data_features_config-history-frms25-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
    
    xfile = 'X_data_features_config-history-frms1-sampleNum35765.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
    yfile = 'Y_data_features_config-history-frms1-sampleNum35765.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
    X,y= load_data_all_features(data_examples_dir, xfile, yfile)
    emsembleClassifierTrainTest(X,y)


if __name__== "__main__": 
    
    executeTest_feature_most_expensive_config()
    
    #executeTest_feature_selected_config()

