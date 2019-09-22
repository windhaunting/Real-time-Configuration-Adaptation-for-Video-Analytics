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

import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from common_classifier import load_data_all_features
from sklearn.metrics import confusion_matrix


current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from profiling.common_prof import dataDir2


def logisticTrainTest(X, y):

    # Splitting the dataset into the Training set and Test set
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size = 0.1, random_state = 0)
    print ("X_train X_test shape:", X_Train.shape, X_Test.shape)
   
    # Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    X_Test = sc_X.transform(X_Test)
    
    # Fitting the classifier into the Training set
    
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_Train, Y_Train)
    
    # Predicting the test set results
    Y_Pred = clf.predict(X_Test)
    
    # Making the Confusion Matrix 
    accuracy = clf.score(X_Test, Y_Test) 
    cm = confusion_matrix(Y_Test, Y_Pred)
    
    print ("logisticTrainTest acc, cm: ", accuracy,  cm)
     
    
    
def executeTest_feature_most_expensive_config():
    '''
    '''
    data_examples_dir =  dataDir2 + 'output_006-cardio_condition-20mins/' + 'data_examples_files/'

    #xfile = 'X_data_features_config-history-frms25-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
    #yfile = 'Y_data_features_config-history-frms25-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
    
    xfile = 'X_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
    yfile = 'Y_data_features_config-weighted_interval-history-frms1-5-10-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
    X,y= load_data_all_features(data_examples_dir, xfile, yfile)

    logisticTrainTest(X,y)
    

def executeTest_feature_selected_config():
    '''
    '''
    data_examples_dir =  dataDir2 + 'output_006-cardio_condition-20mins/' + 'data_examples_files_feature_selected_config/'

    #xfile = 'X_data_features_config-history-frms25-sampleNum8025.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
    #yfile = 'Y_data_features_config-history-frms25-sampleNum8025.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
    
    xfile = 'X_data_features_config-history-frms1-sampleNum35765.pkl'    # 'X_data_features_config-history-frms1-sampleNum8025.pkl'
    yfile = 'Y_data_features_config-history-frms1-sampleNum35765.pkl'    #'Y_data_features_config-history-frms1-sampleNum8025.pkl'
    X,y= load_data_all_features(data_examples_dir, xfile, yfile)
    logisticTrainTest(X,y)
    


if __name__== "__main__": 
    
    #executeTest_feature_most_expensive_config()
    executeTest_feature_selected_config()
