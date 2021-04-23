#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:50:57 2020

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
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support, classification_report, balanced_accuracy_score, precision_recall_curve, roc_curve, auc
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2

from data_file_process import read_pickle_data


from common_plot import plotScatterLineOneFig

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from profiling.common_prof import dataDir3


class ModelClassify(object):
    def __init__(self):
        #self.data_classification_dir = data_classification_dir  # input file directory
        pass
    
    
    def read_whole_data_instances(self, data_file):
        # Input:  data file
        # output: X and y 
        #data_file = self.data_classification_dir + "data_instance_xy.pkl"
        
        data = read_pickle_data(data_file)
        
        X = data[:, :-1]
        y = data[:, -1]
        
        print("X: ", X.shape, y.shape)
        return X, y

    
    def get_train_test_data(self, X, y):
        #get train test
       
        print ("X, y shape: ", X.shape, y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        return X_train, X_test, y_train, y_test
       

    def train_model_logistic_cv(self, X_train, y_train):
        # cross validation, y_train including train and validation data set
        clf = LogisticRegressionCV(cv=2, random_state=0).fit(X_train, y_train)
        
        print ("clf para: ", clf.get_params())
        return clf

        
    def logistic_train_test(self, X, y):
        
        X = SelectKBest(chi2, k=20).fit_transform(X, y)
        X_train, X_test, y_train, y_test = self.get_train_test_data(X, y)
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
                

        clf = self.train_model_logistic_cv(X_train, y_train)
        
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        F1_test_score = round(f1_score(y_train_pred, y_train, average='weighted'), 3) 
        train_acc_score = round(accuracy_score(y_train, y_train_pred), 3)  # accuracy_score(y_train, svm_model.predict(X_train))
        test_acc_score = round(accuracy_score(y_test, y_test_pred), 3)  #svm_model.score(X_test, y_test) 

        print ("logistic_train_test y predicted config: ", F1_test_score, train_acc_score, test_acc_score)


        """
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
        lg_model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
        cv_results = cross_validate(lg_model, X, y, cv=5)

        print("cv result: ", cv_results)
        #lg_model.fit(X_train,y_train)
        
        # Predicting the test set results
        
        y_train_used_pred = lg_model.predict(X_train)
        y_test_used_pred = lg_model.predict(X_test)
        
        test_acc_score = round(accuracy_score(y_test, y_test_used_pred), 3)  #svm_model.score(X_test, y_test) 
        F1_test_score = round(f1_score(y_test_used_pred, y_test, average='weighted'), 3) 
        train_acc_score = round(accuracy_score(y_train, lg_model.predict(X_train)), 3)  # accuracy_score(y_train, svm_model.predict(X_train))
        
        # creating a confusion matrix 
        cm = confusion_matrix(y_test, y_test_used_pred) 
        print ("rftTrainTest y predicted config: ", y_test_used_pred)
        print ("rftTrainTest training acc: ", train_acc_score)
        print ("rftTrainTest testing acc cm, f1-score: ", test_acc_score, cm, F1_test_score)
        """
     
    def train_model_svm_cv(self, X_train, y_train):
        # cross validation, y_train including train and validation data set
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100]}
        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters)
        clf.fit(X_train, y_train)
        
        print ("clf: ", type(clf.best_params_), clf.best_params_, clf.best_score_)
        return clf.best_params_
    
    def svm_train_test(self, X, y):
           
        X_train, X_test, y_train, y_test = self.get_train_test_data(X, y)
    
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        
        best_params = self.train_model_svm_cv(X_train, y_train)
        svc = svm.SVC(**best_params)
        svc.fit(X_train, y_train)
        
        y_train_pred = svc.predict(X_train)

        y_test_pred = svc.predict(X_test)
        
        F1_test_score = round(f1_score(y_train_pred, y_train, average='weighted'), 3) 
        train_acc_score = round(accuracy_score(y_train, y_train_pred), 3)  # accuracy_score(y_train, svm_model.predict(X_train))
        test_acc_score = round(accuracy_score(y_test, y_test_pred), 3)  #svm_model.score(X_test, y_test) 

        print ("svm_train_test y predicted config: ", F1_test_score, train_acc_score, test_acc_score)


    def train_model_one_video(self):
        video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                        'output_003_dance/', 'output_004_dance/',  \
                        'output_005_dance/', 'output_006_yoga/', \
                        'output_007_yoga/', 'output_008_cardio/', \
                        'output_009_cardio/', 'output_010_cardio/']
            
        for video_dir in video_dir_lst[3:4]: # [4:5]:    # [2:3]:   #[1:2]:  # [1:2]:  #[0:1]:        #[1:2]:
            
            data_pickle_dir = dataDir3 + video_dir + "/jumping_number_result/" 
            data_file = data_pickle_dir + "data_instance_xy_01.pkl"
            X, y = self.read_whole_data_instances(data_file)
            
            self.logistic_train_test(X, y)
            #self.svm_train_test(X, y)
            
if __name__== "__main__": 
    
    model_obj = ModelClassify()
    model_obj.train_model_one_video()