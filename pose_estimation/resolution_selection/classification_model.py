#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:48:35 2020

@author: fubao
"""

#train and test 




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
import joblib
import matplotlib.pyplot as plt
from blist import blist

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support, classification_report, balanced_accuracy_score, precision_recall_curve, roc_curve, auc
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from data_file_process import write_pickle_data
from data_file_process import read_pickle_data
from common_plot import plotScatterLineOneFig
from common_plot import plotOneScatter

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/../..')

from profiling.common_prof import dataDir3

dataDir3 = "../" + dataDir3

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
        
        # make it balanced
        #from imblearn.over_sampling import RandomOverSampler
        #ros = RandomOverSampler(random_state=0)
        #X_train, y_train = ros.fit_resample(X_train, y_train)
    
        #sm = SMOTE(random_state=42)
        #X_train, y_train = sm.fit_resample(X_train, y_train)
        
        clf = self.train_model_logistic_cv(X_train, y_train)
        
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        F1_test_score = round(f1_score(y_train_pred, y_train, average='micro'), 3) 
        train_acc_score = round(accuracy_score(y_train, y_train_pred), 3)  # accuracy_score(y_train, svm_model.predict(X_train))
        test_acc_score = round(accuracy_score(y_test, y_test_pred), 3)  #svm_model.score(X_test, y_test) 

        print ("logistic_train_test y predicted config: ", F1_test_score, train_acc_score, test_acc_score, y_test_pred)

        return y_test, y_test_pred


     
    def train_model_svm_cv(self, X_train, y_train):
        # cross validation, y_train including train and validation data set
        parameters = {'kernel':('sigmoid', 'rbf'), 'C':[1, 10, 100]}
        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters)
        clf.fit(X_train, y_train)
        
        print ("clf: ", type(clf.best_params_), clf.best_params_, clf.best_score_)
        return clf.best_params_
    
    def get_rf_model_train(self, X, y):
        #get the random forest model
        # X, y are the trianing and validation data
        pca = PCA(n_components=10).fit(X)
        X = pca.transform(X)
            
        #sc_X = StandardScaler()
        #X_train = sc_X.fit_transform(X_train)
        #X_test = sc_X.transform(X_test)

        rfr = RandomForestClassifier(max_depth=20, random_state=0)
        rfr.fit(X, y)   
        #print(rfr.feature_importances_)
        
        """
        best_params = self.train_rf_cross_validation(X, y)  # (X, y) # (X_train, y_train)
        rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=0)
        rfr.fit(X, y) # (X, y)  # 
        """
        
        y_train_pred = rfr.predict(X)
        #y_train_pred = list(map(lambda ele: round(ele[0]), y_train_pred))
        
        train_F1_score = round(f1_score(y_train_pred, y, average='micro'), 3) 
        train_acc_score = round(accuracy_score(y_train_pred, y), 3)  # accuracy_score(y_train, svm_model.predict(X_train))
        print ("get_rf_model_train F1_test_score: ", train_F1_score, train_acc_score)
        
        return rfr, pca
    
    def test_on_data_y_unknown(self, model, X_test, pca):
        # test on a test data with model already trained, y label unknown
        # X_test is just one instance
        #X_test = np.reshape(X_test, (1, -1))
        #print ("test_on_data_y_unknown X_test shape: ", X_test)
        #pca = PCA(n_components=10)
        
        X_test = np.reshape(X_test, (1, -1))
        X_test = pca.transform(X_test)
        #print ("test_on_data_y_unknown X_test pcaaa shape: ", X_test)
        y_test_pred = model.predict(X_test)
        
        #print ("test_on_data_y_unknown y_test_pred r2: ", X_test,  y_test_pred)
        
        return y_test_pred
       
        
        
    def test_on_data_y_known(self, model, X_test, y_test, pca):
        # test on a test data with model already trained, y label is known
        
        #pca = PCA(n_components=pca_component)
        #X_test = pca.fit_transform(X_test)
        
        X_test = pca.transform(X_test)
        
        y_test_pred = model.predict(X_test)
        y_test_pred = list(map(lambda ele: round(ele), y_test_pred))
        #print ("rf_train_test y predicted config: ", X_test, y_test, y_test_pred)
        
        F1_test_score = round(f1_score(y_test_pred, y_test, average='micro'), 3) 
        test_acc_score = round(accuracy_score(y_test_pred, y_test), 3)  # accuracy_score(y_train, svm_model.predict(X_train))
        print ("get_rf_model_train F1_test_score: ", F1_test_score, test_acc_score)
           
        
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
        
        F1_test_score = round(f1_score(y_train_pred, y_train, average='micro'), 3) 
        train_acc_score = round(accuracy_score(y_train, y_train_pred), 3)  # accuracy_score(y_train, svm_model.predict(X_train))
        test_acc_score = round(accuracy_score(y_test, y_test_pred), 3)  #svm_model.score(X_test, y_test) 

        print ("svm_train_test y predicted config: ", F1_test_score, train_acc_score, test_acc_score)

        return y_test, y_test_pred

    def plot_result(self, y_test, y_test_pred, output_dir):
        
        fig = plotOneScatter(y_test, y_test_pred, "y_true", "y_pred", "")
        out_file_path = output_dir +  "test_res.pdf"
        fig.savefig(out_file_path)
        
    def train_model_one_video(self):
        video_dir_lst =  ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/', \
                    'output_011_dance/', 'output_012_dance/', \
                    'output_013_dance/', 'output_014_dance/', \
                    'output_015_dance/', 'output_016_dance/', \
                    'output_017_dance/', 'output_018_dance/', \
                    'output_019_dance/', 'output_020_dance/', \
                    'output_021_dance/', 'output_022_dance/', \
                    'output_023_dance/', 'output_024_dance/', \
                    'output_025_dance/', 'output_026_dance/', \
                    'output_027_dance/', 'output_028_dance/', \
                    'output_029_dance/', 'output_030_dance/', \
                    'output_031_dance/', 'output_032_dance/', \
                    'output_033_dance/', 'output_034_dance/', \
                    'output_035_dance/']
        
    
        for video_dir in video_dir_lst[3:4]: # [4:5]:    # [2:3]:   #[1:2]:  # [1:2]:  #[0:1]:        #[1:2]:
            
            data_pickle_dir = dataDir3 + video_dir + "/jumping_number_result/" 
            subDir = data_pickle_dir + "resolution_prediction/intervalFrm-1_speedType-ema_minAcc-0.9/"

            data_file = subDir + "data_instance_speed_resolution_xy.pkl"
            X, y = self.read_whole_data_instances(data_file)
            
            self.logistic_train_test(X, y)
            #self.svm_train_test(X, y)

  
    def train_model_multiple_videos(self):
        # combine several videos together
        
        # select one video to train and test
        video_dir_lst =  ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/', \
                    'output_011_dance/', 'output_012_dance/', \
                    'output_013_dance/', 'output_014_dance/', \
                    'output_015_dance/', 'output_016_dance/', \
                    'output_017_dance/', 'output_018_dance/', \
                    'output_019_dance/', 'output_020_dance/', \
                    'output_021_dance/', 'output_022_dance/', \
                    'output_023_dance/', 'output_024_dance/', \
                    'output_025_dance/', 'output_026_dance/', \
                    'output_027_dance/', 'output_028_dance/', \
                    'output_029_dance/', 'output_030_dance/', \
                    'output_031_dance/', 'output_032_dance/', \
                    'output_033_dance/', 'output_034_dance/', \
                    'output_035_dance/']
            
        for i, video_dir in enumerate(video_dir_lst):    # combine all to generate more data
            
            data_pickle_dir = dataDir3 + video_dir + "jumping_number_result/" 
            subDir = data_pickle_dir + "resolution_prediction/intervalFrm-1_speedType-ema_minAcc-1.0/"
            data_file = subDir + "data_instance_speed_resolution_xy.pkl"
            X, y = self.read_whole_data_instances(data_file)
            
            print("X, y: ", X.shape, y.shape)
            
            if i == 0:
                x_input_arr = X
                y_arr = y
            
            else:
                x_input_arr = np.vstack((x_input_arr, X))
                y_arr = np.hstack((y_arr, y))

            
        print("x_input_arr , y: ", x_input_arr.shape, y_arr.shape)

        y_test, y_test_pred = self.rf_regressor_train_test(x_input_arr, y_arr)
            
        #y_test, y_test_pred = self.svr_train_test (x_input_arr, y_arr)
        
        output_dir = data_pickle_dir
        self.plot_result(y_test, y_test_pred, output_dir)
        
    
    def train_rest_test_one_video(self, predicted_video_dir):
        # pick one video ast test and the rest as the training
        # select one video to train and test
        video_dir_lst =  ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/', \
                    'output_011_dance/', 'output_012_dance/', \
                    'output_013_dance/', 'output_014_dance/', \
                    'output_015_dance/', 'output_016_dance/', \
                    'output_017_dance/', 'output_018_dance/', \
                    'output_019_dance/', 'output_020_dance/', \
                    'output_021_dance/', 'output_022_dance/', \
                    'output_023_dance/', 'output_024_dance/', \
                    'output_025_dance/', 'output_026_dance/', \
                    'output_027_dance/', 'output_028_dance/', \
                    'output_029_dance/', 'output_030_dance/', \
                    'output_031_dance/', 'output_032_dance/', \
                    'output_033_dance/', 'output_034_dance/', \
                    'output_035_dance/']
            
        X_test = None
        y_test = None
        first_flag = True
        for i, video_dir in enumerate(video_dir_lst):    # combine all to generate more data
            
            
            data_pickle_dir = dataDir3 + video_dir + "jumping_number_result/" 
            subDir = data_pickle_dir + "resolution_prediction/intervalFrm-1_speedType-ema_minAcc-0.92/"
            data_file = subDir + "data_instance_speed_resolution_xy.pkl"
            X, y = self.read_whole_data_instances(data_file)
            
            print("X, y: ", X.shape, y.shape)
            
            if predicted_video_dir == video_dir:        # get this video for test
                X_test = X
                y_test = y
            
            else:
                if first_flag:
                    x_input_arr = X
                    y_arr = y
                    first_flag = False
                
                else:
                    x_input_arr = np.vstack((x_input_arr, X))
                    y_arr = np.hstack((y_arr, y))
                    
       
        rfr, pca = self.get_rf_model_train(x_input_arr, y_arr)   
        print("rfr:" , )
        
        write_subDir = dataDir3 + predicted_video_dir + "jumping_number_result/"  + "resolution_prediction/intervalFrm-1_speedType-ema_minAcc-0.92/"
        model_file = write_subDir + "model_classifi_reso.joblib.pkl"
        
        x_input_arr_file = write_subDir + "trained_x_instances.pkl"
        write_pickle_data(x_input_arr, x_input_arr_file)
        
        _ = joblib.dump(rfr, model_file, compress=4)
        
        # get testing result
        self.test_on_data_y_known(rfr, X_test, y_test, pca)
        

if __name__== "__main__": 

    model_obj = ModelClassify()
    # model_obj.train_model_one_video() 

    #model_obj.train_model_multiple_videos()
    
    video_dir_lst =  ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/']
    
    for predicted_video_dir in video_dir_lst:
                    
        #predicted_video_dir = 'output_021_dance/'     # select different video id for testing
        model_obj.train_rest_test_one_video(predicted_video_dir)

