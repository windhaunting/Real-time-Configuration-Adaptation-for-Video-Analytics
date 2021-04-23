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
import joblib

import matplotlib.pyplot as plt
from collections import defaultdict
from blist import blist

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

from data_file_process import write_pickle_data
from data_file_process import read_pickle_data
from common_plot import plotOneScatterLine
from common_plot import plotOneScatter

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from profiling.common_prof import dataDir3


class ModelRegression(object):
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
        
        #print("X ,y: ", X.shape, y.shape)
        return X, y

    
    def get_train_test_data(self, X, y):
        #get train test
       
        print ("X, y shape: ", X.shape, y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        return X_train, X_test, y_train, y_test
       

        
    def train_rf_cross_validation(self, X_train, y_train):
        # Perform Grid-Search cross validation
        
        gsc = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid={
                'max_depth': range(5, 30, 5),
                'n_estimators': (50, 100, 200),
            },
            cv=5, scoring='neg_mean_absolute_error', verbose=0,  n_jobs=-1)
        
        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_
        
        
        rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=0)
                
        # Perform K-Fold CV
        scores = cross_val_score(rfr, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')  # scoring='neg_mean_absolute_error')
        print ("train_rf_cross_validation training scores r2: ", best_params, scores)
        
        return best_params


    def rf_regressor_train_test(self, X, y):
        # train and test with random forest regression
        pca = PCA(n_components=20)
        X = pca.fit_transform(X)
        
        X_train, X_test, y_train, y_test = self.get_train_test_data(X, y)
    
        #sc_X = StandardScaler()
        #X_train = sc_X.fit_transform(X_train)
        #X_test = sc_X.transform(X_test)
        
        
        rfr = RandomForestRegressor(max_depth=20, random_state=0)
        rfr.fit(X, y)   # (X_train, y_train) # (X, y)  # 
        print(rfr.feature_importances_)
        
        """
        best_params = self.train_rf_cross_validation(X_train, y_train)  # (X, y) # (X_train, y_train)
        rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=0)
        rfr.fit(X_train, y_train) # (X, y)  # 
        """
        
        y_train_pred = rfr.predict(X_train)
        y_train_pred = list(map(lambda ele: round(ele), y_train_pred))

        r2 = r2_score(y_train, y_train_pred, multioutput='variance_weighted')
        mse = mean_squared_error(y_train, y_train_pred) #  squared=False)
        mae = mean_absolute_error(y_train, y_train_pred)
        
        print ("rf_train_test training result r2: ", r2, mse, mae)
        
        y_test_pred = rfr.predict(X_test)
        y_test_pred = list(map(lambda ele: round(ele), y_test_pred))
        #print ("rf_train_test y predicted config: ", y_test_pred)
        
        r2 = r2_score(y_test, y_test_pred, multioutput='variance_weighted')
        mse = mean_squared_error(y_test, y_test_pred) #  squared=False)
        mae = mean_absolute_error(y_test, y_test_pred)
        
        print ("rf_train_test testing r2: ", r2, mse, mae)
        
        return y_test, y_test_pred
    
    
    def train_SVR_cross_validation(self, X_train, y_train):
        # Perform Grid-Search cross validation
        
        gsc = GridSearchCV(
            estimator=SVR(),
            param_grid={
                'C': range(1, 8, 2),
                'kernel': ("linear", "rbf", 'poly', 'sigmoid'),
            },
            cv=5, scoring='neg_mean_absolute_error', verbose=0,  n_jobs=-1)
        
        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_
        
        
        svr = SVR(C=best_params["C"], kernel=best_params['kernel'])
        # Perform K-Fold CV
        scores = cross_val_score(svr, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')  # scoring='neg_mean_absolute_error')
        print ("train_SVR_cross_validation training scores r2: ", best_params, scores)
        
        return best_params

    def svr_train_test(self, X, y):
        # support vector regressor
        pca = PCA(n_components=20)
        X = pca.fit_transform(X)
        
        X_train, X_test, y_train, y_test = self.get_train_test_data(X, y)
    
        """
        svr = make_pipeline(StandardScaler(), SVR(C=5.0, kernel = 'rbf', epsilon=0.2))
        svr.fit(X_train, y_train)
        
        """
        
        best_params = self.train_SVR_cross_validation(X_train, y_train)  # (X, y) # (X_train, y_train)
        svr = SVR(C=best_params["C"], kernel=best_params['kernel'])
        svr.fit(X_train, y_train) # (X, y)  # 
        
        
        y_train_pred = svr.predict(X_train)
        y_train_pred = list(map(lambda ele: round(ele), y_train_pred))

        r2 = r2_score(y_train, y_train_pred, multioutput='variance_weighted')
        mse = mean_squared_error(y_train, y_train_pred) #  squared=False)
        mae = mean_absolute_error(y_train, y_train_pred)
        
        print ("svr train_test training result r2: ", r2, mse, mae)
        
        y_test_pred = svr.predict(X_test)
        y_test_pred = list(map(lambda ele: round(ele), y_test_pred))
        print ("svr_train_test y predicted config: ", y_test_pred)
        
        r2 = r2_score(y_test, y_test_pred, multioutput='variance_weighted')
        mse = mean_squared_error(y_test, y_test_pred) #  squared=False)
        mae = mean_absolute_error(y_test, y_test_pred)
        
        print ("svr train_test testing r2: ", r2, mse, mae)
        return y_test, y_test_pred


    def plot_result(self, y_test, y_test_pred, output_dir):
        
        fig = plotOneScatter(y_test, y_test_pred, "y_true", "y_pred", "")
        out_file_path = output_dir +  "test_res.pdf"
        fig.savefig(out_file_path)
        
        

    def train_model_one_video(self):
        
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
            
        for video_dir in video_dir_lst[3:4]:    # [4:5]:   #[1:2]:  # [1:2]:  #[0:1]:        #[1:2]:
            
            data_pickle_dir = dataDir3 + video_dir + "jumping_number_result/" 
            subDir = data_pickle_dir + "intervalFrm-1_speedType-ema_minAcc-0.95/"
            data_file = subDir + "data_instance_xy.pkl"
            X, y = self.read_whole_data_instances(data_file)
            
            y_test, y_test_pred = self.rf_regressor_train_test(X, y)
            
            output_dir = data_pickle_dir
            self.plot_result(y_test, y_test_pred, output_dir)


        
    def get_rf_model_train(self, X, y):
        #get the random forest model
        # X, y are the trianing and validation data
        pca = PCA(n_components=10).fit(X)
        X = pca.transform(X)
            
        #sc_X = StandardScaler()
        #X_train = sc_X.fit_transform(X_train)
        #X_test = sc_X.transform(X_test)

        rfr = RandomForestRegressor(max_depth=20, random_state=0)
        rfr.fit(X, y)   
        #print(rfr.feature_importances_)
        
        """
        best_params = self.train_rf_cross_validation(X, y)  # (X, y) # (X_train, y_train)
        rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=0)
        rfr.fit(X, y) # (X, y)  # 
        """
        
        y_train_pred = rfr.predict(X)
        #y_train_pred = list(map(lambda ele: round(ele[0]), y_train_pred))

        r2_train = r2_score(y, y_train_pred, multioutput='uniform_average')
        rmse_train = mean_squared_error(y, y_train_pred, squared=False)
        mae_train = mean_absolute_error(y, y_train_pred)
        print ("get_rf_model_train training result r2: ", r2_train, rmse_train, mae_train)   
        
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
        
        r2 = r2_score(y_test, y_test_pred, multioutput='uniform_average')
        rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        mae = mean_absolute_error(y_test, y_test_pred)
        
        print ("test_on_data_y_known testing r2: ", r2, rmse, mae, y_test_pred[0])
    
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
            subDir = data_pickle_dir + "intervalFrm-1_speedType-ema_minAcc-1.0/"
            data_file = subDir + "data_instance_xy.pkl"
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
            subDir = data_pickle_dir + "jumping_number_prediction/intervalFrm-1_speedType-ema_minAcc-0.92/"
            data_file = subDir + "data_instance_xy.pkl"
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
        
        write_subDir = dataDir3 + predicted_video_dir + "jumping_number_result/"  + "jumping_number_prediction/intervalFrm-1_speedType-ema_minAcc-0.92/"
        model_file = write_subDir + "model_regression_jumping_frm.joblib.pkl"
        
        x_input_arr_file = write_subDir + "trained_x_instances.pkl"
        write_pickle_data(x_input_arr, x_input_arr_file)
        
        _ = joblib.dump(rfr, model_file, compress=4)
        
        # get testing result
        self.test_on_data_y_known(rfr, X_test, y_test, pca)
        
        

if __name__== "__main__": 
    
    model_obj = ModelRegression()
    # model_obj.train_model_one_video()
    
    # get more video data
    #model_obj.train_model_multiple_videos()
    
    
    
    video_dir_lst =  ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/']
    
    for predicted_video_dir in video_dir_lst:
                    
        #predicted_video_dir = 'output_021_dance/'     # select different video id for testing
        model_obj.train_rest_test_one_video(predicted_video_dir)
    