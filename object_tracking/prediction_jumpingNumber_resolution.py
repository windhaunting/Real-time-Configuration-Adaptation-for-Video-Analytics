#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 00:00:50 2021

@author: fubao
"""


import pickle
import sys
import os
import numpy as np
import time
import math
import joblib
import imblearn
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from blist import blist

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from skopt import BayesSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, f1_score, precision_recall_fscore_support
from sklearn.decomposition import PCA

from sklearn.svm import SVR
from sklearn.multioutput import  MultiOutputClassifier

# training and test the adaptive configuration
from tracking_get_training_data import min_acc_threshold
from tracking_get_training_data import max_jump_number
from tracking_get_training_data import data_dir
from tracking_get_training_data import video_dir_lst
from tracking_get_training_data import interval_frm
from tracking_get_training_data import DataGenerate

from tracking_data_preprocess import read_pickle_data
from tracking_data_preprocess import write_pickle_data




    
class ModelClassifier(object):
    def __init__(self):
        #self.data_classification_dir = data_classification_dir  # input file directory
        pass
    
    
    def read_whole_data_instances(self, data_file):
        # Input:  data file
        # output: X and y 
        #data_file = self.data_classification_dir + "data_instance_xy.pkl"
        
        data = read_pickle_data(data_file)
        #print("read_whole_data_instances X ,y: ", data.shape)
        X = data[:, :-2]
        y = data[:, -2:]
        
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
            estimator=RandomForestClassifier(),
            param_grid={
                'max_depth': range(5, 30, 5),
                'n_estimators': (50, 100, 200),
            },
            cv=10, scoring='neg_mean_absolute_error', verbose=0,  n_jobs=-1)
        
        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_
        
                
        rfr = RandomForestClassifier(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=0)
                
        # Perform K-Fold CV
        
        scores = cross_val_score(rfr, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')  # scoring='neg_mean_absolute_error')
        print ("train_rf_cross_validation training scores r2: ", best_params, scores)
        
        return best_params


    def train_xgboost_cross_validation(self, X_train, y_train):
        # Perform Grid-Search cross validation
        
        gsc = GridSearchCV(
            estimator=GradientBoostingClassifier(),
            param_grid={
                'max_depth': range(5, 30, 5),
                'n_estimators': (50, 100, 200),
                'learning_rate': (0.1, 2, 0.2)
            },
            cv=10, scoring='neg_mean_absolute_error', verbose=0,  n_jobs=-1)
        
        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_
        
                
        #rfr = RandomForestClassifier(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=0)
        
        rfr = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=20, random_state=0)
        
        # Perform K-Fold CV
        
        scores = cross_val_score(rfr, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')  # scoring='neg_mean_absolute_error')
        print ("train_rf_cross_validation training scores r2: ", best_params, scores)
        
        return best_params
    
    
    def search_best_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
        
        opt = BayesSearchCV(
        RandomForestClassifier(random_state=42),
        {
            'n_estimators': (5,300),
            'max_features': ['auto','sqrt'],
            'max_depth': (2,50),
            'min_samples_split': (2,10),
            'min_samples_leaf': (1,7),
            'bootstrap': ["True","False"]
        },
        n_iter=32,
        cv=3,
        scoring='roc_auc'
    )
        opt.fit(X_train, y_train)
        
        print("val. score: %s" % opt.best_score_)
        print("test score: %s" % opt.score(X_test, y_test))

        return opt
    
    
    def get_rf_model_train(self, X, y):
        #get the random forest model
        # X, y are the trianing and validation data

        pca = PCA(n_components=5).fit(X)
        X = pca.transform(X)
            
        #sc_X = StandardScaler()
        #X_train = sc_X.fit_transform(X_train)
        #X_test = sc_X.transform(X_test)

        #rfr = RandomForestRegressor(max_depth=20, random_state=0)
        
        mor = RandomForestClassifier(max_depth=20)
        
        
        """
        
        print ("get_rf_model_train  enter here: ", X.shape, y.shape)
        svmc = svm.SVC(kernel='rbf', gamma=0.2857, C=100, decision_function_shape='ovr')

        mor = MultiOutputClassifier(svmc)
        print("get_rf_model_train: ", type(mor))
        """
        
        
        """
        best_params = self.train_rf_cross_validation(X, y)  # (X, y) # (X_train, y_train)
        rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=0)
        rfr.fit(X, y) # (X, y)  # 
        """
        
        mor.fit(X, y)   
        y_train_pred = mor.predict(X)
        #y_train_pred = list(map(lambda ele: round(ele[0]), y_train_pred))
        """
        r2_train = r2_score(y, y_train_pred, multioutput='uniform_average')
        rmse_train = mean_squared_error(y, y_train_pred)
        mae_train = mean_absolute_error(y, y_train_pred)
        print ("get_rf_model_train training result r2: ", r2_train, rmse_train, mae_train)   
        """
        mean_macro_f1 = 0.0
        mean_micro_f1 = 0.0
        
        for i in range(y.shape[1]):
            macro_f1 = precision_recall_fscore_support(y_train_pred[:, i], y[:, i], average='macro')
            micro_f1 = precision_recall_fscore_support(y_train_pred[:, i], y[:, i], average='micro')
            print ("get_rf_model_train training macro micro f1 score  ", i, macro_f1, micro_f1)
            mean_macro_f1 += macro_f1[2]
            mean_micro_f1 += micro_f1[2]


        mean_macro_f1 /= y.shape[1]
        mean_micro_f1 /= y.shape[1]
        print ("rf_classification_train_test training mean_macro_f1 mean_micro_f1:", mean_macro_f1, mean_micro_f1)
        
        
        return mor, pca

    def test_on_data_y_unknown(self, model, X_test, pca):
        # test on a test data with model already trained, y label unknown
        # X_test is just one instance
        #X_test = np.reshape(X_test, (1, -1))
        #print ("test_on_data_y_unknown X_test shape: ", X_test)
        #pca = PCA(n_components=10)
        
        X_test = np.reshape(X_test, (1, -1))
        X_test = pca.transform(X_test)
        #print ("test_on_data_y_unknown X_test pcaaa shape: ", X_test.shape)
        y_test_pred = model.predict(X_test)
        
        #print ("test_on_data_y_unknown y_test_pred r2: ", X_test,  y_test_pred)
        
        return y_test_pred
    
    def test_on_data_y_known(self, model, X_test, y_test, pca):
        # test on a test data with model already trained, y label is known
        
        X_test = pca.transform(X_test)
        
        y_test_pred = model.predict(X_test)
        #y_test_pred = list(map(lambda ele: round(ele), y_test_pred))
        #print ("rf_train_test y predicted config: ", X_test, y_test, y_test_pred)
        
        """
        r2 = r2_score(y_test, y_test_pred, multioutput='uniform_average')
        rmse = mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        
        print ("test_on_data_y_known testing r2: ", r2, rmse, mae, y_test_pred[0])
        """
        mean_macro_f1 = 0.0
        mean_micro_f1 = 0.0
        for i in range(y_test.shape[1]):
            macro_f1 = precision_recall_fscore_support(y_test[:, i], y_test_pred[:, i], average='macro')
            micro_f1 = precision_recall_fscore_support(y_test[:, i], y_test_pred[:, i], average='micro')
            #print ("test_on_data_y_known testing macro micro f1 score  ", i, macro_f1, micro_f1)
            
            mean_macro_f1 += macro_f1[2]
            mean_micro_f1 += micro_f1[2]
            
        mean_macro_f1 /= y_test.shape[1]
        mean_micro_f1 /= y_test.shape[1]
        print ("test_on_data_y_known mean_macro_f1 mean_micro_f1:", mean_macro_f1, mean_micro_f1)
        
        return mean_macro_f1, mean_micro_f1


    def rf_classification_train_test(self, X, y):
        # train and test with random forest regression
        pca = PCA(n_components=5)
        X = pca.fit_transform(X)
        
        X_train, X_test, y_train, y_test = self.get_train_test_data(X, y)
    
        #sc_X = StandardScaler()
        #X_train = sc_X.fit_transform(X_train)
        #X_test = sc_X.transform(X_test)
        
        rfr = RandomForestClassifier(max_depth=20)
        #rfr = RandomForestRegressor(max_depth=20)
        rfr.fit(X_train, y_train)  # (X, y)   # (X_train, y_train) # (X, y)  # 
        #print(rfr.feature_importances_)
        
        """
        best_params = self.train_rf_cross_validation(X_train, y_train)  # (X, y) # (X_train, y_train)
        rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=0)
        rfr.fit(X_train, y_train) # (X, y)  # 
        """
        
        y_train_pred = rfr.predict(X_train)
        #y_train_pred = list(map(lambda ele: round(ele[0]), y_train_pred))
        
        """
        r2_train = r2_score(y_train, y_train_pred, multioutput='uniform_average')
        rmse_train = mean_squared_error(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        print ("rf_train_test training result r2: ", r2_train, rmse_train, mae_train)
        """
        mean_macro_f1 = 0.0
        mean_micro_f1 = 0.0
        for i in range(y_train.shape[1]):
            macro_f1 = precision_recall_fscore_support(y_train[:, i], y_train_pred[:, i], average='macro')
            micro_f1 = precision_recall_fscore_support(y_train[:, i], y_train_pred[:, i], average='micro')
            #print ("rf_classification_train_test training macro micro f1 score  ", i, macro_f1, micro_f1)
            
            mean_macro_f1 += macro_f1[2]
            mean_micro_f1 += micro_f1[2]
            
        mean_macro_f1 /= y_train.shape[1]
        mean_micro_f1 /= y_train.shape[1]
        print ("rf_classification_train_test training mean_macro_f1 mean_micro_f1:", mean_macro_f1, mean_micro_f1)
        
        y_test_pred = rfr.predict(X_test)
        #y_test_pred = list(map(lambda ele: round(ele[0]), y_test_pred))
        #print ("rf_train_test y predicted config: ", y_test_pred)
        
        """
        r2_test = r2_score(y_test, y_test_pred, multioutput='uniform_average')
        mse_test = mean_squared_error(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        print ("rf_train_test testing r2: ", r2_test, mse_test, mae_test)
        """
    
        mean_macro_f1 = 0.0
        mean_micro_f1 = 0.0
        for i in range(y_test.shape[1]):
            macro_f1 = precision_recall_fscore_support(y_test[:, i], y_test_pred[:, i], average='macro')
            micro_f1 = precision_recall_fscore_support(y_test[:, i], y_test_pred[:, i], average='micro')
            #print ("rf_classification_train_test training macro micro f1 score  ", i, macro_f1, micro_f1)
            
            mean_macro_f1 += macro_f1[2]
            mean_micro_f1 += micro_f1[2]
            
        mean_macro_f1 /= y_test.shape[1]
        mean_micro_f1 /= y_test.shape[1]
        print ("rf_classification_train_test testing mean_macro_f1 mean_micro_f1:", mean_macro_f1, mean_micro_f1)
        
        
        return rfr, y_test, y_test_pred
    
    
    def test_on_multiple_mixed_video(self, min_acc):
        # mixed all videos together, then split into train, test
        
       #predicted_video_dir = 'output_021_dance/'     # select different video id for testing
        first_flag = True  # first video
        for i, video_dir in enumerate(video_dir_lst[0:23]):    # combine all to generate more data
            
            
            data_pickle_dir = data_dir + video_dir + "jumping_number_result/" 
            subDir = data_pickle_dir + "jumpingNumber_resolution_selection/intervalFrm-5_speedType-ema_minAcc-" + str(min_acc) + "/"
            data_file = subDir + "data_instance_speed_jumpingNumber_resolution_objectSizeRatio_xy.pkl"
            X, y = self.read_whole_data_instances(data_file)
            
            #print("X, y: ", X.shape, y.shape)
            
            #else:
            if first_flag:
                x_input_arr = X
                y_arr = y
                first_flag = False
                
            else:
                x_input_arr = np.vstack((x_input_arr, X))
                y_arr = np.vstack((y_arr, y))
        
        print("trained finished test shape:" , x_input_arr.shape, y_arr.shape)
        rfr, y_test, y_test_pred = self.rf_classification_train_test(x_input_arr, y_arr)
    
        output_dir = data_dir + "dynamic_jumpingNumber_resolution_selection_output/"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            
        write_subDir = output_dir + "intervalFrm-5_speedType-ema_minAcc-" + str(min_acc) +"/"
        if not os.path.exists(write_subDir):
            os.mkdir(write_subDir)
        
        model_file = write_subDir + "model_regression.joblib" + "_all_videos"  + ".pkl"
        
        x_input_arr_file = write_subDir + "all_other_trained_x_instances.pkl"
        write_pickle_data(x_input_arr, x_input_arr_file)
        
        _ = joblib.dump(rfr, model_file, compress=4)
        
        
    
    def visualization_stat(self, list_arr):
        
        unique, counts = np.unique(list_arr, return_counts=True)  
        print(list(zip(unique, counts)))
        
        #print (np.asarray((unique, counts)).T)
        
    
    def train_rest_test_one_video_one_model(self, predicted_video_dir, min_acc_threshold, single_featue):
        # pick one video ast test and the rest as the training
        # select one video to train and test
 
        #predicted_video_dir = 'output_021_dance/'     # select different video id for testing
        X_test = None
        y_test = None
        first_flag = True  # first video
        for i, video_dir in enumerate(video_dir_lst[0:38]):             # combine all to generate more data
            
            
            if single_featue == 'objectSizeChange':
                data_sub_dir = data_dir + video_dir + "jumping_number_result_" + single_featue + "/" + "jumpingNumber_resolution_selection_" + single_featue + "/"
            else:
                data_sub_dir = data_dir + video_dir + "jumping_number_result/jumpingNumber_resolution_selection/"

            #data_sub_dir = data_dir + video_dir + "jumping_number_result_each_frm/jumpingNumber_resolution_selection/" 

            data_pickle_dir = data_sub_dir + "intervalFrm-" + str(interval_frm) + "_speedType-ema_minAcc-" + str(min_acc_threshold) + "/"   # "minAcc_" + str(min_acc) + "/"
            data_path_file = data_pickle_dir + "data_instance_speed_jumpingNumber_resolution_objectSizeRatio_xy.pkl"
            
            X, y = self.read_whole_data_instances(data_path_file)
            #print("X, y: ", X.shape, y.shape)
            
            if predicted_video_dir == video_dir:        # get this video for test
                X_test = X
                y_test = y
            
            #else:
            if first_flag:
                x_input_arr = X
                y_arr = y
                first_flag = False
                
            else:
                x_input_arr = np.vstack((x_input_arr, X))
                y_arr = np.vstack((y_arr, y))
        
        #self.visualization_stat(y_arr)
        rfr, pca = self.get_rf_model_train(x_input_arr, y_arr)   
        print("trained finished test shape:" , x_input_arr.shape, y_arr.shape, X_test.shape, y_test.shape)
        
        #rfr = self.search_best_model(x_input_arr, y_arr)
        
        if single_featue == 'objectSizeChange':
            write_subDir = data_dir + predicted_video_dir +  "data_instance_xy_" + single_featue + "/minAcc_" + str(min_acc_threshold) + "/"
            if not os.path.exists(data_dir + predicted_video_dir +  "data_instance_xy_" + single_featue):
                os.mkdir(data_dir + predicted_video_dir +  "data_instance_xy_" + single_featue)
            
        else:
            write_subDir = data_dir + predicted_video_dir +  "data_instance_xy/"  + "minAcc_" + str(min_acc_threshold) + "/"
            if not os.path.exists(data_dir + predicted_video_dir +  "data_instance_xy/"):
                os.mkdir(data_dir + predicted_video_dir +  "data_instance_xy/")
            
        
        if not os.path.exists(write_subDir):
            os.mkdir(write_subDir)

        model_file = write_subDir + "model_classifier.joblib" + "_exclusive_" + str(predicted_video_dir[:-1])  + ".pkl"
        
        x_input_arr_file = write_subDir + "all_other_trained_x_instances.pkl"
        write_pickle_data(x_input_arr, x_input_arr_file)
        
        _ = joblib.dump(rfr, model_file, compress=4)
        
        #pca_component = 3
        mean_macro_f1, mean_micro_f1 = self.test_on_data_y_known(rfr, X_test, y_test, pca)
        
        return mean_macro_f1, mean_micro_f1
    

    def unbalanced_processing(self, X, y):
        print("unbalanced_processing: ", X.shape, y.shape)
        oversample = SMOTE()   # k_neighbors=6
        X1 = np.copy(X)
        X1, y1 = oversample.fit_resample(X1, y[:, 0])
        
        print("X, y1 shape: ", X1.shape, y1.shape)
        X2, y2 = oversample.fit_resample(X, y[:, 1])
        print("X, y1 shape: ", X2.shape, y2.shape)
        
        return X1, y1, X2, y2    
    
    def get_rf_model_train_two_models(self, X1, y1, X2, y2):
        #get the random forest model
        # X, y are the trianing and validation data
        # X, y are the trianing and validation data

        pca1 = PCA(n_components=5).fit(X1)
        X1 = pca1.transform(X1)
        mor1 = RandomForestClassifier(max_depth=20)
        
        mor1.fit(X1, y1)   
        y_train_pred_1 = mor1.predict(X1)
        #y_train_pred = list(map(lambda ele: round(ele[0]), y_train_pred))
        """
        r2_train = r2_score(y, y_train_pred, multioutput='uniform_average')
        rmse_train = mean_squared_error(y, y_train_pred)
        mae_train = mean_absolute_error(y, y_train_pred)
        print ("get_rf_model_train training result r2: ", r2_train, rmse_train, mae_train)   
        """
        
        pca2 = PCA(n_components=5).fit(X2)
        X2 = pca2.transform(X2)        
        mor2 = RandomForestClassifier(max_depth=20)
        
        mor2.fit(X2, y2)   
        y_train_pred_2 = mor2.predict(X2)
        
        mean_macro_f1 = 0.0
        mean_micro_f1 = 0.0

        frmRate_macro_f1 = precision_recall_fscore_support(y_train_pred_1, y1, average='macro')[2]
        frmRate_micro_f1 = precision_recall_fscore_support(y_train_pred_1, y1, average='micro')[2]
            
        reso_macro_f1 = precision_recall_fscore_support(y_train_pred_2, y2, average='macro')[2]
        reso_micro_f1 = precision_recall_fscore_support(y_train_pred_2, y2, average='micro')[2]

        mean_macro_f1 = frmRate_macro_f1 + reso_macro_f1
        mean_micro_f1 = frmRate_micro_f1 + reso_micro_f1

        mean_macro_f1 /= 2
        mean_micro_f1 /= 2
        
        print ("get_rf_model_train_two_models training mean_macro_f1 mean_micro_f1:", mean_macro_f1, mean_micro_f1)
        
        
        return mor1, pca1, mor2, pca2
    
    def test_on_data_y_known_two_models(self, model1, X_test1, y_test1, pca1, model2, X_test2, y_test2, pca2):
        # test on a test data with model already trained, y label is known
        
        X_test1 = pca1.transform(X_test1)
        y_test_pred_1 = model1.predict(X_test1)
        #y_test_pred = list(map(lambda ele: round(ele), y_test_pred))
        #print ("rf_train_test y predicted config: ", X_test, y_test, y_test_pred)
        
        X_test2 = pca2.transform(X_test2)
        y_test_pred_2 = model2.predict(X_test2)
        
        """
        r2 = r2_score(y_test, y_test_pred, multioutput='uniform_average')
        rmse = mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        
        print ("test_on_data_y_known testing r2: ", r2, rmse, mae, y_test_pred[0])
        """
        mean_macro_f1 = 0.0
        mean_micro_f1 = 0.0

        frmRate_macro_f1 = precision_recall_fscore_support(y_test_pred_1, y_test1, average='macro')[2]
        frmRate_micro_f1 = precision_recall_fscore_support(y_test_pred_1, y_test1, average='micro')[2]
            
        reso_macro_f1 = precision_recall_fscore_support(y_test_pred_2, y_test2, average='macro')[2]
        reso_micro_f1 = precision_recall_fscore_support(y_test_pred_2, y_test2, average='micro')[2]

        mean_macro_f1 = frmRate_macro_f1 + reso_macro_f1
        mean_micro_f1 = frmRate_micro_f1 + reso_micro_f1
        mean_macro_f1 /= 2
        mean_micro_f1 /= 2
        
        print ("test_on_data_y_known_two_models mean_macro_f1 mean_micro_f1:", mean_macro_f1, mean_micro_f1)
        
        return mean_macro_f1, mean_micro_f1
    
    
    def train_rest_test_one_video_two_models(self, predicted_video_dir, min_acc_threshold, single_featue):
        # pick one video ast test and the rest as the training
        # select one video to train and test
 
        #predicted_video_dir = 'output_021_dance/'     # select different video id for testing
        X_test = None
        y_test = None
        first_flag = True  # first video
        for i, video_dir in enumerate(video_dir_lst[0:38]):             # combine all to generate more data
            
            
            if single_featue == 'objectSizeChange':
                data_sub_dir = data_dir + video_dir + "jumping_number_result_" + single_featue + "/" + "jumpingNumber_resolution_selection_" + single_featue + "/"
            else:
                data_sub_dir = data_dir + video_dir + "jumping_number_result/jumpingNumber_resolution_selection/"

            #data_sub_dir = data_dir + video_dir + "jumping_number_result_each_frm/jumpingNumber_resolution_selection/" 

            data_pickle_dir = data_sub_dir + "intervalFrm-" + str(interval_frm) + "_speedType-ema_minAcc-" + str(min_acc_threshold) + "/"   # "minAcc_" + str(min_acc) + "/"
            data_path_file = data_pickle_dir + "data_instance_speed_jumpingNumber_resolution_objectSizeRatio_xy.pkl"
            
            X, y = self.read_whole_data_instances(data_path_file)
            #print("X, y: ", X.shape, y.shape)
            
            if predicted_video_dir == video_dir:        # get this video for test
                X_test = X
                y_test = y
            
            #else:
            if first_flag:
                x_input_arr = X
                y_arr = y
                first_flag = False
                
            else:
                x_input_arr = np.vstack((x_input_arr, X))
                y_arr = np.vstack((y_arr, y))
        
        #self.visualization_stat(y_arr)
                
        x_input_arr1, y_arr1, x_input_arr2, y_arr2 = self.unbalanced_processing(x_input_arr, y_arr)   
        mor1, pca1, mor2, pca2 = self.get_rf_model_train_two_models(x_input_arr1, y_arr1, x_input_arr2, y_arr2)   
        print("trained finished test shape:" , x_input_arr.shape, y_arr.shape, X_test.shape, y_test.shape)
        
        #rfr = self.search_best_model(x_input_arr, y_arr)
        
        if single_featue == 'objectSizeChange':
            write_subDir = data_dir + predicted_video_dir +  "data_instance_xy_" + single_featue + "/minAcc_" + str(min_acc_threshold) + "/"
            if not os.path.exists(data_dir + predicted_video_dir +  "data_instance_xy_" + single_featue):
                os.mkdir(data_dir + predicted_video_dir +  "data_instance_xy_" + single_featue)
            
        else:
            write_subDir = data_dir + predicted_video_dir +  "data_instance_xy/"  + "minAcc_" + str(min_acc_threshold) + "/"
            if not os.path.exists(data_dir + predicted_video_dir +  "data_instance_xy/"):
                os.mkdir(data_dir + predicted_video_dir +  "data_instance_xy/")
            
        
        """
        if not os.path.exists(write_subDir):
            os.mkdir(write_subDir)

        model_file = write_subDir + "model_classifier_frame_rate.joblib" + "_exclusive_" + str(predicted_video_dir[:-1])  + ".pkl"
        
        #x_input_arr_file = write_subDir + "all_other_trained_x_instances.pkl"
        #write_pickle_data(x_input_arr, x_input_arr_file)
        _ = joblib.dump(mor1, model_file, compress=4)
        
        model_file = write_subDir + "model_classifier_resolution.joblib" + "_exclusive_" + str(predicted_video_dir[:-1])  + ".pkl"
        _ = joblib.dump(mor2, model_file, compress=4)
        """
        
        #pca_component = 3
        #X_test1, y_test1, X_test2, y_test2 = self.unbalanced_processing(X_test, y_test)
        mean_macro_f1, mean_micro_f1 = self.test_on_data_y_known_two_models(mor1, X_test, y_test[:, 0], pca1, mor2, X_test, y_test[:, 1], pca2)
        
        return mean_macro_f1, mean_micro_f1
    
    
        
if __name__== "__main__": 
    
    model_obj = ModelClassifier()
    #model_obj.train_model_one_video()
    
    
    #1. mixed videos ; Configuration  Prediction  Performance  on  Whole  VideoDataset
    #model_obj.test_on_multiple_mixed_video(min_acc_threshold)

    
    #2. exclusive one video, other videos are used as training dataset
    # this one video as testing and applied the adaptive configuration algorithm on it
    
    
    average_mean_macro_f1 = 0.0
    average_mean_micro_f1 = 0.0

    video_dir_lst_tested = video_dir_lst[0:5]    # [5:6]   # [5, 15]
    # prediction on one video
    for predicted_video_dir in video_dir_lst_tested:
                    
        #predicted_video_dir = 'output_021_dance/'     # select different video id for testing
        single_featue = 'objectSizeChange'   # full
        #mean_macro_f1, mean_micro_f1 = model_obj.train_rest_test_one_video_one_model(predicted_video_dir, min_acc_threshold, single_featue)
        mean_macro_f1, mean_micro_f1 = model_obj.train_rest_test_one_video_two_models(predicted_video_dir, min_acc_threshold, single_featue)
        
        # get applied result's acc and delay
        #model_obj.get_prediction_acc_delay(predicted_video_dir, min_acc_threshold)
        average_mean_macro_f1 += mean_macro_f1
        average_mean_micro_f1 += mean_micro_f1
        
        

    average_mean_macro_f1 /= len(video_dir_lst_tested) 
    average_mean_micro_f1 /= len(video_dir_lst_tested) 
    
    print ("final average f1 score ",  average_mean_macro_f1, average_mean_micro_f1)
    

    