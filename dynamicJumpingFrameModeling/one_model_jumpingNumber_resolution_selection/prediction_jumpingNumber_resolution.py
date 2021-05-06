# -*- coding: utf-8 -*-

# final version here


# logistic regression
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

import pickle
import sys
import os
import numpy as np
import time
import math
import joblib
import matplotlib.pyplot as plt
from blist import blist


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
#from skopt import BayesSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.decomposition import PCA

from sklearn import svm
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

from get_data_jumpingNumber_resolution import samplingResoDataGenerate
from get_data_jumpingNumber_resolution import video_dir_lst
from get_data_jumpingNumber_resolution import min_acc_threshold

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from data_file_process import write_pickle_data
from data_file_process import read_pickle_data
from common_plot import plotOneScatterLine
from common_plot import plotOneScatter

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/../..')

from profiling.common_prof import dataDir3
from profiling.common_prof import computeOKS_1to1
from profiling.common_prof import PLAYOUT_RATE
from profiling.common_prof import NUM_KEYPOINT   

#global dataDir3
dataDir3 = "../" + dataDir3

n_components=15

class ModelRegression(object):
    def __init__(self):
        #self.data_classification_dir = data_classification_dir  # input file directory
        pass
    
    
    def read_whole_data_instances(self, data_file):
        # Input:  data file
        # output: X and y 
        #data_file = self.data_classification_dir + "data_instance_xy.pkl"
        
        data = read_pickle_data(data_file)
        
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
            cv=5, scoring='neg_mean_absolute_error', verbose=0,  n_jobs=-1)
        
        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_
        
        
        rfr = RandomForestClassifier(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=0)
        #rfr = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)

        # Perform K-Fold CV
        scores = cross_val_score(rfr, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')  # scoring='neg_mean_absolute_error')
        print ("train_rf_cross_validation training scores r2: ", best_params, scores)
        
        return best_params


    def rf_classification_train_test(self, X, y):
        #classification here  # regression or  classification
        # train and test with random forest
        pca = PCA(n_components=min(n_components, X.shape[1]))
        X = pca.fit_transform(X)
        
        X_train, X_test, y_train, y_test = self.get_train_test_data(X, y)
    
        #sc_X = StandardScaler()
        #X_train = sc_X.fit_transform(X_train)
        #X_test = sc_X.transform(X_test)
        
        print ("get_rf_model_train  enter here: ", X.shape, y.shape)
        #rfr = RandomForestRegressor(max_depth=20)
        
        #rfr = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)


        """
        mor = RandomForestClassifier(max_depth=20)
        """
        svmc = svm.SVC(kernel='rbf', gamma=0.2857, C=100, decision_function_shape='ovr')

        mor = MultiOutputClassifier(svmc)
        print("get_rf_model_train: ", type(mor))
        
        mor.fit(X_train, y_train)  # (X, y)   # (X_train, y_train) # (X, y)  # 
        #print(mor.feature_importances_)
        
        """
        best_params = self.train_rf_cross_validation(X_train, y_train)  # (X, y) # (X_train, y_train)
        rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=0)
        rfr.fit(X_train, y_train) # (X, y)  # 
        """
        
        y_train_pred = mor.predict(X_train)
        #y_train_pred = list(map(lambda ele: round(ele[0]), y_train_pred))

        """
        r2_train = r2_score(y_train, y_train_pred, multioutput='uniform_average')
        rmse_train = mean_squared_error(y_train, y_train_pred) #, squared=False)
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
        
        
        
        
        y_test_pred = mor.predict(X_test)
        #y_test_pred = list(map(lambda ele: round(ele[0]), y_test_pred))
        #print ("rf_train_test y predicted config: ", y_test_pred)
        
        """
        r2_test = r2_score(y_test, y_test_pred, multioutput='uniform_average')
        mse_test = mean_squared_error(y_test, y_test_pred)  #, squared=False)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        print ("rf_train_test testing r2: ", r2_test, mse_test, mae_test)
        """
        mean_macro_f1 = 0.0
        mean_micro_f1 = 0.0
        for i in range(y_test.shape[1]):
            macro_f1 = precision_recall_fscore_support(y_test[:, i], y_test_pred[:, i], average='macro')
            micro_f1 = precision_recall_fscore_support(y_test[:, i], y_test_pred[:, i], average='micro')
            #print ("rf_classification_train_test testing micro, macro micro f1 score  ", i, macro_f1, micro_f1)
            mean_macro_f1 += macro_f1[2]
            mean_micro_f1 += micro_f1[2]
            
        mean_macro_f1 /= y_test.shape[1]
        mean_micro_f1 /= y_test.shape[1]
        print ("rf_classification_train_test testing mean_macro_f1 mean_micro_f1:", mean_macro_f1, mean_micro_f1)
        
        
        
        
        return mor, y_test, y_test_pred
    
    
    def train_SVR_cross_validation(self, X_train, y_train):
        # Perform Grid-Search cross validation
        
        gsc = GridSearchCV(
            estimator=SVR(),
            param_grid={
                'C': range(1, 8, 2),
                'kernel': ("rbf", 'poly'),
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
        pca = PCA(n_components=min(n_components, X.shape[1]))
        X = pca.fit_transform(X)
        
        X_train, X_test, y_train, y_test = self.get_train_test_data(X, y)
    
        """
        svr = make_pipeline(StandardScaler(), SVR(C=5.0, kernel = 'rbf', epsilon=0.2))
        svr.fit(X_train, y_train)
        
        """
        
        """
        best_params = self.train_SVR_cross_validation(X_train, y_train)  # (X, y) # (X_train, y_train)
        svr = SVR(C=best_params["C"], kernel=best_params['kernel'])
        svr.fit(X_train, y_train) # (X, y)  # 
        """
        
        svr = SVR(C=5, kernel='poly')
        svr = MultiOutputRegressor(svr)
        svr.fit(X_train, y_train) # (X, y)  # 

        y_train_pred = svr.predict(X_train)
        y_train_pred = list(map(lambda ele: round(ele), y_train_pred))

        r2 = r2_score(y_train, y_train_pred, multioutput='uniform_average')
        rmse = mean_squared_error(y_train, y_train_pred)  #, squared=False)
        mae = mean_absolute_error(y_train, y_train_pred)
        
        print ("svr train_test training result r2: ", r2, rmse, mae)
        
        y_test_pred = svr.predict(X_test)
        y_test_pred = list(map(lambda ele: round(ele), y_test_pred))
        print ("svr_train_test y predicted config: ", y_test_pred)
        
        r2 = r2_score(y_test, y_test_pred, multioutput='uniform_average')
        rmse = mean_squared_error(y_test, y_test_pred) #, squared=False)
        mae = mean_absolute_error(y_test, y_test_pred)
        
        print ("svr train_test testing r2: ", r2, rmse, mae)
        return y_test, y_test_pred


    def plot_result(self, y_test, y_test_pred, output_dir):
        
        fig = plotOneScatter(y_test, y_test_pred, "y_true", "y_pred", "")
        out_file_path = output_dir +  "test_res.pdf"
        fig.savefig(out_file_path)
        
        

    def train_test_model_on_one_video(self):
        
        #global dataDir3
        #dataDir3 = "../" + dataDir3
        
        for video_dir in video_dir_lst[3:4]:    # [4:5]:   #[1:2]:  # [1:2]:  #[0:1]:        #[1:2]:
            
            data_pickle_dir = dataDir3 + video_dir + "jumping_number_result/" 
            subDir = data_pickle_dir + "jumpingNumber_resolution_selection/intervalFrm-10_speedType-ema_minAcc-0.9/"
            data_file = subDir + "data_instance_speed_jumpingNumber_resolution_xy.pkl"
            X, y = self.read_whole_data_instances(data_file)
            
            y_test, y_test_pred = self.rf_classification_train_test(X, y)
            
            output_dir = data_pickle_dir
            self.plot_result(y_test, y_test_pred, output_dir)


    def get_rf_model_train(self, X, y):
        #get the random forest model
        # X, y are the trianing and validation data
        pca = PCA(n_components=min(n_components, X.shape[1])).fit(X)
        X = pca.transform(X)
            
        #sc_X = StandardScaler()
        #X_train = sc_X.fit_transform(X_train)
        #X_test = sc_X.transform(X_test)

        #rfr = RandomForestRegressor(max_depth=20, random_state=0)
        
        mor = RandomForestClassifier(max_depth=20, n_estimators = 20, random_state=0)
        
        """
        
        print ("get_rf_model_train  enter here: ", X.shape, y.shape)
        svmc = svm.SVC(kernel='rbf', gamma=0.2857, C=100, decision_function_shape='ovr')

        mor = MultiOutputClassifier(svmc)
        print("get_rf_model_train: ", type(mor))
        """
        
        mor.fit(X, y)   
        #print(rfr.feature_importances_)
        
        """
        best_params = self.train_rf_cross_validation(X, y)  # (X, y) # (X_train, y_train)
        rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=0)
        rfr.fit(X, y) # (X, y)  # 
        """
        
        y_train_pred = mor.predict(X)
        #y_train_pred = list(map(lambda ele: round(ele[0]), y_train_pred))
        """
        r2_train = r2_score(y, y_train_pred, multioutput='uniform_average')
        rmse_train = mean_squared_error(y, y_train_pred)  #, squared=False)
        mae_train = mean_absolute_error(y, y_train_pred)
        print ("get_rf_model_train training result r2: ", r2_train, rmse_train, mae_train)   
        """
        
        mean_macro_f1 = 0.0
        mean_micro_f1 = 0.0
        for i in range(y.shape[1]):
            macro_f1 = precision_recall_fscore_support(y_train_pred[:, i], y[:, i], average='macro')
            micro_f1 = precision_recall_fscore_support(y_train_pred[:, i], y[:, i], average='micro')
            #print ("rf_classification_train_test testing micro, macro micro f1 score  ", i, macro_f1, micro_f1)
            mean_macro_f1 += macro_f1[2]
            mean_micro_f1 += micro_f1[2]
            
        mean_macro_f1 /= y.shape[1]
        mean_micro_f1 /= y.shape[1]
        print ("get_rf_model_train training mean_macro_f1 mean_micro_f1:", mean_macro_f1, mean_micro_f1)
        
        
        
        return mor, pca
        

    def train_model_multiple_videos(self, min_acc):
        # combine several videos together
        # select all the rest  video except one video  to train and save the model
        
        #global dataDir3
        #dataDir3 = "../" + dataDir3
        
        for i, video_dir in enumerate(video_dir_lst[0:35]):    # combine all to generate more data
            
            data_pickle_dir = dataDir3 + video_dir + "jumping_number_result/" 
            subDir = data_pickle_dir + "jumpingNumber_resolution_selection/intervalFrm-10_speedType-ema_minAcc-" + str(min_acc) + "/"
            data_file = subDir + "data_instance_speed_jumpingNumber_resolution_objectSizeRatio_xy.pkl"
            X, y = self.read_whole_data_instances(data_file)
            
            print("X, y: ", X.shape, y.shape)
            
            if i == 0:
                x_input_arr = X
                y_arr = y
            
            else:
                x_input_arr = np.vstack((x_input_arr, X))
                y_arr = np.vstack((y_arr, y))

            
        print("x_input_arr , y: ", x_input_arr.shape, y_arr.shape)

        rfr, y_test, y_test_pred = self.rf_classification_train_test(x_input_arr, y_arr)
            
        """
        #y_test, y_test_pred = self.svr_train_test (x_input_arr, y_arr)
        output_dir = dataDir3 + "dynamic_jumpingNumber_resolution_selection_output/"
        write_subDir = output_dir + "intervalFrm-10_speedType-ema_minAcc-" + str(min_acc) +"/"
        model_file = write_subDir + "model_regression_jumping_frm_all.joblib.pkl"        
        _ = joblib.dump(rfr, model_file, compress=4)
        
        output_dir = data_pickle_dir
        self.plot_result(y_test, y_test_pred, output_dir)
        """
    
    def train_rest_test_one_video(self, predicted_video_dir, min_acc, single_featue):
        # pick one video ast test and the rest as the training
        # select one video to train and test
        #predicted_video_dir = 'output_021_dance/'     # select different video id for testing
        X_test = None
        y_test = None
        first_flag = True  # first video
        for i, video_dir in enumerate(video_dir_lst[0:10]):    # combine all to generate more data
            
            data_pickle_dir = dataDir3 + video_dir + "jumping_number_result/" 
            #data_pickle_dir = dataDir3 + video_dir + "jumping_number_result_each_frm/" 

            subDir = data_pickle_dir + "jumpingNumber_resolution_selection/intervalFrm-10_speedType-ema_minAcc-" + str(min_acc) + "/"
            data_file = subDir + "data_instance_speed_jumpingNumber_resolution_objectSizeRatio_xy.pkl"
            X, y = self.read_whole_data_instances(data_file)
            
            #print("X, y: ", X.shape, y.shape)
            
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
                    y_arr = np.vstack((y_arr, y))
                
        # total is 159 size,  dimension is 
        if single_featue == 'keypoint_velocity_removed':    # velocity has 57 
            x_input_arr = x_input_arr[:, 36:]  # np.hstack(X_test[:, 0:37], X_test[:, 0:37])
            #y_arr = y_arr[:, 36:]  # np.hstack(X_test[:, 0:37])  #   y_test[:, 0:57]
        elif single_featue == 'relative_velocity_removed':
            x_input_arr = np.hstack((x_input_arr[:, 0:36], x_input_arr[:, 57:]))
            #y_arr = np.hstack((y_arr[:, 0:36], y_arr[:, 57:]))
        elif single_featue == 'objectSizeChange_removed':
            x_input_arr = np.hstack((x_input_arr[:, 0:57], x_input_arr[:, 59:]))
            #y_arr = np.hstack((y_arr[:, 0:57], y_arr[:, 59:]))
        elif single_featue == 'opticalFlow_removed':
            x_input_arr = x_input_arr[:, 0:59]   #  159 total
            #y_arr = y_arr[:, 0:59]   # total     
                        
        
        rfr, pca = self.get_rf_model_train(x_input_arr, y_arr)   
        print("trained finished test shape:" , x_input_arr.shape, y_arr.shape, X_test.shape, y_test.shape)
        write_subDir = dataDir3 + predicted_video_dir + "jumping_number_result/"  + "jumpingNumber_resolution_selection/intervalFrm-10_speedType-ema_minAcc-" + str(min_acc) + "/"

        #write_subDir = dataDir3 + predicted_video_dir + "jumping_number_result_each_frm/"  + "jumpingNumber_resolution_selection/intervalFrm-10_speedType-ema_minAcc-" + str(min_acc) + "/"
        model_file = write_subDir + "model_regression.joblib" + "_exclusive_" + str(predicted_video_dir[:-1])  + ".pkl"
        
        x_input_arr_file = write_subDir + "trained_x_instances.pkl"
        write_pickle_data(x_input_arr, x_input_arr_file)
        
        _ = joblib.dump(rfr, model_file, compress=4)
        
        
        # total is 159 size,  dimension is 
        if single_featue == 'keypoint_velocity_removed':    # velocity has 57 
            X_test = X_test[:, 36:]  # np.hstack(X_test[:, 0:37], X_test[:, 0:37])
            #y_test = y_test[:, 36:]  # np.hstack(X_test[:, 0:37])  #   y_test[:, 0:57]
        elif single_featue == 'relative_velocity_removed':
            X_test = np.hstack((X_test[:, 0:36], X_test[:, 57:]))
            #y_test = np.hstack((y_test[:, 0:36], y_test[:, 57:]))
        elif single_featue == 'objectSizeChange_removed':
            X_test = np.hstack((X_test[:, 0:57], X_test[:, 59:]))
            #y_test = np.hstack((y_test[:, 0:57], y_test[:, 59:]))
        elif single_featue == 'opticalFlow_removed':
            X_test = X_test[:, 0:59]   #  159 total
            #y_test = y_test[:, 0:59]   # total     
            
        #pca_component = 10
        mean_macro_f1, mean_micro_f1, acc_fr, acc_res, aver_acc = self.test_on_data_y_known(rfr, X_test, y_test, pca)
        
        #write_subDir = dataDir3 + predicted_video_dir + "jumping_number_result/"  + "jumpingNumber_resolution_selection/intervalFrm-10_speedType-ema_minAcc-0.92/"
        #out_data_pickle_file = write_subDir + "predicted_y_out.pkl"       # with other videos
        #write_pickle_data(y_test_pred, out_data_pickle_file)
     
        return mean_macro_f1, mean_micro_f1, acc_fr, acc_res, aver_acc

    def test_on_data_y_unknown(self, model, X_test, pca):
        # test on a test data with model already trained, y label unknown
        # X_test is just one instance
        #X_test = np.reshape(X_test, (1, -1))
        #print ("test_on_data_y_unknown X_test shape: ", X_test)
        
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
        #y_test_pred = list(map(lambda ele: round(ele), y_test_pred))
        #print ("rf_train_test y predicted config: ", X_test, y_test, y_test_pred)
        
        """
        r2 = r2_score(y_test, y_test_pred, multioutput='uniform_average')
        rmse = mean_squared_error(y_test, y_test_pred) # , squared=False)
        mae = mean_absolute_error(y_test, y_test_pred)
        print ("test_on_data_y_known testing r2: ", r2, rmse, mae, y_test_pred[0])
        """
        
        mean_macro_f1 = 0.0
        mean_micro_f1 = 0.0
        for i in range(y_test.shape[1]):
            macro_f1 = precision_recall_fscore_support(y_test[:, i], y_test_pred[:, i], average='macro')
            micro_f1 = precision_recall_fscore_support(y_test[:, i], y_test_pred[:, i], average='micro')
            #print ("rf_classification_train_test testing micro, macro micro f1 score  ", i, macro_f1, micro_f1)
            mean_macro_f1 += macro_f1[2]
            mean_micro_f1 += micro_f1[2]
            
        mean_macro_f1 /= y_test.shape[1]
        mean_micro_f1 /= y_test.shape[1]
        
        acc_fr = accuracy_score(y_test[:, 0], y_test_pred[:, 0])
        acc_res = accuracy_score(y_test[:, 1], y_test_pred[:, 1])
        aver_acc = (acc_fr + acc_res) / 2.0
        print ("test_on_data_y_known_two_models mean_macro_f1 mean_micro_f1:", mean_macro_f1, mean_micro_f1, acc_fr, acc_res, aver_acc)
            
        return mean_macro_f1, mean_micro_f1, acc_fr, acc_res, aver_acc


    def get_prediction_acc_delay(self, predicted_video_dir, min_acc):
        # get the predicted video's accuracy anpredicted_video_dird delay
        #  predicted_video_id as the testing data
        # predicted_video_dir:  such as output_021_dance
        # predicted_out_file is the prediction jumping number and delay
        
        interval_frm = 1
        
        data_pose_keypoint_dir = dataDir3 + predicted_video_dir

        data_pickle_dir = dataDir3 + predicted_video_dir + 'frames_pickle_result/'
        ResoDataGenerateObj = samplingResoDataGenerate()
        config_est_frm_arr, acc_frame_arr, spf_frame_arr = ResoDataGenerateObj.get_data_numpy(data_pose_keypoint_dir, data_pickle_dir)
     
        print ("get_prediction_acc_delay config_est_frm_arr: ", config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape)
        
        output_pickle_dir = dataDir3 + predicted_video_dir + "jumping_number_result/" 
        subDir = output_pickle_dir + "jumpingNumber_resolution_selection/intervalFrm-10_speedType-ema_minAcc-" + str(min_acc) + "/"
        
        
        #model_file = subDir + "model_regression.joblib" + "_exclusive_" + str(predicted_video_dir[:-1])  + ".pkl"       # with other videos

        # read the model 
        model_dir = dataDir3 + "dynamic_jumpingNumber_resolution_selection_output/intervalFrm-10_speedType-ema_minAcc-" + str(min_acc) +"/"
        model_file = model_dir + "model_regression_jumping_frm_all.joblib.pkl"    
        
        
        test_x_instances_file = subDir + "trained_x_instances.pkl"
        X = read_pickle_data(test_x_instances_file)

        pca = PCA(n_components=min(n_components, X.shape[1])).fit(X)
        
        model = joblib.load(model_file)


        # get estimated accuracy and delay
        # start from 2nd frame 
        reso_curr = 0   # current resolution
        prev_indx = 0
        current_indx = 1
        FRM_NO =  spf_frame_arr[reso_curr].shape[0]                # total frame no for the whole video

        # use predicted result to apply to this new video and get delay and accuracy
        delay_arr = []      
        up_to_delay = max(0, spf_frame_arr[reso_curr][0] - 1.0/PLAYOUT_RATE)         # up to current delay, processing time - streaming time
        
        acc_arr = []
        acc_seg = 0        # current segment's acc
        
        segment_indx = 0
        
        jumping_frm_number_lst = []
        
        arr_ema_absolute_speed = np.zeros((NUM_KEYPOINT, 3))
        arr_ema_relative_speed  = np.zeros((8, 3))    # 8 points to calculate relative speed
        
        while(current_indx < FRM_NO):
            # get jumping frame number 
            # get feature x for up to current frame
            current_frm_est = config_est_frm_arr[reso_curr][current_indx]  # detected_est_frm_arr[current_indx] #  current frame is detected, so we use

            #print("detected_est_frm_arr: ", len(detected_est_frm_arr), current_indx, prev_used_indx)
            prev_frm_est = config_est_frm_arr[reso_curr][prev_indx]  # last frame used detected_est_frm_arr[prev_used_indx]
                
            arr_ema_absolute_speed = ResoDataGenerateObj.get_absolute_speed(current_frm_est, prev_frm_est, interval_frm, arr_ema_absolute_speed)

            # get relative speed 
            arr_ema_relative_speed = ResoDataGenerateObj.get_relative_speed_to_body_center(current_frm_est, prev_frm_est, interval_frm, arr_ema_relative_speed)
            
            feature_x_absolute = ResoDataGenerateObj.get_feature_x(arr_ema_absolute_speed)

            feature_x_relative = ResoDataGenerateObj.get_feature_x(arr_ema_relative_speed)
            
            feature_x_object_size = ResoDataGenerateObj.get_object_size_x(current_frm_est)      
                    
                    
            feature_x = np.hstack((feature_x_absolute, feature_x_relative, feature_x_object_size))
                    
            predicted_y = self.test_on_data_y_unknown(model, feature_x, pca)
            
            #print ("get_prediction_acc_delay: ", predicted_y)

            jumping_frm_number = int(predicted_y[0][0])
            
            reso_curr = int(predicted_y[0][1])
            
            
            # get accuracy of this segment
            acc_seg = self.get_accuracy_segment(current_indx, jumping_frm_number, reso_curr, config_est_frm_arr)
            
            # get delay up to this segment
            up_to_delay = max(0, spf_frame_arr[reso_curr][current_indx] - (1.0/PLAYOUT_RATE) * jumping_frm_number)
            delay_arr.append(up_to_delay)
            
            acc_arr.append(acc_seg)
            
            prev_indx = current_indx            # update prev_indx as current index
            
            current_indx += jumping_frm_number         # not jumping_frm_number + 1
            segment_indx += 1
            
            #jumping_frm_number_lst.append(jumping_frm_number)
            
            #print ("get_prediction_acc_delay current_indx: ", FRM_NO, current_indx, segment_indx, acc_seg, up_to_delay)
            
            #break   # debug only

        acc_arr = np.asarray(acc_arr)
        delay_arr = np.asarray(delay_arr)
        print ("get_prediction_acc_delay acc_arr, delay_arr: ",  acc_arr, delay_arr)
        
        detect_out_result_dir = subDir + "video_applied_detection_result/"
        if not os.path.exists(detect_out_result_dir):

            os.mkdir(detect_out_result_dir)

        arr_acc_segment_file = detect_out_result_dir + "arr_acc_segment_.pkl"
        arr_delay_up_to_segment_file = detect_out_result_dir + "arr_delay_up_to_segment_.pkl"
        write_pickle_data(acc_arr, arr_acc_segment_file)
        write_pickle_data(delay_arr, arr_delay_up_to_segment_file)
        
        return 


    def get_accuracy_segment(self, start_frm_indx, jumping_frm_number, reso_curr, config_est_frm_arr):
        # get the accuracy when jumping frm number
          
          
        accumulated_acc = 0
        ref_pose = config_est_frm_arr[0][start_frm_indx]     # reference pose ground truth
          
        end_indx = start_frm_indx + jumping_frm_number
        curr_indx = start_frm_indx
        while (curr_indx < end_indx):
          
          # get accuracy
            if curr_indx >= config_est_frm_arr.shape[1]:  # finished video streaming
                jumping_frm_number = curr_indx - start_frm_indx   # last segment
                break
            curr_pose = config_est_frm_arr[reso_curr][curr_indx]
            oks = computeOKS_1to1(ref_pose, curr_pose, sigmas = None)     # oks with reference pose
            accumulated_acc += oks
      
            curr_indx += 1
        average_acc = accumulated_acc/(jumping_frm_number)  # jumping_frm_number include the start_frame_index
        
        # print ("get_accuracy_segment average_acc: ", average_acc)
        return average_acc
    

if __name__== "__main__": 
    
    model_obj = ModelRegression()
    #model_obj.train_test_model_on_one_video()
    
    # train and save model with multiple video dataset
    #model_obj.train_model_multiple_videos(min_acc_threshold)
    
    # exclusive one video, other videos are used as training dataset
    # this one video as testing and applied the adaptive configuration algorithm on it
        
    # prediction on one video
    average_mean_macro_f1 = 0.0
    average_mean_micro_f1 = 0.0
    aver_accur = 0.0 
    video_dir_lst_tested = video_dir_lst[0:5]
    for predicted_video_dir in video_dir_lst_tested:
                    
        #predicted_video_dir = 'output_021_dance/'     # select different video id for testing
        single_featue = 'all' # 'relative_velocity_removed' # 'all'
        
        mean_macro_f1, mean_micro_f1, acc_fr, acc_res, aver_acc = model_obj.train_rest_test_one_video(predicted_video_dir, min_acc_threshold, single_featue)
        #model_obj.get_prediction_acc_delay(predicted_video_dir, min_acc_threshold)
        
        
        average_mean_macro_f1 += mean_macro_f1
        average_mean_micro_f1 += mean_micro_f1
        aver_accur += aver_acc
        
    average_mean_macro_f1 /= len(video_dir_lst_tested) 
    average_mean_micro_f1 /= len(video_dir_lst_tested) 
    aver_accur /= len(video_dir_lst_tested)
    print ("final average f1 score ",  average_mean_macro_f1, average_mean_micro_f1, aver_accur)
    