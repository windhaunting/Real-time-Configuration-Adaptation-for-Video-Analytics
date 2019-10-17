#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:40:23 2019

@author: fubao
"""

#test the accuracy and delay after applying classification result



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

from common_classifier import readProfilingResultNumpy
from common_classifier import load_data_all_features
from common_classifier import read_config_name_from_file
from common_classifier import feature_selection

from common_plot import plotScatterLineOneFig
from common_plot import plotOneScatterLine
from common_plot import plotOneBar
from common_plot import plotTwoLinesOneFigure

from classifier_svm import combineMultipleVideoDataTrainTest

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir3


def parseFrmIdHelper(row):
    '''
    each row format example: ../input_output/one_person_diy_video_dataset/005_dance_frames/000002.jpg
    '''
    
    #print ("rowrowrowrowrow: ", row)
    video_id = int(row[0].split("/")[-2].split("_")[0])
    frm_id = int(row[0].split('/')[-1].split('.')[0])
    
    return [video_id, frm_id]
    
def getAccFromEachConfigHelper(row, dict_acc_frm_arr):
    '''
    each row is a video_id, frm_id, config id
    '''
    row_config = row[2]
    video_id = row[0]
    frm_id = row[1]
    
    acc_frm_arr = dict_acc_frm_arr[video_id]

    acc = acc_frm_arr[row_config][frm_id]   # interval 1second actually
    
    return acc




def testVideoClassificationResultAcc(dict_acc_frm_arr, x_video_frm_id_arr, y_pred, y_gt_out):
    '''
    test video classification
    '''
    
    # parse integer video id and frame id
    x_video_frm_id_arr = x_video_frm_id_arr.reshape(x_video_frm_id_arr.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    
    video_video_frm_arr = np.apply_along_axis(parseFrmIdHelper, 1, x_video_frm_id_arr)    
    
    combined_gt_arr = np.hstack((video_video_frm_arr, y_gt_out))
    acc_gt_arr = np.apply_along_axis(getAccFromEachConfigHelper, 1, combined_gt_arr, dict_acc_frm_arr)
    print ("acc_gt_arr: ", acc_gt_arr.shape, np.mean(acc_gt_arr))
    
    combined_pred_arr = np.hstack((video_video_frm_arr, y_pred))
    acc_pred_arr = np.apply_along_axis(getAccFromEachConfigHelper, 1, combined_pred_arr, dict_acc_frm_arr)
    print ("acc_pred_arr: ", acc_pred_arr.shape, np.mean(acc_pred_arr))
    


def getDelayFromEachConfigHelper(row, dict_spf_frm_arr):
    '''
    each row is a video_id, frm_id, config id
    '''
    row_config = row[2]
    video_id = row[0]
    frm_id = row[1]
    
    spf_frm_arr = dict_spf_frm_arr[video_id]

    spf = spf_frm_arr[row_config][frm_id]   # interval 1second actually
    
    return spf

    
 
def testVideoClassificationResultDelay(dict_spf_frm_arr, x_video_frm_id_arr, y_pred, y_gt_out):
    '''
    test video classification
    '''
    
    # parse integer video id and frame id
    x_video_frm_id_arr = x_video_frm_id_arr.reshape(x_video_frm_id_arr.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    
    video_video_frm_arr = np.apply_along_axis(parseFrmIdHelper, 1, x_video_frm_id_arr)    
    
    combined_gt_arr = np.hstack((video_video_frm_arr, y_gt_out))
    spf_gt_arr = np.apply_along_axis(getDelayFromEachConfigHelper, 1, combined_gt_arr, dict_spf_frm_arr)
    delay_gt_arr = spf_gt_arr*25 - 1
    print ("delay_gt_arr gt test video clip length: %s; delay: %s ", delay_gt_arr.shape[0], np.sum(delay_gt_arr))
    
    
    combined_pred_arr = np.hstack((video_video_frm_arr, y_pred))
    spf_pred_arr = np.apply_along_axis(getDelayFromEachConfigHelper, 1, combined_pred_arr, dict_spf_frm_arr)
    delay_pred_arr = spf_pred_arr*25 - 1
    print ("delay_pred_arr, pred test video clip length;  delay,  ", delay_pred_arr.shape[0], np.sum(delay_pred_arr))
    
    
    total_delay_gt = 0
    for d in delay_gt_arr:
        total_delay_gt += d
        if total_delay_gt < 0:
            total_delay_gt = 0
            
    
    total_delay_pred = 0
    for d in delay_pred_arr:
        total_delay_pred += d
        if total_delay_pred < 0:
            total_delay_pred = 0
        
    print ("total_delay_gt pred  delay,  ", total_delay_gt, total_delay_pred)
    #print ("delay_pred_arr  delay,  ", delay_pred_arr)

def getAccSpfArrAllVideo():
    
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                'output_003_dance/', 'output_004_dance/',  \
                'output_005_dance/', 'output_006_yoga/', \
                'output_007_yoga/', 'output_008_cardio/', \
                'output_009_cardio/', 'output_010_cardio/']
        
    dict_acc_frm_arr = defaultdict()
    dict_spf_frm_arr = defaultdict()
    for video_dir in video_dir_lst[0:8]:
        data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
    
        acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)
        video_id = int(video_dir.split("_")[1])
        
        dict_acc_frm_arr[video_id] = acc_frame_arr
        dict_spf_frm_arr[video_id] = spf_frame_arr
        
    return dict_acc_frm_arr, dict_spf_frm_arr
    

def executeTest():
    dict_acc_frm_arr, dict_spf_frm_arr = getAccSpfArrAllVideo()
    
    x_video_frm_id_arr, y_pred, y_gt_out = combineMultipleVideoDataTrainTest()

    testVideoClassificationResultAcc(dict_acc_frm_arr, x_video_frm_id_arr, y_pred, y_gt_out)
    
    testVideoClassificationResultDelay(dict_spf_frm_arr, x_video_frm_id_arr, y_pred, y_gt_out)
    
    
if __name__== "__main__": 
    
    data_test_classification__dir =  dataDir3 + 'test_classification_result/'
    
    if not os.path.exists(data_test_classification__dir):
        os.mkdir(data_test_classification__dir)
        
    
    executeTest()
    