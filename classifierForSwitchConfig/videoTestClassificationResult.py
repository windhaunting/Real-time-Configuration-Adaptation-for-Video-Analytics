#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:40:23 2019

@author: fubao
"""

#test the accuracy and delay after applying classification result



import sys
import os
import numpy as np


from collections import defaultdict

from common_classifier import readProfilingResultNumpy
from common_classifier import get_cmu_model_config_acc_spf
from common_classifier import getAccSpfArrAllVideo

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir3
from profiling.common_prof import PLAYOUT_RATE


def parseFrmIdHelper(row):
    '''
    each row format example: ../input_output/one_person_diy_video_dataset/005_dance_frames/000001.jpg
    '''
    
    #print ("rowrowrowrowrow: ", row)
    video_id = int(row[0].split("/")[-2].split("_")[0])
    frm_id = int(row[0].split('/')[-1].split('.')[0])-1          # index starting at 0
    
    return [video_id, frm_id]
    
def getAccFromEachConfigHelper(row, dict_acc_frm_arr):
    '''
    each row is a video_id, frm_id, config id
    '''
    video_id = row[0]
    frm_id = row[1]
    row_config = row[2]
    
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
    
    video_frm_id_arr = np.apply_along_axis(parseFrmIdHelper, 1, x_video_frm_id_arr)    
    
    combined_gt_arr = np.hstack((video_frm_id_arr, y_gt_out))
    acc_gt_arr = np.apply_along_axis(getAccFromEachConfigHelper, 1, combined_gt_arr, dict_acc_frm_arr)
    print ("acc_gt_arr: ", acc_gt_arr.shape, np.mean(acc_gt_arr))
    
    
    combined_pred_arr = np.hstack((video_frm_id_arr, y_pred))
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

    spf = spf_frm_arr[row_config][frm_id]   # interval 1 second actually
    
    #print ("row_config", row_config)
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
    delay_gt_arr = spf_gt_arr*PLAYOUT_RATE - 1
    #print ("delay_gt_arr gt test video clip length: %s; delay: %s ", delay_gt_arr.shape[0], np.sum(delay_gt_arr), spf_gt_arr)
    
    
    combined_pred_arr = np.hstack((video_video_frm_arr, y_pred))
    spf_pred_arr = np.apply_along_axis(getDelayFromEachConfigHelper, 1, combined_pred_arr, dict_spf_frm_arr)
    delay_pred_arr = spf_pred_arr*PLAYOUT_RATE - 1
    #print ("delay_pred_arr, pred test video clip length;  delay,  ", delay_pred_arr.shape[0], np.sum(delay_pred_arr))
    
    
    total_delay_gt = []
    for d in delay_gt_arr:
        if d < 0:
            d = 0
        total_delay_gt.append(d)
        
    total_delay_pred = []
    for d in delay_pred_arr:
        if d < 0:
            d = 0
        total_delay_pred.append(d)
    
    aver_total_delay_gt = sum(total_delay_gt)/len(total_delay_gt)   
    aver_total_delay_pred = sum(total_delay_pred)/len(total_delay_pred)   
    print ("total_delay_gt pred  delay,  ", aver_total_delay_gt, aver_total_delay_pred)
    #print ("delay_pred_arr  delay,  ", delay_pred_arr)

    

def executeTest():
    dict_acc_frm_arr, dict_spf_frm_arr = getAccSpfArrAllVideo()
    
    #from classifier_svm import combineMultipleVideoDataTrainTest
    from classifier_rft import combineMultipleVideoDataTrainTest
    #from classifier_xgboost import combineMultipleVideoDataTrainTest
    #from classifier_logisticReg import combineMultipleVideoDataTrainTest
    
    x_video_frm_id_arr, y_pred, y_gt_out = combineMultipleVideoDataTrainTest()

    testVideoClassificationResultAcc(dict_acc_frm_arr, x_video_frm_id_arr, y_pred, y_gt_out)
    
    testVideoClassificationResultDelay(dict_spf_frm_arr, x_video_frm_id_arr, y_pred, y_gt_out)
    
    
if __name__== "__main__": 
    
    data_test_classification__dir =  dataDir3 + 'test_classification_result/'
    
    if not os.path.exists(data_test_classification__dir):
        os.mkdir(data_test_classification__dir)
        
    
    executeTest()
    