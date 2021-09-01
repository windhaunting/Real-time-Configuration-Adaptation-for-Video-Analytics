#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:40:23 2019

@author: fubao
"""

#test the accuracy and delay after applying classification result



import sys
import os

from common_classifier import getAccSpfArrAllVideo

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir3

    

def executeTest():
    dict_acc_frm_arr, dict_spf_frm_arr = getAccSpfArrAllVideo()
    
    #from classifier_svm import combineMultipleVideoDataTrainTest
    from classifier_rft import execute_all_video_train_test
    #from classifier_xgboost import combineMultipleVideoDataTrainTest
    #from classifier_logisticReg import combineMultipleVideoDataTrainTest
    
    x_video_frm_id_arr, y_pred, y_gt_out = execute_all_video_train_test()

   
    
    
if __name__== "__main__": 
    
    data_test_classification__dir =  dataDir3 + 'test_classification_result/'
    
    if not os.path.exists(data_test_classification__dir):
        os.mkdir(data_test_classification__dir)
        
    
    executeTest()
    