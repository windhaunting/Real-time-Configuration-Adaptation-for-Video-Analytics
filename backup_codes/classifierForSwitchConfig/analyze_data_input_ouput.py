#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:36:05 2019

@author: fubao
"""

# analyse the input data and selected output y for traditional model and neural network models

import sys
import os
import csv
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight

#from data_proc_input_NN import get_all_video_x_y_data

from common_classifier import getAccSpfArrAllVideo
from common_classifier import read_all_config_name_from_file
from common_plot import plotScatterLineOneFig
from common_plot import plotTwoSubplots
from common_plot import plotThreeSubplots

from common_plot import plot_bar_distribution

from analyze_data_config_equivalence import get_acc_or_spf_from_frm

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir3
from profiling.common_prof import frameRates
from profiling.common_prof import PLAYOUT_RATE



def idx_2_config_reso(row, id_config_dict):
    # transfer id to a reso integer
    #print ("idx_confrrrr", row)
    idx_conf = int(row)
    config = id_config_dict[idx_conf]
    reso = int(config.split("-")[0].split("x")[1])
    
    return reso

def idx_2_config_frm_rate(row, id_config_dict):
    # transfer id to a frame rate integer
    idx_conf = int(row)
    config = id_config_dict[idx_conf]
    frm_rt = int(config.split("-")[1])
    
    return frm_rt
    
    
def plot_feature_out_relation(x_feature_train, y_train_gt, id_config_dict, data_classification_dir_out):
    # plot the feature with config relations
    
    # plot feature with reso
    
    # plot feature with frame_rate
    x_video_fram_path_arr = x_feature_train[:, 0]
    
    feature_num = x_feature_train.shape[1]
    print ("y_train_gt", y_train_gt.shape)
    reso_arr = np.apply_along_axis(idx_2_config_reso, 1, y_train_gt, id_config_dict)
    frmRt_arr = np.apply_along_axis(idx_2_config_frm_rate, 1, y_train_gt, id_config_dict)
    print ("reso_arr", reso_arr.shape)
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(data_classification_dir_out + "feature_resolution.pdf")

    for feat_ind in range(1, feature_num):
        #xxx
        xList = range(0, x_feature_train.shape[0])
        yList1 = x_feature_train[:, feat_ind]
        yList2 = reso_arr
        yList3 = frmRt_arr
        yList4 = y_train_gt     # config id
        
        x_label = "Segment"
        y_label_1 = "Feature-" + str(feat_ind)
        y_label_2 = "Resolution"
        y_label_3 = "Frm Rate"
        y_label_4 = "Config_id"
        title_name = ""
        #print ("x_lst before ", xList.shape, yList1.shape)
        '''
        fig = plotTwoLinesOneFigure(x_lst, y_lst, xlabel, ylabel, title_name)
        pdf.savefig(fig)
        print ("x_lst", x_lst.shape)
        y_lst = frmRt_arr
        xlabel= "Feature-" + str(feat_ind)
        ylabel= "Frm Rate"
        title_name = ""
        '''
        fig = plotScatterLineOneFig(xList, yList1, x_label, y_label_1, title_name)    
        #fig = plotThreeSubplots(xList, yList1, yList2, yList3, x_label, y_label_1, y_label_2, y_label_3, title_name)
        pdf.savefig(fig)
        
        if feat_ind == 1:
            fig = plotScatterLineOneFig(xList, yList2, x_label, y_label_2, title_name)    
            pdf.savefig(fig)
            
            fig = plotScatterLineOneFig(xList, yList3, x_label, y_label_3, title_name)    
            pdf.savefig(fig)
            
            fig = plotScatterLineOneFig(xList, yList4, x_label, y_label_4, title_name)    
            pdf.savefig(fig)
            
    pdf.close()

    
def save_x_y_out_data(x_feature_train, y_train_gt, id_config_dict, classifier_result_dir, data_classification_dir_out):
    
    # load y_predicted
    #y_test_pred = np.load(classifier_result_dir + 'y_test_used_pred.pkl', allow_pickle=True)          # frame by frame but it also calculate the 1sec interval with each frame starting
        
    #load test_video_frm_id_arr predicted
    train_video_frm_id_arr = np.load(classifier_result_dir + 'all_video_frm_id_arr.pkl', allow_pickle=True)          # frame by frame but it also calculate the 1sec interval with each frame starting
    
    dict_acc_frm_arr, dict_spf_frm_arr = getAccSpfArrAllVideo()
    
    print ("train_video_frm_id_arr train_video_frm_id_arr ", train_video_frm_id_arr)
        
    y_train_gt_acc_arr= get_acc_or_spf_from_frm(train_video_frm_id_arr, dict_acc_frm_arr, y_train_gt)
        
    
    y_train_gt_spf_arr= get_acc_or_spf_from_frm(train_video_frm_id_arr, dict_spf_frm_arr, y_train_gt) # *PLAYOUT_RATE-1
      

    total_frm_config_reso = []
    
    #print ("frm_Config_reso:" , frm_Config_reso)
                
    combine_X = np.hstack((train_video_frm_id_arr, x_feature_train))

    np.savetxt( data_classification_dir_out + "x_feature_train_with_frm_path.csv", combine_X, delimiter=",", fmt='%s')

    for frmRt in frameRates[::-1]: 
        
        frm_config_reso =[]            # each frm's config's resolution
        #print ("frmRt:", frmRt)
        y_num = y_train_gt.shape[0]
        for i in range(0, y_num):
            id_conf = int(y_train_gt[i])
            conf = id_config_dict[id_conf]  # config
            fr = int(conf.split("-")[1])
            if fr == frmRt:
                reso = int(conf.split("-")[0].split("x")[1])
                #frm_Config_reso.append(reso)
                video_frm_index = str(train_video_frm_id_arr[i]).split("/")[-2:]
                video_id = video_frm_index[0].split("_")[0]
                frm_start_id = video_frm_index[1].split(".")[0]
                start_second = (int(frm_start_id)-1)/PLAYOUT_RATE
                acc = y_train_gt_acc_arr[i] 
                spf = y_train_gt_spf_arr[i]
                frm_config_reso.append([video_id+'-'+frm_start_id, start_second, fr, reso,  acc, spf, id_conf])
        
        out_file_name = "dist-frame_rate-" + str(frmRt) + ".pdf"
        lst_frame_rate_config = [frm_config_reso[k][3] for k in range(0, len(frm_config_reso))]
        #print ("lst_frame_rate_config: ", lst_frame_rate_config)
        plot_bar_distribution(lst_frame_rate_config, data_classification_dir_out + out_file_name, 'Reso', 'Frequency', 'Frame rate ' + str(frmRt) + ' resolution  distribution')
        
        frm_config_reso = sorted(frm_config_reso, key=lambda ele: ele[0])
        with open(data_classification_dir_out + 'frmRt-' + str(frmRt) +'-arr_frm_Config_reso.csv', 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["video_frm_index", "start_second", "frm_rate", "reso", "acc", "spf", "id_conf"])    
                for val in frm_config_reso:
                    writer.writerow(val)
        total_frm_config_reso += frm_config_reso 
    
    total_frm_config_reso = sorted(total_frm_config_reso, key=lambda ele: ele[0])
    with open(data_classification_dir_out + 'total_frmRt-' +'-arr_frm_Config_reso.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["video_frm_index", "start_second", "frm_rate", "reso", "acc", "spf", "id_conf"])    
        for val in total_frm_config_reso:
            writer.writerow(val)
    
    
def get_x_y_out_data(data_pose_keypoint_dir, classifier_result_dir):
    # read from file
    
    
    config_id_dict, id_config_dict = read_all_config_name_from_file(data_pose_keypoint_dir, False)

    x_feature_train = np.load(classifier_result_dir + 'all_video_x_feature.pkl', allow_pickle=True)
    
    y_train_gt = np.load(classifier_result_dir + 'all_video_y_gt.pkl', allow_pickle=True)          # frame by frame but it also calculate the 1sec interval with each frame starting
    
    
    data_classification_dir = dataDir3  +'test_classification_result/' + 'analyse_input_data/'
    plot_bar_distribution(y_train_gt, data_classification_dir + "distribution_y_out.pdf", 'Class', 'Frequency', 'Class Frequency')
    
    
    plot_feature_out_relation(x_feature_train, y_train_gt, id_config_dict, data_classification_dir)
    
    save_x_y_out_data(x_feature_train, y_train_gt, id_config_dict, classifier_result_dir, data_classification_dir)
    
    
    

def analyse_input_data():
    '''
    data preliminary analysis
    '''
   
    data_pose_keypoint_dir =  dataDir3 + 'output_001_dance/'

    classifier_result_dir = dataDir3 +  "test_classification_result/min_accuracy-0.95/"
    
    
    get_x_y_out_data(data_pose_keypoint_dir, classifier_result_dir) 
    
    # plot class distribution
    #plot_y_class_distribution(y_train)

    #plot the accuracy for the y_train
    
    


if __name__== "__main__": 
    analyse_input_data()
    
    