#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:02:52 2019

@author: fubao
"""

import csv
import copy
import sys
import os
import numpy as np
import tkinter
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


from collections import OrderedDict
from collections import defaultdict


from common_classifier import get_cmu_model_config_acc_spf
from common_classifier import getAccSpfArrAllVideo

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from profiling.common_prof import dataDir3
from profiling.common_prof import PLAYOUT_RATE


# analyze the equivalence class for the output

'''
  30 OrderedDict([(0, '1120x832-25-cmu'), (1, '960x720-25-cmu'), (2, '1120x832-15-cmu'), (3, '640x480-25-cmu'), 
  (4, '960x720-15-cmu'), (5, '480x352-25-cmu'), (6, '1120x832-10-cmu'), (7, '640x480-15-cmu'), (8, '960x720-10-cmu'), 
  (9, '320x240-25-cmu'), (10, '480x352-15-cmu'), (11, '640x480-10-cmu'), (12, '1120x832-5-cmu'), (13, '320x240-15-cmu'),
  (14, '960x720-5-cmu'), (15, '480x352-10-cmu'), (16, '320x240-10-cmu'), (17, '640x480-5-cmu'), (18, '480x352-5-cmu'),
  (19, '1120x832-2-cmu'), (20, '960x720-2-cmu'), (21, '320x240-5-cmu'), (22, '640x480-2-cmu'), (23, '1120x832-1-cmu'), 
  (24, '960x720-1-cmu'), (25, '480x352-2-cmu'), (26, '320x240-2-cmu'), (27, '640x480-1-cmu'), (28, '480x352-1-cmu'), 
  (29, '320x240-1-cmu')])
  
'''
    
    
def plot_config_acc_spf(data_pickle_dir, data_pose_keypoint_dir):
    
    acc_frame_arr, spf_frame_arr, id_config_dict = get_cmu_model_config_acc_spf(data_pickle_dir, data_pose_keypoint_dir)
    
    print("acc_frame_arr shape shape: ", acc_frame_arr.shape, spf_frame_arr.shape)
    # each 1 sec interval
    
    
    acc_frame_arr = acc_frame_arr[:, 0::PLAYOUT_RATE]
    spf_frame_arr = spf_frame_arr[:, 0::PLAYOUT_RATE]
    
    conf_num =  acc_frame_arr.shape[0]
    total_sec = acc_frame_arr.shape[1]

    print("acc_frame_arr shape: ",  acc_frame_arr[:, 0])
    
    
    id_config_dict = OrderedDict(sorted(id_config_dict.items(), key=lambda t: t[0]))
    print("id_config_dict id_config_dict: ", len(id_config_dict), id_config_dict)
    xLst = []
    yLst = []
    
    plt.figure()
        

    plotFigs = []
    for i in range(20, 30):     # conf_num
        xLst = spf_frame_arr[i, :total_sec]
        yLst = acc_frame_arr[i, :total_sec]
        
        plotFig = plt.scatter(xLst, yLst, s=5)
        
        plotFigs.append(plotFig)
        
    #plt.title('Moving speed of the cat')
    plt.xlabel("SPF")
    plt.ylabel("Accuracy")
    
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.title("Config Profiling of a video")
    plt.grid(True)
       
    plt.legend(plotFigs,
           [id_config_dict[i] for i in range(0, len(id_config_dict))],
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
    #plt.show()
    
    data_equalclass_dir = data_pose_keypoint_dir + "verify_frm_features/"
    
    plt.savefig(data_equalclass_dir + 'profiling_config_03.pdf')




def plot_config_acc_spf_errorbar(data_pickle_dir, data_pose_keypoint_dir, metric, percentage):
    
    acc_frame_arr, spf_frame_arr, id_config_dict = get_cmu_model_config_acc_spf(data_pickle_dir, data_pose_keypoint_dir)
    
    print("acc_frame_arr shape shape: ", acc_frame_arr.shape, spf_frame_arr.shape)
    # each 1 sec interval
    
    
    acc_frame_arr = acc_frame_arr[:, 0::PLAYOUT_RATE]
    spf_frame_arr = spf_frame_arr[:, 0::PLAYOUT_RATE]
    
    conf_num =  acc_frame_arr.shape[0]
    total_sec = acc_frame_arr.shape[1]

    print("acc_frame_arr shape: ",  acc_frame_arr[:, 0])
    
    
    id_config_dict = OrderedDict(sorted(id_config_dict.items(), key=lambda t: t[0]))
    print("id_config_dict id_config_dict: ", len(id_config_dict), id_config_dict)
    xLst = np.zeros(len(id_config_dict))
    yLst = np.zeros(len(id_config_dict))

    
    if metric == 'mean':
        
        xLst = [np.mean(spf_frame_arr[i, :total_sec]) for i in range(0, len(id_config_dict))]
        yLst = [np.mean(acc_frame_arr[i, :total_sec]) for i in range(0, len(id_config_dict))]
    elif metric == 'percentile':
        xLst = [np.percentile(spf_frame_arr[i, :total_sec], percentage) for i in range(0, len(id_config_dict))]
        yLst = [np.percentile(acc_frame_arr[i, :total_sec], percentage) for i in range(0, len(id_config_dict))]
        
        
    fig, ax = plt.subplots()

    yerror = [np.std(acc_frame_arr[i, :total_sec]) for i in range(0, len(id_config_dict))]
    xerror = [np.std(spf_frame_arr[i, :total_sec]) for i in range(0, len(id_config_dict))]
    
    ax.scatter(xLst, yLst, s=10)
    ax.errorbar(xLst, yLst, yerr=yerror, xerr=xerror, ecolor='r', elinewidth = 0.5, linestyle='None')

    for i in range(0, len(id_config_dict)):
        xy = (xLst[i], yLst[i])
        width = xerror[i]*2
        height = yerror[i]*2
        ellipse = Ellipse(xy=xy, width=width, height=height, 
                            edgecolor='g', fc='None', lw=0.5)
        
        ax.add_patch(ellipse)
    
    #plt.title('Moving speed of the cat')
    ax.set_xlabel("Mean of second per frame in an interval")
    ax.set_ylabel("Mean of Accuracy")
    
    
    for i in range(len(id_config_dict)):
        plt.annotate(i, (xLst[i], yLst[i]))       # id_config_dict[i]
    
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    ax.set_title("Config Profiling of a video")
    ax.grid(True)

    '''
    plt.legend(plotFig,
           "",
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
    '''
    fig.show()
    
    data_equalclass_dir = data_pose_keypoint_dir + "verify_frm_features/"
    
    fig.savefig(data_equalclass_dir + 'profiling_config_error_bar_ellipse_' + metric + '-' + str(percentage) +'.pdf')
    
    
    
def acc_pt_distance(acc_p1, acc_pt2):
    # acc_p1:  [acc, config_string]; the same as acc_pt2
    
    return abs(acc_p1[0] - acc_pt2[0])


def group_data_acc(acc_confg_lst, acc_min_bounded, threshold_divided):
    # get the second element e and then select the data around threshold_divided (e- t, e+t)
    # acc_min_bounded consider 0.9 for example
    # threshold_divided is set as different bounded 
    acc_confg_lst = [acc for acc in acc_confg_lst if acc[0] >=acc_min_bounded]
    acc_above_thres_confg_lst = sorted(acc_confg_lst, key=lambda e: e[0])
    
    #print ("acc_confg_lst: ", acc_above_thres_confg_lst)
        
    visited_pt = []  # = [0]*len(acc_above_thres_confg_lst)
    if len(acc_above_thres_confg_lst) == 0:
        return []
    #centerPt = acc_above_thres_confg_lst[1] if len(acc_above_thres_confg_lst) >= 2 else acc_above_thres_confg_lst[0]
    #print ("centerPt: ", centerPt)
        
    
    group_lst  = []
    while(len(acc_above_thres_confg_lst)):
        subgrp_lst = []
        cp_config_lst = copy.deepcopy(acc_above_thres_confg_lst)
        #print ("cp_config_lstcp_config_lst: ", len(cp_config_lst), cp_config_lst, )
        centerPt = cp_config_lst[1] if len(cp_config_lst) >= 2 else cp_config_lst[0]
        
        i = 0
        flag = False
        while( i < len(cp_config_lst)):
            
            # pop an element
            pt = cp_config_lst[i]
            #print ("pt: ", pt, centerPt)
            if acc_pt_distance(pt, centerPt) <= threshold_divided:
                 visited_pt.append(pt)
                 acc_above_thres_confg_lst.remove(pt)
                 subgrp_lst.append(pt)
                 flag = True
            i += 1
        
        if not flag:
            visited_pt.append(pt)
            acc_above_thres_confg_lst.remove(centerPt)
            subgrp_lst.append(pt)
        group_lst.append(subgrp_lst)
    #print ("group_lst: ", len(group_lst), group_lst)
    
    return group_lst


def plot_ellipse_inside_point(x, group_lst):
    '''
    x axis: x 
    given the group_lst of points to make a eclipse around that
    '''
    
    lst_ellipse = []
    for gr in group_lst:
        
        if len(gr) == 1:
            height = 0.02
            xy = (x, gr[0][0])
        else:
            height = gr[-1][0] - gr[0][0]
            xy = (x, gr[0][0] + height/2)
        width = 0.3
        ellipse = Ellipse(xy=xy, width=width, height=height, 
                            edgecolor='g', fc='None', lw=0.5)
        
        lst_ellipse.append(ellipse)
    return lst_ellipse
        
    
def plot_segment_config_acc_spf(data_pickle_dir, data_pose_keypoint_dir):
    
    acc_frame_arr, spf_frame_arr, id_config_dict = get_cmu_model_config_acc_spf(data_pickle_dir, data_pose_keypoint_dir)
    
    #print("acc_frame_arr shape shape: ", acc_frame_arr.shape, spf_frame_arr.shape)
    # each 1 sec interval
    
    
    acc_frame_arr = acc_frame_arr[:, 0::PLAYOUT_RATE]
    spf_frame_arr = spf_frame_arr[:, 0::PLAYOUT_RATE]
    
    conf_num =  acc_frame_arr.shape[0]
    total_sec = acc_frame_arr.shape[1]

    print("acc_frame_arr shape: ", acc_frame_arr[:, 0], conf_num)
    
    
    id_config_dict = OrderedDict(sorted(id_config_dict.items(), key=lambda t: t[0]))
    print("id_config_dict id_config_dict: ", len(id_config_dict), id_config_dict)
    xLst = np.zeros(total_sec)
    yLst = np.zeros(total_sec)
    
    
    
    fig, ax = plt.subplots()
    
    plotFigs = []
    
    start_point_index = 429
    plot_points_num = start_point_index + 100 #total_sec         # 2     # total_sec
    
    for i in range(0, conf_num):
        xLst = np.array(range(start_point_index, plot_points_num))
        yLst = acc_frame_arr[i, start_point_index:plot_points_num]
            
        plotFig = plt.scatter(xLst, yLst, s=1)
            
        plotFigs.append(plotFig)
        
    #plt.title('Moving speed of the cat')
    ax.set_xlabel("Segment (s)")
    ax.set_ylabel("Accuracy")
    
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    ax.set_title("Config Acc of a video")
    ax.grid(True)
      
    #plt.show()
    
   
    dict_group_lsts = OrderedDict()
    acc_min_bounded = 0.9
    threshold_divided = 0.02
    for i in range(start_point_index, plot_points_num):
        acc_confg_lst = []
        
        for j in range(0, conf_num):
            acc_confg_lst.append([acc_frame_arr[j, i], id_config_dict[j]])
                
                
        #print ("acc_confg_lst: ", acc_confg_lst)
        group_lst = group_data_acc(acc_confg_lst, acc_min_bounded, threshold_divided)
    
        dict_group_lsts[i] = group_lst
        x = i
        lst_ellipse = plot_ellipse_inside_point(x, group_lst)
        
        for ellipse in lst_ellipse:
            ax.add_patch(ellipse)

        
    
    data_equalclass_dir = data_pose_keypoint_dir + "verify_frm_features/"
    
    fig.savefig(data_equalclass_dir + 'video005_segment_config_group_threshold-' + str(threshold_divided) + '.pdf')
      
    
    
    data_file_out = data_equalclass_dir + 'video005_segment_config_group_threshold-' + str(threshold_divided) + '.csv'
      
    w = csv.writer(open(data_file_out, "w"))
    for key, val in dict_group_lsts.items():
        val_lst = [key] + val
        w.writerow(val_lst)



def get_acc_helper(row, dict_acc_frm_arr):
    
    #print ("rrrrr row: ", row)
    frm_path = row[0]
    video_id=int(frm_path.split("/")[-2].split("_")[0])
    frm_id = int(frm_path.split("/")[-1].split(".")[0]) - 1
    y_out_config_id = row[1]
    
    acc_frame_arr = dict_acc_frm_arr[video_id]
    
    return acc_frame_arr[y_out_config_id][frm_id]

def get_acc_or_spf_from_frm(test_video_frm_id_arr, dict_acc_or_spf_frm_arr, y_out_arr):
    '''
    get the acc or spf for the y_out_arr 
    y_out_arr is index id output class
    
    dict_acc_or_spf_frm_arr :   dict_acc_frm_arr or dict_spf_frm_arr
    '''
    
    test_video_frm_id_arr = test_video_frm_id_arr.reshape(test_video_frm_id_arr.shape[0], -1)
    y_out_arr = y_out_arr.reshape(y_out_arr.shape[0], -1)
    
    combined_arr = np.hstack((test_video_frm_id_arr, y_out_arr))
    
    y_out_acc_or_spf_arr = np.apply_along_axis(get_acc_helper, 1, combined_arr, dict_acc_or_spf_frm_arr)    

    #print ("y_out_acc_arr :", y_out_acc_or_spf_arr.shape, y_out_acc_or_spf_arr[:2])
    
    return y_out_acc_or_spf_arr



def get_test_y_pred_acc_spf(classifier_result_dir, all_video_data_flag, dict_acc_frm_arr, dict_spf_frm_arr):
    
    if all_video_data_flag:
    
        y_test_gt = np.load(classifier_result_dir + 'y_test.pkl', allow_pickle=True)          # frame by frame but it also calculate the 1sec interval with each frame starting
        
        # load y_predicted
        y_test_pred = np.load(classifier_result_dir + 'y_test_used_pred.pkl', allow_pickle=True)          # frame by frame but it also calculate the 1sec interval with each frame starting
        
        #load test_video_frm_id_arr predicted
        test_video_frm_id_arr = np.load(classifier_result_dir + 'test_video_frm_id_arr.pkl', allow_pickle=True)          # frame by frame but it also calculate the 1sec interval with each frame starting
    
        #print("acc_frame_arr shape shape: ", test_video_frm_id_arr.shape, y_test_gt.shape, y_test_pred.shape)
            
        y_test_gt_acc_arr= get_acc_or_spf_from_frm(test_video_frm_id_arr, dict_acc_frm_arr, y_test_gt)
        
        y_test_pred_acc_arr = get_acc_or_spf_from_frm(test_video_frm_id_arr, dict_acc_frm_arr, y_test_pred)
    
        y_test_gt_spf_arr= get_acc_or_spf_from_frm(test_video_frm_id_arr, dict_spf_frm_arr, y_test_gt) # *PLAYOUT_RATE-1
        
        y_test_pred_spf_arr = get_acc_or_spf_from_frm(test_video_frm_id_arr, dict_spf_frm_arr, y_test_pred)  # *PLAYOUT_RATE-1
        
        
    return y_test_gt_acc_arr, y_test_pred_acc_arr, y_test_gt_spf_arr, y_test_pred_spf_arr
        
    
def plot_acc_trend_pred(data_pickle_dir, data_pose_keypoint_dir, classifier_result_dir, all_video_data_flag, dict_acc_frm_arr, dict_spf_frm_arr):
    '''
    plot the trend of predicted acc
    '''
    
    if all_video_data_flag:
        y_test_gt_acc_arr, y_test_pred_acc_arr, y_test_gt_spf_arr, y_test_pred_spf_arr = get_test_y_pred_acc_spf(classifier_result_dir, all_video_data_flag, dict_acc_frm_arr, dict_spf_frm_arr)
    
        num_data = len(y_test_gt_acc_arr)
        
        fig, ax = plt.subplots()
            
        start_point_index = 0
        plot_points_num = num_data # start_point_index + 100 #total_sec         # 2     # total_sec
                
            
        xLst = np.array(range(start_point_index, plot_points_num))    
        
        ax.plot(xLst, y_test_gt_acc_arr, 'o-', label="Ground_truth")
        ax.plot(xLst, y_test_pred_acc_arr, 'x-', label="Predicted")
        #ax.set_ylim((0, 1))
        #plt.title('Moving speed of the cat')
        ax.set_xlabel("Segment (s)")
        ax.set_ylabel("Accuracy")
        
        ax.legend()
        
        ax.set_title("Config Acc of test dataset (multiple mixed videos)")
        ax.grid(True)
        
        #fig.show()
        
        data_equalclass_dir = data_pose_keypoint_dir + "verify_frm_features/"
        if not os.path.exists(data_equalclass_dir):
            os.mkdir(data_equalclass_dir)

        fig.savefig(data_equalclass_dir + 'multiVideos_predicted_gt_acc' + '.pdf')
          
        diff_pred_gt_arr = y_test_gt_acc_arr - y_test_pred_acc_arr
        #print("diff_pred_gt_arr shape shape: ", diff_pred_gt_arr.shape, diff_pred_gt_arr)
        fig, axs = plt.subplots()
        axs.hist(diff_pred_gt_arr, bins=60)
        fig.savefig(data_equalclass_dir + 'multiVideos_predicted_gt_acc_diff_histogram' + '.pdf')
        
    
        num_data = len(y_test_gt_spf_arr)
        
        fig, ax = plt.subplots()
            
        start_point_index = 0
        plot_points_num = num_data # start_point_index + 100 #total_sec         # 2     # total_sec
                
            
        xLst = np.array(range(start_point_index, plot_points_num))    
        
        ax.plot(xLst, y_test_gt_spf_arr, 'o-', label="Ground_truth")
        ax.plot(xLst, y_test_pred_spf_arr, 'x-', label="Predicted")
        #ax.set_ylim((0, 1))
        #plt.title('Moving speed of the cat')
        ax.set_xlabel("Segment (s)")
        ax.set_ylabel("Delay")
        
        ax.legend()

        
        ax.set_title("Delay of a test dataset (multiple mixed videos)")
        ax.grid(True)
        
        #fig.show()
        
        data_equalclass_dir = data_pose_keypoint_dir + "verify_frm_features/"
        
        fig.savefig(data_equalclass_dir + 'multiVideos_predicted_gt_spf' + '.pdf')
          
        diff_spf_pred_gt_arr = y_test_gt_spf_arr - y_test_pred_spf_arr
        print("diff_pred_gt_arr shape shape:xxxx ", diff_pred_gt_arr.shape, diff_pred_gt_arr)
        
        fig, axs = plt.subplots()
        
        axs.hist(diff_spf_pred_gt_arr, bins=60)
        #axs.plot(xLst, diff_spf_pred_gt_arr, 'o-', label="difference")

        fig.savefig(data_equalclass_dir + 'multiVideos_predicted_gt_spf_diff_histogram' + '.pdf')
        
            
    else:
        
        # load y_test_ground_truth
        
        y_test_gt = np.load(classifier_result_dir + 'y_test.pkl', allow_pickle=True)          # frame by frame but it also calculate the 1sec interval with each frame starting
    
        # load y_predicted
        y_test_pred = np.load(classifier_result_dir + 'y_test_used_pred.pkl', allow_pickle=True)          # frame by frame but it also calculate the 1sec interval with each frame starting
        
        #load test_video_frm_id_arr predicted
        test_video_frm_id_arr = np.load(classifier_result_dir + 'test_video_frm_id_arr.pkl', allow_pickle=True)          # frame by frame but it also calculate the 1sec interval with each frame starting
    
        print("acc_frame_arr shape shape: ", test_video_frm_id_arr.shape, y_test_gt.shape, y_test_pred.shape)
        
        acc_frame_arr, spf_frame_arr, id_config_dict = get_cmu_model_config_acc_spf(data_pickle_dir, data_pose_keypoint_dir)
            
        dict_acc_frm_arr[5] = acc_frame_arr            # video 005
        y_test_gt_acc_arr= get_acc_or_spf_from_frm(test_video_frm_id_arr, dict_acc_frm_arr, y_test_gt)
        
        y_test_pred_acc_arr = get_acc_or_spf_from_frm(test_video_frm_id_arr, dict_acc_frm_arr, y_test_pred)
    
            
        num_data = len(y_test_gt)
        
        fig, ax = plt.subplots()
            
        start_point_index = 0
        plot_points_num = num_data # start_point_index + 100 #total_sec         # 2     # total_sec
                
            
        xLst = np.array(range(start_point_index, plot_points_num))    
        
        ax.plot(xLst, y_test_gt_acc_arr, 'o-')
        ax.plot(xLst, y_test_pred_acc_arr, 'x-')
        #ax.set_ylim((0, 1))
        #plt.title('Moving speed of the cat')
        ax.set_xlabel("Segment (s)")
        ax.set_ylabel("Accuracy")
        
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        ax.set_title("Config Acc of a video")
        ax.grid(True)
        
        #fig.show()
        
        data_equalclass_dir = data_pose_keypoint_dir + "verify_frm_features/"
        
        #fig.savefig(data_equalclass_dir + 'video005_predicted_gt_acc' + '.pdf')
          
        diff_pred_gt_arr = y_test_gt_acc_arr - y_test_pred_acc_arr
        print("diff_pred_gt_arr shape shape: ", diff_pred_gt_arr.shape, diff_pred_gt_arr)
        fig, axs = plt.subplots()
        axs.hist(diff_pred_gt_arr, bins=50)
        fig.savefig(data_equalclass_dir + 'video005_predicted_gt_acc_diff_histogram' + '.pdf')
    


def get_each_video_acc(test_video_frm_id_arr, y_test_gt_acc_arr, y_test_pred_acc_arr):
    
        dict_videos_arr_gt =  defaultdict(list)
        dict_videos_arr_pred = defaultdict(list)
        
        
        for i in range(0, test_video_frm_id_arr.shape[0]):
            video_id = int(test_video_frm_id_arr[i].split("/")[-2].split("_")[0])
            dict_videos_arr_gt[video_id].append(y_test_gt_acc_arr[i])
            dict_videos_arr_pred[video_id].append(y_test_pred_acc_arr[i])
            
        print (" dict_videos_arr_pred: ", dict_videos_arr_pred[1][0])
        
        return dict_videos_arr_gt, dict_videos_arr_pred
        
            
def get_accuracy_loose_threshold(data_pickle_dir, data_pose_keypoint_dir, classifier_result_dir, all_video_data_flag, dict_acc_frm_arr, dict_spf_frm_arr, theta):
    '''
    get the accuracy of loose threshold (precision) [-theta, theta]
    '''    
    if all_video_data_flag:
        
        y_test_gt_acc_arr, y_test_pred_acc_arr, y_test_gt_spf_arr, y_test_pred_spf_arr = get_test_y_pred_acc_spf(classifier_result_dir, all_video_data_flag, dict_acc_frm_arr, dict_spf_frm_arr)

        
        #count how many with [-theta, theta]
        diff_pred_gt_arr = np.abs(y_test_gt_acc_arr - y_test_pred_acc_arr)
        
        num_satisfied = len(diff_pred_gt_arr[diff_pred_gt_arr <= theta])
        prec = num_satisfied/len(diff_pred_gt_arr)
        
        print("diff_pred_gt_arrssss shape shape: ", diff_pred_gt_arr.shape, diff_pred_gt_arr.shape, prec)
        
        test_video_frm_id_arr = np.load(classifier_result_dir + 'test_video_frm_id_arr.pkl', allow_pickle=True)          # frame by frame but it also calculate the 1sec interval with each frame starting
        
        # get the each video's array in the predicted array
        dict_videos_arr_gt, dict_videos_arr_pred = get_each_video_acc(test_video_frm_id_arr, y_test_gt_acc_arr, y_test_pred_acc_arr)
        
        video_num = 18
        for i in range(1, video_num):
            gt_arr = np.asarray(dict_videos_arr_gt[i])
            pred_arr = np.asarray(dict_videos_arr_pred[i])
            
            diff_pred_gt_arr = gt_arr - pred_arr
            if len(gt_arr) > 0 and diff_pred_gt_arr.size == 0:
                prec == 0
            elif diff_pred_gt_arr.size == 0:
                print("video_% d_ not exisiting in the test data set" %(i))
                continue
            else:
                num_satisfied = len(diff_pred_gt_arr[diff_pred_gt_arr <= theta])
                prec = num_satisfied/(diff_pred_gt_arr).shape[0]
        
            print("video_% d_theta_threshold_boundary acc% 5f" %(i, prec))
            


def get_config_acc_spf_classifier_result_dir(classifier_result_dir, dict_acc_frm_arr, dict_spf_frm_arr):
    '''
    get the accuracy after applying video classifier result
    '''
    
    # parse integer video id and frame id
    
    
    y_test_gt_acc_arr, y_test_pred_acc_arr, y_test_gt_spf_arr, y_test_pred_spf_arr = get_test_y_pred_acc_spf(classifier_result_dir, all_video_data_flag, dict_acc_frm_arr, dict_spf_frm_arr)


    print ("y_test_gt_acc_arr: ", y_test_gt_acc_arr.shape, np.mean(y_test_gt_acc_arr))

    print ("y_test_pred_acc_arr: ", y_test_pred_acc_arr.shape, np.mean(y_test_pred_acc_arr))
    

    print ("delay_gt_arr,  ", y_test_gt_spf_arr)
    up_to_segment_delay_gt = y_test_gt_spf_arr[0]
    total_delay_gt = [up_to_segment_delay_gt]
    for i in range(1, y_test_gt_spf_arr.shape[0]):
        #if up_to_segment_delay_gt <= 0:        # catching  up
        #    up_to_segment_delay_gt = 0
            
        up_to_segment_delay_gt += y_test_gt_spf_arr[i]
        total_delay_gt.append(up_to_segment_delay_gt)
        
    
    up_to_segment_delay_pred = y_test_pred_spf_arr[0]
    total_delay_pred = [up_to_segment_delay_pred]
    
    # print ("delay_pred_arr: ", delay_pred_arr.shape)
    for i in range(1, y_test_pred_spf_arr.shape[0]):
        #if up_to_segment_delay_pred <= 0:
        #    up_to_segment_delay_pred = 0
        up_to_segment_delay_pred += y_test_pred_spf_arr[i]
        total_delay_pred.append(up_to_segment_delay_pred)
    
    
    #print ("total_delay_gt,  ", total_delay_gt)
    aver_total_delay_gt = sum(y_test_gt_spf_arr)/len(y_test_gt_spf_arr)   
    if aver_total_delay_gt < 0:
        aver_total_delay_gt = 0
        
    aver_total_delay_pred = sum(y_test_pred_spf_arr)/len(y_test_pred_spf_arr)   
    if aver_total_delay_pred < 0:
        aver_total_delay_pred = 0
    print ("aver_total_delay_gt pred  processing time,  ", aver_total_delay_gt, aver_total_delay_pred)
    #print ("delay_pred_arr  delay,  ", delay_pred_arr)


        
    
    
    
def execute_plot_config(all_video_data_flag):
    '''
    execute plot the config after classification.
    '''
    
    
    
    if all_video_data_flag:
        
        dict_acc_frm_arr, dict_spf_frm_arr = getAccSpfArrAllVideo()
        
        minAccuracy = 0.93        # 0.90 0.93  0.95
        
        classifier_result_dir = dataDir3 + 'test_classification_result/' + 'min_accuracy-' + str(minAccuracy)+ '/'
            
        #data_pickle_dir = dataDir3 + 'output_005_dance/' + 'frames_pickle_result/'

        get_config_acc_spf_classifier_result_dir(classifier_result_dir, dict_acc_frm_arr, dict_spf_frm_arr)
    
    
        data_pose_keypoint_dir = classifier_result_dir 
        plot_acc_trend_pred("", data_pose_keypoint_dir, classifier_result_dir, all_video_data_flag, dict_acc_frm_arr, dict_spf_frm_arr)          
        
        theta = 0.1
        #get_accuracy_loose_threshold(data_pickle_dir, data_pose_keypoint_dir, classifier_result_dir, all_video_data_flag, dict_acc_frm_arr, dict_spf_frm_arr, theta)          
        
        
    else:
        
        video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                        'output_003_dance/', 'output_004_dance/',  \
                        'output_005_dance/', 'output_006_yoga/', \
                        'output_007_yoga/', 'output_008_cardio/', \
                        'output_009_cardio/', 'output_010_cardio/', \
                        'output_011_dance/', 'output_012_dance/']
         

        
        for i, video_dir in enumerate(video_dir_lst[4:5]):
            data_pose_keypoint_dir =  dataDir3 + video_dir
            data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
                       
            #plot_config_acc_spf(data_pickle_dir, data_pose_keypoint_dir)     
            '''
            metric = 'percentile' # 'mean'  # 'percentile'
            percentage = 70   # '' #90
            plot_config_acc_spf_errorbar(data_pickle_dir, data_pose_keypoint_dir, metric, percentage)
            '''
            
            #plot_segment_config_acc_spf(data_pickle_dir, data_pose_keypoint_dir)     
            
            classifier_result_dir = dataDir3 + video_dir + 'classifier_result/'
            
            plot_acc_trend_pred(data_pickle_dir, data_pose_keypoint_dir, classifier_result_dir, all_video_data_flag, "", "")     
        
        
        
if __name__== "__main__": 
    
    #all_video_data_flag = False
    all_video_data_flag = True
    execute_plot_config(all_video_data_flag)