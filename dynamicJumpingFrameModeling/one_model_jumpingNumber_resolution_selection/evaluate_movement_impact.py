#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 21:08:48 2021

@author: fubao
"""

import sys
import os
import cv2
import time

import numpy as np
from glob import glob

import matplotlib 
matplotlib.use('Agg') 

from matplotlib import pyplot as plt

import matplotlib.pylab as pylab
params = {'legend.fontsize': 60,
          'figure.figsize': (11.69, 8.27),
         'axes.labelsize':60,
         'axes.titlesize':62,
         'xtick.labelsize':62,
         'ytick.labelsize':62}
pylab.rcParams.update(params)

from get_data_jumpingNumber_resolution import samplingResoDataGenerate
from get_data_jumpingNumber_resolution import video_dir_lst
from get_data_jumpingNumber_resolution import min_acc_threshold
from get_data_jumpingNumber_resolution import max_jump_number

from prediction_jumpingNumber_resolution import *

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from data_file_process import write_pickle_data
from data_file_process import read_pickle_data
from common_video_process import drawHuman

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/../..')

from classifierForSwitchConfig.common_classifier import read_poseEst_conf_frm_more_dim
from classifierForSwitchConfig.common_classifier import readProfilingResultNumpy

from profiling.common_prof import dataDir3
from profiling.common_prof import PLAYOUT_RATE
from profiling.common_prof import NUM_KEYPOINT   
from profiling.common_prof import resoStrLst_OpenPose

from profiling.common_prof import computeOKS_1to1
from profiling.writeIntoPickleConfigFrameAccSPFPoseEst import read_config_name_from_file

dataDir3 = "../" + dataDir3


# evaluate the object movement's impact the accuracy and spf  (with different configurations)


def get_absolute_velocity(current_frm_est, prev_frm_est, count_jumping_frames, arr_ema_absolute_velocity=None):
    # On frame velocity
    
    time_interval = count_jumping_frames                     # time interval unit is in frames, not in second (1.0/PLAYOUT_RATE)*count_jumping_frames
    arr_vec_diff = current_frm_est - prev_frm_est
    
    arr_abs_vec_diff =  np.absolute(arr_vec_diff)           # arr_vec_diff  
    #print("abs_vec_diff: ", arr_abs_vec_diff)
    arr_velocity_vec = arr_abs_vec_diff/time_interval       # current speed
    
    #arr_ema_absolute_speed = arr_speed_vec * ALPHA + (1.0-ALPHA) * arr_ema_absolute_speed
    
    return arr_velocity_vec     # arr_ema_absolute_speed
    


def get_each_interval_average_velocity(interval_len, config_est_frm_arr, acc_frame_arr, spf_frame_arr, video_dir):
    # interval_len is the frame length; calcuate the average velocity first
    
    FRM_NO = config_est_frm_arr.shape[1]
    
    
    lsts_arr_acc_interval = []
    lst_arr_spf_interval = []
    reso_indx_lst = [1, 24]
    
    for reso_indx in reso_indx_lst:
        #reso_indx =18   #  1 vs 14
        #reso = int(resoStrLst_OpenPose[reso_indx].split('x')[0]) * int(resoStrLst_OpenPose[reso_indx].split('x')[1])
        prev_frm_est = config_est_frm_arr
        
        arr_velocity_interval = []
        
        arr_acc_interval = []
        arr_spf_interval = []
        start_frm_indx = 0
        while(start_frm_indx < FRM_NO):
                    
            end_frm = start_frm_indx + interval_len-1
            
            if end_frm >= FRM_NO:
                break
            prev_frm_est = config_est_frm_arr[reso_indx][start_frm_indx]
    
            current_frm_est = config_est_frm_arr[reso_indx][end_frm]
            arr_velocity_vec = get_absolute_velocity(current_frm_est, prev_frm_est, interval_len, arr_ema_absolute_velocity=None)
            
            
            velocity_interval = np.mean(arr_velocity_vec[:, 0:2])  # , axis=0)
           # print("arr_velocity_vec: ", arr_velocity_vec.shape, velocity_interval.shape)
            arr_velocity_interval.append(velocity_interval)
            
            acc_interval = np.mean(acc_frame_arr[reso_indx][start_frm_indx:end_frm])
            arr_acc_interval.append(acc_interval)
            
            spf_interval = np.mean(spf_frame_arr[reso_indx][start_frm_indx:end_frm])
            arr_spf_interval.append(spf_interval)
            
            start_frm_indx = end_frm 
    
        mean_velocity = np.mean(arr_velocity_interval)
    
        lsts_arr_acc_interval.append(arr_acc_interval)
        lst_arr_spf_interval.append(arr_spf_interval)
    #print("arr_velocity_interval: ", arr_velocity_interval, len(arr_velocity_interval), mean_velocity, min(arr_velocity_interval), max(arr_velocity_interval))
    
    
    #print("get_each_interval_average_velocity arr_velocity_interval: ", np.mean(arr_acc_interval), np.mean(arr_spf_interval))
    
    sub_dir = dataDir3 + video_dir + "/test_evaluation_result/"
    
    """
    outputPlotPdf = sub_dir + "arr_velocity_interval_reso_indx_" + str(reso_indx) + ".pdf"
    
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
        
    
    xlabel = "Interval"
    ylabel = "Velocity"
    plot_statistics_one_line(arr_velocity_interval, outputPlotPdf, xlabel, ylabel)
    
    ylabel = "Accuracy"

    outputPlotPdf = sub_dir + "arr_acc_interval_reso_indx_" + str(reso_indx) + ".pdf"
    plot_statistics_one_line(arr_acc_interval, outputPlotPdf, xlabel, ylabel)

    ylabel = "SPF"
    outputPlotPdf = sub_dir + "arr_spf_interval_reso_indx_" + str(reso_indx) + ".pdf"
    plot_statistics_one_line(arr_spf_interval, outputPlotPdf, xlabel, ylabel)
    """

    adapted_arr_acc_interval, adapted_arr_spf_interval = threshold_based_line_configuration_adpatation(mean_velocity, interval_len, config_est_frm_arr, acc_frame_arr, spf_frame_arr, video_dir)


    outputPlotPdf = sub_dir + "arr_acc_interval_combined.pdf"

    changeLengendLst = ["720p-25", "720p-1", "Movement threshold adapted"]
    
    lsts_arr_acc_interval.append(adapted_arr_acc_interval) 
    xlabel = "Interval"
    ylabel = "Accuracy"
    
    plot_statistics_multi_lines(lsts_arr_acc_interval, outputPlotPdf,  changeLengendLst, xlabel, ylabel)
    
    lst_arr_spf_interval.append(adapted_arr_spf_interval)
    outputPlotPdf = sub_dir + "arr_spf_interval_combined.pdf"
    
    lsts_arr_acc_interval.append(adapted_arr_acc_interval) 
    xlabel = "Interval"
    ylabel = "SPF"
    
    plot_statistics_multi_lines(lst_arr_spf_interval, outputPlotPdf,  changeLengendLst, xlabel, ylabel)
    

    return arr_velocity_interval, arr_acc_interval, arr_spf_interval, mean_velocity


def threshold_based_line_configuration_adpatation(mean_velocity, interval_len, config_est_frm_arr, acc_frame_arr, spf_frame_arr, video_dir):
    # threshold of velocity is > mean; then use a configur
    
    FRM_NO = config_est_frm_arr.shape[1]
    start_frm_indx = 0
     
    start_frm_config_indx = 1   # 
    
    adapted_arr_acc_interval = []
    adapted_arr_spf_interval = []
    while(start_frm_indx < FRM_NO):
     
        end_frm = start_frm_indx + interval_len-1
    
        if end_frm >= FRM_NO:
            break
    
        acc_interval = np.mean(acc_frame_arr[start_frm_config_indx][start_frm_indx:end_frm])
        adapted_arr_acc_interval.append(acc_interval)
    
        spf_interval = np.mean(spf_frame_arr[start_frm_config_indx][start_frm_indx:end_frm])
        adapted_arr_spf_interval.append(spf_interval)
        
        prev_frm_est = config_est_frm_arr[start_frm_config_indx][start_frm_indx]
        current_frm_est = config_est_frm_arr[start_frm_config_indx][end_frm]
        arr_velocity_vec = get_absolute_velocity(current_frm_est, prev_frm_est, interval_len, arr_ema_absolute_velocity=None)
        curr_aver_velo = np.mean(arr_velocity_vec)
        
        if curr_aver_velo >= mean_velocity:
            # use higher configuration
            start_frm_config_indx = 1
        else:
            start_frm_config_indx = 24
            
        start_frm_indx = end_frm 


    print("threshold_based_line_configuration_adpatation arr_velocity_interval: ", np.mean(adapted_arr_acc_interval), np.mean(adapted_arr_spf_interval))
    """
    xlabel = "Interval"
    ylabel = "Accuracy"
    sub_dir = dataDir3 + video_dir + "/test_evaluation_result/"
    outputPlotPdf = sub_dir + "arr_acc_interval_configuration_adaptation_threshold.pdf"
    plot_statistics_one_line(adapted_arr_acc_interval, outputPlotPdf, xlabel, ylabel)

    ylabel = "SPF"

    outputPlotPdf = sub_dir + "arr_spf_interval_configuration_adaptation_threshold.pdf"
    plot_statistics_one_line(arr_spf_interval, outputPlotPdf, xlabel, ylabel)
    """
  
    return adapted_arr_acc_interval, adapted_arr_spf_interval



def movement_velocity_to_frameRate_relationship(interval_len, config_est_frm_arr, acc_frame_arr, spf_frame_arr, video_dir, id_config_dict, min_accuracy_threshold):
    # xxxxxxxx
    #  (30, 13728, 17, 3) (30, 13728) (30, 13703) {0: '1120x832-25-cmu', 1: '960x720-25-cmu', 2: '1120x832-15-cmu', 3: '640x480-25-cmu', 4: '960x720-15-cmu', 5: '480x352-25-cmu', 6: '1120x832-10-cmu', 7: '640x480-15-cmu', 8: '960x720-10-cmu', 9: '320x240-25-cmu', 10: '480x352-15-cmu', 11: '640x480-10-cmu', 12: '1120x832-5-cmu', 13: '320x240-15-cmu', 14: '960x720-5-cmu', 15: '480x352-10-cmu', 16: '320x240-10-cmu', 17: '640x480-5-cmu', 18: '480x352-5-cmu', 19: '1120x832-2-cmu', 20: '960x720-2-cmu', 21: '320x240-5-cmu', 22: '640x480-2-cmu', 23: '1120x832-1-cmu', 24: '960x720-1-cmu', 25: '480x352-2-cmu', 26: '320x240-2-cmu', 27: '640x480-1-cmu', 28: '480x352-1-cmu', 29: '320x240-1-cmu'}
    # select two frame rates one fixed resolution  # 1: '960x720-25-cmu',    14: '960x720-5-cmu', 8: '960x720-10-cmu', 20: '960x720-2-cmu',  24: '960x720-1-cmu',
    
    
    FRM_NO = config_est_frm_arr.shape[1]
    
    
    config_indx_lst = [1, 14]  # [1, 14]  #[1, 8, 20]        # frame rate 25, 10, 2
    
    
    prev_frm_est = config_est_frm_arr
    
    arr_velocity_x_interval = []
    arr_velocity_y_interval = []

    arr_acc_interval = []
    arr_spf_interval = []
    start_frm_indx = 0
    
    reso_indx = 1
    arr_frm_rate_used = []
    
    arr_velocity_mean_interval = []
    while(start_frm_indx < FRM_NO):
        
        end_frm = start_frm_indx + interval_len-1
        
        if end_frm >= FRM_NO:
            break
        
        prev_frm_est = config_est_frm_arr[reso_indx][start_frm_indx]
        
        current_frm_est = config_est_frm_arr[reso_indx][end_frm]
        arr_velocity_vec = get_absolute_velocity(current_frm_est, prev_frm_est, interval_len, arr_ema_absolute_velocity=None)
        
        velocity_interval = np.mean(arr_velocity_vec[:, 0:2], axis = 0)           #  axis=0)[0]
        #print("arr_velocity_vec: ", arr_velocity_vec.shape, velocity_interval.shape)
        velocity_interval_x = velocity_interval[0]
        velocity_interval_y = velocity_interval[1]
        
        
        arr_velocity_x_interval.append(velocity_interval_x)
        arr_velocity_y_interval.append(velocity_interval_y)
        mean_velocity = np.linalg.norm(velocity_interval,ord=1) # scale to same range as frame rate for plot only  #  # (np.mean(velocity_interval))   # np.linalg.norm(velocity_interval,ord=1)
        
        arr_velocity_mean_interval.append(mean_velocity)  
        
        reso_indx = config_indx_lst[1]
        acc_interval_lowest = np.mean(acc_frame_arr[reso_indx][start_frm_indx:end_frm])
        
     
        print("velocity_interval_x: ", acc_interval_lowest,  min_accuracy_threshold, )
        
        if acc_interval_lowest >= min_accuracy_threshold:
            frm_rate = 5
            reso_indx = config_indx_lst[1]
            acc_interval = acc_interval_lowest
   
        else:
            frm_rate = 25
            reso_indx = config_indx_lst[0]
            acc_interval = np.mean(acc_frame_arr[reso_indx][start_frm_indx:end_frm])
            
        arr_frm_rate_used.append(frm_rate)
        
        
        # scale to 25 arange
        
        
        """
        reso_indx = config_indx_lst[2]
        acc_interval_lowest = np.mean(acc_frame_arr[reso_indx][start_frm_indx:end_frm])
        
        reso_indx = config_indx_lst[1]
        acc_interval_middle = np.mean(acc_frame_arr[reso_indx][start_frm_indx:end_frm])
        
        
        print("velocity_interval_x: ", acc_interval_lowest, acc_interval_middle,  min_accuracy_threshold)
        
        if acc_interval_lowest >= min_accuracy_threshold:
            frm_rate = 1
            reso_indx = config_indx_lst[2]
            acc_interval = acc_interval_lowest
        elif acc_interval_middle >= min_accuracy_threshold:
            frm_rate = 10
            reso_indx = config_indx_lst[1]
            acc_interval = acc_interval_middle
        else:
            frm_rate = 25
            reso_indx = config_indx_lst[0]
            acc_interval = np.mean(acc_frame_arr[reso_indx][start_frm_indx:end_frm])

        arr_frm_rate_used.append(frm_rate)
        #arr_acc_interval.append(acc_interval)
        #spf_interval = np.mean(spf_frame_arr[reso_indx][start_frm_indx:end_frm])
        #arr_spf_interval.append(spf_interval)
        """
        
        start_frm_indx = end_frm 
    
    
    print("final arr_frm_rate_used:", arr_velocity_x_interval, arr_frm_rate_used)
    
    
    #arr_velocity_mean_interval = (arr_velocity_mean_interval - min(arr_velocity_mean_interval)) * 25.0 /(max(arr_velocity_mean_interval) - min(arr_velocity_mean_interval))
    
    xlabel = "Time interval"
    ylabel = "Frame Rate/Speed"
    sub_dir = dataDir3 + video_dir + "/test_evaluation_result/speed_frameRate/"
    outputPlotPdf = sub_dir + "arr_speed_frameRate.pdf"
    changeLengendLst = ['Frame Rate', 'Speed']
    
    lst_speed_frameRate_interval = [arr_frm_rate_used, arr_velocity_mean_interval]
    plot_statistics_multi_lines(lst_speed_frameRate_interval, outputPlotPdf,  changeLengendLst, xlabel, ylabel)

    xlabel = "Time interval"
    ylabel = "Frame Rate"
    outputPlotPdf = sub_dir + "arr_timeInterval_frameRate.pdf"
    plot_statistics_one_line(arr_frm_rate_used, outputPlotPdf, xlabel, ylabel)

    np.savetxt(sub_dir + 'arr_timeInterval_frameRate.tsv', arr_frm_rate_used, delimiter= '\t')


    xlabel = "Time interval"
    ylabel = "Speed in X axis"
    outputPlotPdf = sub_dir + "arr_timeInterval_speed_x.pdf"
    plot_statistics_one_line(arr_velocity_x_interval, outputPlotPdf, xlabel, ylabel)
    
    xlabel = "Time interval"
    ylabel = "Speed in Y axis"
    outputPlotPdf = sub_dir + "arr_timeInterval_speed_y.pdf"
    plot_statistics_one_line(arr_velocity_y_interval, outputPlotPdf, xlabel, ylabel)
    
    xlabel = "Time interval"
    ylabel = "Speed"
    outputPlotPdf = sub_dir + "arr_timeInterval_speed.pdf"
    plot_statistics_one_line(arr_velocity_mean_interval, outputPlotPdf, xlabel, ylabel)

    np.savetxt(sub_dir + 'arr_velocity_mean_interval.tsv', arr_velocity_mean_interval, delimiter= '\t')


def movement_velocity_to_Resolution_relationship(interval_len, config_est_frm_arr, acc_frame_arr, spf_frame_arr, video_dir, id_config_dict, min_accuracy_threshold):
    # xxxxxxxx
    #  (30, 13728, 17, 3) (30, 13728) (30, 13703) {0: '1120x832-25-cmu', 1: '960x720-25-cmu', 2: '1120x832-15-cmu', 3: '640x480-25-cmu', 4: '960x720-15-cmu', 5: '480x352-25-cmu', 6: '1120x832-10-cmu', 7: '640x480-15-cmu', 8: '960x720-10-cmu', 9: '320x240-25-cmu', 10: '480x352-15-cmu', 11: '640x480-10-cmu', 12: '1120x832-5-cmu', 13: '320x240-15-cmu', 14: '960x720-5-cmu', 15: '480x352-10-cmu', 16: '320x240-10-cmu', 17: '640x480-5-cmu', 18: '480x352-5-cmu', 19: '1120x832-2-cmu', 20: '960x720-2-cmu', 21: '320x240-5-cmu', 22: '640x480-2-cmu', 23: '1120x832-1-cmu', 24: '960x720-1-cmu', 25: '480x352-2-cmu', 26: '320x240-2-cmu', 27: '640x480-1-cmu', 28: '480x352-1-cmu', 29: '320x240-1-cmu'}
    # select two frame rates one fixed resolution  # 1: '960x720-25-cmu',    14: '960x720-5-cmu', 8: '960x720-10-cmu', 20: '960x720-2-cmu',  24: '960x720-1-cmu',
    
    # select two resolutions with same frame rate: 0: '1120x832-25-cmu',  9: '320x240-25-cmu'
    
    FRM_NO = config_est_frm_arr.shape[1]
    
    
    config_indx_lst = [0, 9]  #
    
    
    prev_frm_est = config_est_frm_arr
    
    arr_velocity_x_interval = []
    arr_velocity_y_interval = []

    arr_acc_interval = []
    arr_spf_interval = []
    start_frm_indx = 0
    
    config_indx = 25
    arr_resolution_used = []
    
    arr_velocity_mean_interval = []
    while(start_frm_indx < FRM_NO):
        
        end_frm = start_frm_indx + interval_len-1
        
        if end_frm >= FRM_NO:
            break
        
        prev_frm_est = config_est_frm_arr[config_indx][start_frm_indx]
        
        current_frm_est = config_est_frm_arr[config_indx][end_frm]
        arr_velocity_vec = get_absolute_velocity(current_frm_est, prev_frm_est, interval_len, arr_ema_absolute_velocity=None)
        
        velocity_interval = np.mean(arr_velocity_vec[:, 0:2], axis = 0)           #  axis=0)[0]
        #print("arr_velocity_vec: ", arr_velocity_vec.shape, velocity_interval.shape)
        velocity_interval_x = velocity_interval[0]
        velocity_interval_y = velocity_interval[1]
        
        
        arr_velocity_x_interval.append(velocity_interval_x)
        arr_velocity_y_interval.append(velocity_interval_y)
        mean_velocity = np.linalg.norm(velocity_interval,ord=1) # scale to same range as frame rate for plot only  #  # (np.mean(velocity_interval))   # np.linalg.norm(velocity_interval,ord=1)
        
        arr_velocity_mean_interval.append(mean_velocity)  
        
        config_indx = config_indx_lst[1]
        acc_interval_lowest = np.mean(acc_frame_arr[config_indx][start_frm_indx:end_frm])
        
     
        print("velocity_interval_x: ", acc_interval_lowest,  min_accuracy_threshold, )
        
        if acc_interval_lowest >= min_accuracy_threshold:
            resolution = 320
            config_indx = config_indx_lst[1]
            acc_interval = acc_interval_lowest
   
        else:
            resolution = 1120
            config_indx = config_indx_lst[0]
            acc_interval = np.mean(acc_frame_arr[config_indx][start_frm_indx:end_frm])
            
        arr_resolution_used.append(resolution)
        
        
        # scale to 25 arange
        
        
        """
        reso_indx = config_indx_lst[2]
        acc_interval_lowest = np.mean(acc_frame_arr[reso_indx][start_frm_indx:end_frm])
        
        reso_indx = config_indx_lst[1]
        acc_interval_middle = np.mean(acc_frame_arr[reso_indx][start_frm_indx:end_frm])
        
        
        print("velocity_interval_x: ", acc_interval_lowest, acc_interval_middle,  min_accuracy_threshold)
        
        if acc_interval_lowest >= min_accuracy_threshold:
            frm_rate = 1
            reso_indx = config_indx_lst[2]
            acc_interval = acc_interval_lowest
        elif acc_interval_middle >= min_accuracy_threshold:
            frm_rate = 10
            reso_indx = config_indx_lst[1]
            acc_interval = acc_interval_middle
        else:
            frm_rate = 25
            reso_indx = config_indx_lst[0]
            acc_interval = np.mean(acc_frame_arr[reso_indx][start_frm_indx:end_frm])

        arr_frm_rate_used.append(frm_rate)
        #arr_acc_interval.append(acc_interval)
        #spf_interval = np.mean(spf_frame_arr[reso_indx][start_frm_indx:end_frm])
        #arr_spf_interval.append(spf_interval)
        """
        
        start_frm_indx = end_frm 
    
    
    print("final arr_resolution_used:", arr_velocity_x_interval, arr_resolution_used)
    
    
    #arr_velocity_mean_interval = (arr_velocity_mean_interval - min(arr_velocity_mean_interval)) * 25.0 /(max(arr_velocity_mean_interval) - min(arr_velocity_mean_interval))
    
    xlabel = "Time interval"
    ylabel = "Resolution/Speed"
    sub_dir = dataDir3 + video_dir + "/test_evaluation_result/speed_resolution/"
    outputPlotPdf = sub_dir + "arr_speed_resolution.pdf"
    changeLengendLst = ['Resolution', 'Speed']
    
    lst_speed_reso_interval = [arr_resolution_used[1:], arr_velocity_mean_interval]
    plot_statistics_multi_lines(lst_speed_reso_interval, outputPlotPdf,  changeLengendLst, xlabel, ylabel)

    xlabel = "Time interval"
    ylabel = "Resolution"
    outputPlotPdf = sub_dir + "arr_timeInterval_resolution.pdf"
    plot_statistics_one_line(arr_resolution_used[1:], outputPlotPdf, xlabel, ylabel)

    np.savetxt(sub_dir + 'arr_timeInterval_resolution.tsv', arr_resolution_used, delimiter= '\t')


    xlabel = "Time interval"
    ylabel = "Speed in X axis"
    outputPlotPdf = sub_dir + "arr_timeInterval_speed_x.pdf"
    plot_statistics_one_line(arr_velocity_x_interval[:-1], outputPlotPdf, xlabel, ylabel)
    
    xlabel = "Time interval"
    ylabel = "Speed in Y axis"
    outputPlotPdf = sub_dir + "arr_timeInterval_speed_y.pdf"
    plot_statistics_one_line(arr_velocity_y_interval[:-1], outputPlotPdf, xlabel, ylabel)
    
    xlabel = "Time interval"
    ylabel = "Speed"
    outputPlotPdf = sub_dir + "arr_timeInterval_speed.pdf"
    plot_statistics_one_line(arr_velocity_mean_interval[:-1], outputPlotPdf, xlabel, ylabel)

    np.savetxt(sub_dir + 'arr_velocity_mean_interval.tsv', arr_velocity_mean_interval, delimiter= '\t')
    

def plot_statistics_multi_lines(yLists, outputPlotPdf, changeLengendLst, xlabel, ylabel):


    #plt.title('Moving speed of the cat')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #print ('changeLengendLst: ', len(changeLengendLst), len(yLists))
    averageYLst = []
    
    for i, ylst in enumerate(yLists):
        ylst = ylst[:240]
        xlst = range(len(ylst))
        plt.plot(xlst, ylst, label=changeLengendLst[i])
        averageYLst.append(sum(ylst) / len(ylst) )
    #plt.xticks(xList)
    #xmarks=[i for i in range(1,len(xList)+1, 2)]
    #plt.xticks(xmarks)
    
    #ax = plt.axes()
    #plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')

    plt.title("__".join(str(round(e,3)) for e in averageYLst))
    plt.legend(loc='best')   # (loc='upper right')

    plt.grid(True)
    plt.savefig(outputPlotPdf)
    
    
def plot_statistics_one_line(arr_num, outputPlotPdf, xlabel, ylabel):
    plt.figure()
    
    #plt.plot(arr_velocity_interval)
    plt.plot(arr_num[:240])
    #plt.scatter(xList, yList)
    
    #plt.title('Moving speed of the cat')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.grid(True)
    #plt.show()
    plt.savefig(outputPlotPdf)
    
    
def plotThreeSubplots(x_lst, y_lst_1, y_lst_2, y_lst_3, x_label, y_label_1, y_label_2, y_label_3, title_name_1, title_name_2):
    '''
    plot two suplots 3X1 structure
    '''
    
    fig,axes=plt.subplots(nrows=3, ncols=1)
    axes[0].plot(x_lst, y_lst_1, zorder=1) 
    sc1 = axes[0].plot(x_lst, y_lst_1, color="r", zorder=2)
    
    axes[1].plot(x_lst, y_lst_2, zorder=1) 
    sc2 = axes[1].plot(x_lst,y_lst_2, color="b", zorder=2)
    
    axes[2].plot(x_lst, y_lst_3, zorder=1) 
    sc3 = axes[2].plot(x_lst,y_lst_3, color="g", zorder=2)
    
    axes[0].set(xlabel='', ylabel=y_label_1)
    axes[1].set(xlabel='', ylabel=y_label_2)
    axes[2].set(xlabel=x_label, ylabel=y_label_3)
    
    #axes[0].legend([sc1], ["Admitted"])
    #axes[1].legend([sc2], ["Not-Admitted"])
    #axes[0].set_title(title_name_1, fontsize=18)
    #axes[1].set_title(title_name_2, fontsize=18)
    #plt.show()
    
    #fig.tight_layout()

    return fig

    
def plot_read_file(video_dir):
    # file_input
    
    sub_dir = dataDir3 + video_dir + "/test_evaluation_result/speed_frame_resolution_together/"

    file_name = sub_dir + 'time_speed_frameRate.tsv'

    arr_speed = []
    arr_frame_rate = []
    arr_reso = []
    cnt = 0
    with open(file_name) as fp:
        for line in fp:
            line_lst = line.strip().split('\t')
            if cnt != 0:
                print("line_lst: ", line_lst, line_lst[-2], line_lst[-1])
                arr_speed.append(float(line_lst[5]))
                arr_frame_rate.append(int(line_lst[6]))
                arr_reso.append(int(line_lst[7]))
            cnt += 1
            if cnt >= 240:
                break
            
    print("arr_speed:", arr_speed, arr_frame_rate, len(arr_frame_rate))

    x_lst = range(0, len(arr_frame_rate))
    y_lst_1 = arr_speed
    y_lst_2 = arr_frame_rate
    y_lst_3 = arr_reso
    x_label = 'Time (sec)'
    y_label_1 = 'Speed'
    y_label_2 = 'Frame rate'
    y_label_3 = 'Resolution (width)'
    title_name_1 = ''
    title_name_2 = ''
    fig = plotThreeSubplots(x_lst, y_lst_1, y_lst_2, y_lst_3, x_label, y_label_1, y_label_2, y_label_3, title_name_1, title_name_2)
    outputPlotPdf = sub_dir + "time_speed_frameRate_resolution.eps"

    fig.savefig(outputPlotPdf)

            
    
def execute_movement_evaluation():
    
    #all_arr_estimated_speed_2_jump_number = blist()  # all video
    all_arr_estimated_speed_2_resolution = None
    ResoDataGenerateObj = samplingResoDataGenerate()
    
    interval_len = 25         # each second
    min_accuracy_threshold = 0.91
    for i, video_dir in enumerate(video_dir_lst[3:4]): # [3:4]):    # [2:3]:   #[1:2]:  # [1:2]:  #[0:1]:        #[1:2]:
        data_pose_keypoint_dir = dataDir3 + video_dir
        
        data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
        #data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result_each_frm/'
        intervalFlag = 'frame'
        all_config_flag = True # False
        config_est_frm_arr, acc_frame_arr, spf_frame_arr = ResoDataGenerateObj.get_data_numpy(data_pose_keypoint_dir, data_pickle_dir, intervalFlag, all_config_flag)
        
        config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)

        print ("get_prediction_acc_delay config_est_frm_arr: ", config_est_frm_arr.shape, acc_frame_arr.shape, spf_frame_arr.shape, id_config_dict)
        
        #movement_velocity_to_frameRate_relationship(interval_len, config_est_frm_arr, acc_frame_arr, spf_frame_arr, video_dir, id_config_dict, min_accuracy_threshold)
    
        #movement_velocity_to_Resolution_relationship(interval_len, config_est_frm_arr, acc_frame_arr, spf_frame_arr, video_dir, id_config_dict, min_accuracy_threshold)
    
        
        plot_read_file(video_dir)

if __name__=="__main__":
    execute_movement_evaluation()
    
