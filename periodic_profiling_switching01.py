#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:39:20 2019

@author: fubao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:45:01 2019

@author: fubao
"""


# use another policy of switching config,
# set a bounded accuracy threshold, and also a delay threshold
# first find out the average delay without profiling time (set as 0), with bounded accuracy, pick the config with the fastest speed
#we then set a delay threshold a little fewer than the average delay

# after that, we use bounded accuracy threshold and bounded delay threshold to find the next config for switching
# also add the current delay as a feature
import os

from glob import glob
import pandas as pd
import numpy as np
import operator

import matplotlib.backends.backend_pdf

from plot import plotFiveSubplots
from plot import plotTwoLineOneplot
from plot import plotLineOneplot
from plot import plotLineOneplotWithSticker
from plot import plotThreeSubplots
from plot import plot_fittingLine
from plot import plotStickerMultipleLines
from plot import plotOnePlotMultiLine
from plot import plotBarTwoDataWithSticker
from plot import plotBarThreeDataWithSticker

from blist import blist
from common import dataDir3
from common import PLAYOUT_RATE

from classifierForSwitchConfig.common_classifier import readProfilingResultNumpy
from classifierForSwitchConfig.common_classifier import read_all_config_name_from_file
from classifierForSwitchConfig.common_classifier import getParetoBoundary
from classifierForSwitchConfig.common_classifier import getParetoBoundary_boundeAcc

'''
online scheduling with periodic profiling result from all configuration or pareto boundary

greedy policy based on maximum accuracy, and then select a cheapest configuration.

'''



VIDEO_LENGTH = 60*10        # 10minutes long
    

def simBoundedAcc(data_pose_keypoint_dir, data_pickle_dir, segmentTime, profile_interval_time, minAccuracy, pareto_bound_flag=False):
    '''
    simulated bounded accuracy
    '''
    
    intervalFlag = 'frame'
    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir, intervalFlag)   # every 1 second to calculate an acc/spf value    

    if pareto_bound_flag == 0:        # all the configurations
        lst_idx_selected_config = list(range(acc_frame_arr.shape[0]))
        lst_idx_selected_config.remove(0) # if 0 in lst_idx_selected_config
    
        acc_frame_arr = acc_frame_arr[lst_idx_selected_config, :]
        spf_frame_arr = spf_frame_arr[lst_idx_selected_config, :]
            
    elif pareto_bound_flag == 1:
        # get a pareto boundary only without minAccuracy bounded
        firstSegment = 0
        acc_arr = acc_frame_arr[:, firstSegment]
        spf_arr = spf_frame_arr[:, firstSegment]
        lst_idx_selected_config = getParetoBoundary(acc_arr, spf_arr)
        
        lst_idx_selected_config.remove(0) # if 0 in lst_idx_selected_config

        print ("lst_idx_selected_config: ", len(lst_idx_selected_config), lst_idx_selected_config)
        
        acc_frame_arr = acc_frame_arr[lst_idx_selected_config, :]
        spf_frame_arr = spf_frame_arr[lst_idx_selected_config, :]
        print ("acc_frame_arr: ",  acc_frame_arr.shape)
    elif pareto_bound_flag == 2:
        # get a pareto boundary only with minAccuracy bounded 
        firstSegment = 0
        acc_arr = acc_frame_arr[:, firstSegment]
        spf_arr = spf_frame_arr[:, firstSegment]
        lst_idx_selected_config = getParetoBoundary_boundeAcc(acc_arr, spf_arr, minAccuracy)
        
        lst_idx_selected_config.remove(0) # if 0 in lst_idx_selected_config

        print ("lst_idx_selected_config: ", len(lst_idx_selected_config), lst_idx_selected_config)
        
        acc_frame_arr = acc_frame_arr[lst_idx_selected_config, :]
        spf_frame_arr = spf_frame_arr[lst_idx_selected_config, :]
        print ("acc_frame_arr: ",  acc_frame_arr.shape)

        
    configNo = acc_frame_arr.shape[0]
    
    total_frm_num = acc_frame_arr.shape[1]       # total frame number
    
    segmentNo = int(acc_frame_arr.shape[1]/(segmentTime * PLAYOUT_RATE)) #  acc_seg_arr.shape[1]
    

    #print ("segmentNo: ", minAccuracy,  configNo, segmentNo, total_frm_num)
    
    config_id_dict, id_config_dict = read_all_config_name_from_file(data_pose_keypoint_dir, False)

    #print ("config_id_dict: ", config_id_dict)

    # get number of segments
    
    #segmentTime = 4     #4s
    segment_total_frames = int(segmentTime * PLAYOUT_RATE)
    profile_interval_frames = int(profile_interval_time * PLAYOUT_RATE)
    
    extra_time_segment = segmentTime - profile_interval_time         # extra time in a segment not profiling 
    extra_frames_segment = int(extra_time_segment * PLAYOUT_RATE)
    
    #print ("segment_total_frames: ", profile_interval_time, segmentTime)
    
    start_proc_frm_id = 0          # start processing frame id
    end_finished_process_frm_id = 0         # finished proccessing  frame id
    end_streaming_frm_id = 0         # delayed frame_id, i.e. streaming frame id
    
    
    predicted_seg_acc_arr = np.zeros(segmentNo)        # the selected segment's accuracies, which is predicted
    actual_seg_acc_arr = np.zeros(segmentNo)            # actual execution's frame's acc based on configs
    
    selected_config_id_arr = np.zeros(segmentNo)
    
    each_frame_acc_arr = np.array([], dtype=np.float64)         # each frame accuracy for the video with selected config processed
    
    accumu_lag_seg_arr = np.zeros(segmentNo)
    
    profiling_time_arr = np.zeros(segmentNo)
    
    selected_config_indx = 0
    seg = 0
    while seg < segmentNo:  # and end_streaming_frm_id < total_frm_num:
        
        # process the profiling period
        
        #update the end_finished_process_frm_id
        end_finished_process_frm_id = start_proc_frm_id + profile_interval_frames
        
        #get profiling used time in this segment
        sub_spf_arr = spf_frame_arr[:, start_proc_frm_id:end_finished_process_frm_id]      # # for all frames in this  profiling period
        seg_spf_arr = np.mean(sub_spf_arr, axis=1)                  # each config is the average spf for this segment
        
        sub_acc_arr = acc_frame_arr[:, start_proc_frm_id:end_finished_process_frm_id]         # for all frames in this profiling period
        seg_acc_arr = np.mean(sub_acc_arr, axis=1)                  # each config is the average acc for this segment

        total_time_profiling = np.sum(sub_spf_arr) #  np.sum(sub_spf_arr[selected_config_indx]) #   not considering profiling time to test now np.sum(seg_spf_arr) only segment time  # np.sum(sub_spf_arr)   # np.sum(seg_spf_arr)
        
        profiling_time_arr[seg] = total_time_profiling
        
        #print ("total_time_profiling: ", seg, profiling_time_arr[seg])
        
        indx_config_above_minAcc = np.where(seg_acc_arr >= minAccuracy)      # the index of the config above the threshold minAccuracy
        #print("indx_config_above_minAcc: ", indx_config_above_minAcc, len(indx_config_above_minAcc[0]))
        
        # in case no profiling config found satisfying the minAcc
        #assert(len(indx_config_above_minAcc[0]) !=0)
        if len(indx_config_above_minAcc[0]) == 0:            
            selected_config_indx = 0    # np.argmax(seg_acc_arr)
            
        else:
            tmp_config_indx = np.argmin(seg_spf_arr[indx_config_above_minAcc])   # selected the minimum spf, i.e. the fastest processing speed
            selected_config_indx = indx_config_above_minAcc[0][tmp_config_indx]      # final selected indx from all config_indx
        
        #print ("selected_config_indx: ", selected_config_indx, seg_spf_arr[selected_config_indx], seg_acc_arr[selected_config_indx], seg_spf_arr, seg_acc_arr)
        
        selected_config_id_arr[seg] = selected_config_indx
        
        # get predicted acc 
        predict_acc = seg_acc_arr[selected_config_indx]
        
        #print ("predict_acc: ", predict_acc)
        predicted_seg_acc_arr[seg] = predict_acc
        
        # updated streaming frame id and new start_proc_frm_id
        end_streaming_frm_id += int(profiling_time_arr[seg]*PLAYOUT_RATE)
        
        if end_streaming_frm_id <= profile_interval_frames:
            end_streaming_frm_id = profile_interval_frames
            
        start_proc_frm_id = end_finished_process_frm_id
        
        end_finished_process_frm_id = start_proc_frm_id + extra_frames_segment
        # then process the rest time period extra_time_segment in this segment according to the selected config
        
        # get this config's extra processing seg's time
        tmp_extra_spf_arr = spf_frame_arr[selected_config_indx, start_proc_frm_id:end_finished_process_frm_id]
        process_total_extra_seg_time = np.sum(tmp_extra_spf_arr)
        
        #get this config's extra procesing seg's acc
        tmp_extra_acc_arr = acc_frame_arr[selected_config_indx, start_proc_frm_id:end_finished_process_frm_id]
        #process_total_extra_seg_acc = np.sum(tmp_extra_acc_arr)  
        
        # new streaming frame id updated
        #print ("process_total_extra_seg_time: ", process_total_extra_seg_time, tmp_extra_spf_arr)
        if process_total_extra_seg_time <= extra_time_segment:              # faster than real time
            end_streaming_frm_id +=  extra_frames_segment
        else:
            end_streaming_frm_id += int(process_total_extra_seg_time*PLAYOUT_RATE)
            
        # faster than real time
        if end_streaming_frm_id <= end_finished_process_frm_id:
            end_streaming_frm_id = end_finished_process_frm_id
        # get accumulated lag
        accumu_lag_seg_arr[seg] = (end_streaming_frm_id - end_finished_process_frm_id)/PLAYOUT_RATE
        #print ("end_finished_process_frm_id: ", end_finished_process_frm_id, end_streaming_frm_id, process_total_extra_seg_time, accumu_lag_seg_arr[seg])
        
        each_frame_acc_seg = np.concatenate((sub_acc_arr[selected_config_indx], tmp_extra_acc_arr), axis=0) # each frame_acc in this seg
        
        # actual acc
        actual_acc_seg = np.mean(each_frame_acc_seg)
        
        actual_seg_acc_arr[seg] = actual_acc_seg
        #print("aaaa: ", actual_seg_acc_arr)
        # get predicted acc
        
        
        each_frame_acc_arr = np.concatenate((each_frame_acc_arr, each_frame_acc_seg), axis=0)
        
        #print ("each_frame_acc_arr: ", each_frame_acc_arr.shape)
        # test only
        
        #print ("actual_acc_seg actual_acc_seg: ", predict_acc, actual_acc_seg)

        #break
        
        start_proc_frm_id = end_finished_process_frm_id       # update start_proc_frm_id

        seg += 1
    
    
    #print ("shapes of selected_config_id_arr: ",  selected_config_id_arr.shape, predicted_seg_acc_arr.shape, actual_seg_acc_arr.shape, each_frame_acc_arr.shape)
    predicted_seg_acc_arr = predicted_seg_acc_arr[:seg]
    
    actual_seg_acc_arr = actual_seg_acc_arr[:seg]
    selected_config_id_arr = selected_config_id_arr[:seg]
    accumu_lag_seg_arr = accumu_lag_seg_arr[:seg]
    profiling_time_arr = profiling_time_arr[:seg]

    #print ("accumu_lag_seg_arr: ", accumu_lag_seg_arr)
    print ("actual_seg_acc_arr: ", actual_seg_acc_arr, np.mean(actual_seg_acc_arr))
    print ("predicted_seg_acc_arr: ", predicted_seg_acc_arr, np.mean(predicted_seg_acc_arr))
    return seg, total_frm_num, predicted_seg_acc_arr, actual_seg_acc_arr, selected_config_id_arr, id_config_dict, each_frame_acc_arr, accumu_lag_seg_arr, profiling_time_arr


def get_similarity_selected_config(prev_lst, cur_lst):
    # compare jaccard similarity
    
    joint = set([v for v in prev_lst if v in set(cur_lst)])
    combined = set(prev_lst + cur_lst)
    sim = len(joint)/len(combined)
    
    return sim

def get_pareto_boundary_change(data_pose_keypoint_dir, data_pickle_dir, segmentTime, profile_interval_time, minAccuracy, pareto_bound_flag):
    #check how the pareto boundary changing overtime if we get pareto from the whole configurations for each segment
    
    
    intervalFlag = 'frame'
    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir, intervalFlag)   # every 1 second to calculate an acc/spf value    

 
        
    configNo = acc_frame_arr.shape[0]
    
    total_frm_num = acc_frame_arr.shape[1]       # total frame number
    
    segmentNo = int(acc_frame_arr.shape[1]/(segmentTime * PLAYOUT_RATE)) #  acc_seg_arr.shape[1]
    

    print ("segmentNo: ", configNo, segmentNo, total_frm_num)
    
    config_id_dict, id_config_dict = read_all_config_name_from_file(data_pose_keypoint_dir, False)

    print ("config_id_dict: ", config_id_dict)

    # get number of segments
    
    #segmentTime = 4     #4s
    segment_total_frames = int(segmentTime * PLAYOUT_RATE)
    profile_interval_frames = int(profile_interval_time * PLAYOUT_RATE)
    
    extra_time_segment = segmentTime - profile_interval_time         # extra time in a segment not profiling 
    extra_frames_segment = int(extra_time_segment * PLAYOUT_RATE)
    
    print ("segment_total_frames: ", segmentTime, segment_total_frames)
    
    start_proc_frm_id = 0          # start processing frame id
    end_finished_process_frm_id = 0         # finished proccessing  frame id
    end_streaming_frm_id = 0         # delayed frame_id, i.e. streaming frame id
    
    
    predicted_seg_acc_arr = np.zeros(segmentNo)        # the selected segment's accuracies, which is predicted
    actual_seg_acc_arr = np.zeros(segmentNo)            # actual execution's frame's acc based on configs
    
    selected_config_id_arr = np.zeros(segmentNo)
    
    each_frame_acc_arr = np.array([], dtype=np.float64)         # each frame accuracy for the video with selected config processed
    
    accumu_lag_seg_arr = np.zeros(segmentNo)
    
    profiling_time_arr = np.zeros(segmentNo)
    
    selected_config_indx = 0
    seg = 0
    prev_lst_idx_selected_config = []
    lst_idx_selected_config = prev_lst_idx_selected_config
    pareto_similarity_lst = []
    while seg < segmentNo:  # and end_streaming_frm_id < total_frm_num:
                   
        if pareto_bound_flag == 1:
            current_segment_start_indx = start_proc_frm_id
            acc_arr = acc_frame_arr[:, current_segment_start_indx]
            spf_arr = spf_frame_arr[:, current_segment_start_indx]
            lst_idx_selected_config = getParetoBoundary(acc_arr, spf_arr)
            
            # get the similarity with previous iteration
            if seg >= 1:
                sim = get_similarity_selected_config(prev_lst_idx_selected_config, lst_idx_selected_config)
                
                #print ("sim: ", sim)
                pareto_similarity_lst.append(sim)
            if seg == 0:    # only compare with the first segment
                prev_lst_idx_selected_config = lst_idx_selected_config      
        
        
        #update the end_finished_process_frm_id
        end_finished_process_frm_id = start_proc_frm_id + profile_interval_frames
        
        #get profiling used time in this segment
        sub_spf_arr = spf_frame_arr[:, start_proc_frm_id:end_finished_process_frm_id]      # # for all frames in this  profiling period
        seg_spf_arr = np.mean(sub_spf_arr, axis=1)                  # each config is the average spf for this segment
        
        sub_acc_arr = acc_frame_arr[:, start_proc_frm_id:end_finished_process_frm_id]         # for all frames in this profiling period
        seg_acc_arr = np.mean(sub_acc_arr, axis=1)                  # each config is the average acc for this segment

        total_time_profiling = np.sum(sub_spf_arr) #  np.sum(sub_spf_arr[selected_config_indx]) #   not considering profiling time to test now np.sum(seg_spf_arr) only segment time  # np.sum(sub_spf_arr)   # np.sum(seg_spf_arr)
        
        profiling_time_arr[seg] = total_time_profiling//2
        
        #print ("total_time_profiling: ", seg, total_time_profiling)
        
        indx_config_above_minAcc = np.where(seg_acc_arr >= minAccuracy)      # the index of the config above the threshold minAccuracy
        #print("indx_config_above_minAcc: ", indx_config_above_minAcc, len(indx_config_above_minAcc[0]))
        
        # in case no profiling config found satisfying the minAcc
        if len(indx_config_above_minAcc[0]) == 0:            
            selected_config_indx = np.argmax(seg_acc_arr)
            
        else:
            tmp_config_indx = np.argmin(seg_spf_arr[indx_config_above_minAcc])   # selected the minimum spf, i.e. the fastest processing speed
            selected_config_indx = indx_config_above_minAcc[0][tmp_config_indx]      # final selected indx from all config_indx
        
        #print ("selected_config_indx: ", selected_config_indx, seg_spf_arr[selected_config_indx], seg_acc_arr[selected_config_indx], seg_spf_arr, seg_acc_arr)
        
        selected_config_id_arr[seg] = selected_config_indx
        
        # get predicted acc 
        predict_acc = seg_acc_arr[selected_config_indx]
        
        #print ("predict_acc: ", predict_acc)
        predicted_seg_acc_arr[seg] = predict_acc
        
        # updated streaming frame id and new start_proc_frm_id
        end_streaming_frm_id += int(profiling_time_arr[seg]*PLAYOUT_RATE)
        
        if end_streaming_frm_id <= profile_interval_frames:
            end_streaming_frm_id = profile_interval_frames
            
        start_proc_frm_id = end_finished_process_frm_id
        
        end_finished_process_frm_id = start_proc_frm_id + extra_frames_segment
        # then process the rest time period extra_time_segment in this segment according to the selected config
        
        # get this config's extra processing seg's time
        tmp_extra_spf_arr = spf_frame_arr[selected_config_indx, start_proc_frm_id:end_finished_process_frm_id]
        process_total_extra_seg_time = np.sum(tmp_extra_spf_arr)
        
        #get this config's extra procesing seg's acc
        tmp_extra_acc_arr = acc_frame_arr[selected_config_indx, start_proc_frm_id:end_finished_process_frm_id]
        #process_total_extra_seg_acc = np.sum(tmp_extra_acc_arr)  
        
        # new streaming frame id updated
        #print ("process_total_extra_seg_time: ", process_total_extra_seg_time, tmp_extra_spf_arr)
        if process_total_extra_seg_time <= extra_time_segment:              # faster than real time
            end_streaming_frm_id +=  extra_frames_segment
        else:
            end_streaming_frm_id += int(process_total_extra_seg_time*PLAYOUT_RATE)
            
        # faster than real time
        if end_streaming_frm_id <= end_finished_process_frm_id:
            end_streaming_frm_id = end_finished_process_frm_id
        # get accumulated lag
        accumu_lag_seg_arr[seg] = (end_streaming_frm_id - end_finished_process_frm_id)/PLAYOUT_RATE
        #print ("end_finished_process_frm_id: ", end_finished_process_frm_id, end_streaming_frm_id, process_total_extra_seg_time, accumu_lag_seg_arr[seg])
        
        each_frame_acc_seg = np.concatenate((sub_acc_arr[selected_config_indx], tmp_extra_acc_arr), axis=0) # each frame_acc in this seg
        
        # actual acc
        actual_acc_seg = np.mean(each_frame_acc_seg)
        
        actual_seg_acc_arr[seg] = actual_acc_seg
        #print("aaaa: ", actual_seg_acc_arr)
        # get predicted acc
        
        
        each_frame_acc_arr = np.concatenate((each_frame_acc_arr, each_frame_acc_seg), axis=0)
        
        #print ("each_frame_acc_arr: ", each_frame_acc_arr.shape)
        # test only
        
        #print ("actual_acc_seg actual_acc_seg: ", predict_acc, actual_acc_seg)

        #break
        
        start_proc_frm_id = end_finished_process_frm_id       # update start_proc_frm_id

        seg += 1
                

    return seg, total_frm_num, pareto_similarity_lst


def plot_profiling_switching_result(segmentNo, actual_seg_acc_arr, accumu_lag_seg_arr, profiling_time_arr, output_plot_pdf_path):
    # plot different profiling switch accuracy, delay, profiling time
    
    x_lst = range(0, segmentNo)

    y_lst1 = actual_seg_acc_arr
    y_lst2 = accumu_lag_seg_arr
    y_lst3 = profiling_time_arr
    
    xlabel = "Segment No"
    ylabel1 = 'Actual accuracy'
    ylabel2 = 'Accumulated delay'
    ylabel3 = 'Profiling_time (s)'

    title_name_2 = ""  # "profiling"
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_plot_pdf_path)
    fig = plotThreeSubplots(x_lst, y_lst1, y_lst2, y_lst3, xlabel, ylabel1, ylabel2, ylabel3, title_name_2, "")
    pdf.savefig(fig)
    pdf.close()


def plot_profiling_actual_accuracy_overtime(segmentNo, actual_seg_acc_arr, minAccuracy, output_plot_pdf_path):
    # plot different profiling switch accuracy, delay, profiling time
    
    x_lst = range(0, segmentNo)

    y_lst1 = actual_seg_acc_arr
    
    percent_above_threshold = round(((actual_seg_acc_arr >= minAccuracy).sum())/segmentNo, 3)
    
    xlabel = "Segment No"
    ylabel1 = 'Actual accuracy'


    title_name_1 = ""  # "Actual accuracy overtime (percentage above threshold-" + str(percent_above_threshold) + ')'
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_plot_pdf_path)
    fig = plotLineOneplot(x_lst, y_lst1, xlabel, ylabel1, title_name_1)
    pdf.savefig(fig)
    pdf.close()


    
def plot_profiling_interval_segment_time_change(x_lst_stickers, lst_average_acc, lst_accumulated_lag, output_plot_pdf_path1, output_plot_pdf_path2):
    
    
    #xlst = range(0, len(lst_average_acc))

    y_lst1 = lst_average_acc
    
    xlabel = "Segment time-Profiling interval ratio"
    ylabel = 'Average accuracy'

    title_name = "" # "Accuracy vs different profiling_interval and segment time"
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_plot_pdf_path1)
    #fig = plotTwoLineOneplot(xlst, y_lst1, y_lst2, xlabel, ylabel, title_name)
    fig = plotLineOneplotWithSticker(x_lst_stickers, y_lst1, xlabel, ylabel, title_name)
    pdf.savefig(fig)
    
    pdf.close()


    pdf = matplotlib.backends.backend_pdf.PdfPages(output_plot_pdf_path2)
    
    y_lst2 = lst_accumulated_lag
    ylabel = 'Accumulated delay (s)'

    fig = plotLineOneplotWithSticker(x_lst_stickers, y_lst2, xlabel, ylabel, title_name)
    pdf.savefig(fig)
    pdf.close()
    



def plot_profiling_pred_actual_acc(segmentNo, predicted_seg_acc_arr, actual_seg_acc_arr, output_plot_pdf_path):
    # plot the actual accuracy vs prediction accuracy
    
    x_lst = predicted_seg_acc_arr
    y_lst = actual_seg_acc_arr
    
    #print ("x_lst: ", x_lst, y_lst)
    xlabel = "Predicted Accuracy"
    ylabel = 'Actual Accuracy'

    title_name = ""  # "Predicted vs Actual Accuracy"
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_plot_pdf_path)
    #fig = plotTwoLineOneplot(xlst, y_lst1, y_lst2, xlabel, ylabel, title_name)
    fig = plot_fittingLine(x_lst, y_lst, xlabel, ylabel, title_name)
    pdf.savefig(fig)
    
    pdf.close()

    
def plot_profiling_inference_profiling_time(segmentTime, x_lst_stickers, lst_total_processing_time, lst_total_profiling_time, output_plot_pdf_path):
     
    
    legend_labels = ['Inference time', 'Profiling time', 'Latency_time']
    y_lst1 = lst_total_processing_time
    y_lst2 = lst_total_profiling_time
    
    
    
    y_lst3 = [y_lst1[i] + y_lst2[i] - VIDEO_LENGTH for i in range(0, len(y_lst1))]
    if segmentTime == 'all':
        xlabel = "Profiling interval ratio"
    else:
        xlabel = "Profiling interval ratio"

    ylabel = 'Time (s)'
    
    
    title_name = ""  #  "Inference time /profiling for a 600-second video"
    # plotBarTwoDataWithSticker(x_lst_stickers, y_lst1, y_lst2, xlabel, ylabel, legend_labels, title_name, output_plot_pdf_path)

    plotBarThreeDataWithSticker(x_lst_stickers, y_lst1, y_lst2, y_lst3, xlabel, ylabel, legend_labels, title_name, output_plot_pdf_path)

    print ('Inference time, Profiling time', y_lst1, y_lst2, y_lst3)
    '''   
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_plot_pdf_path)
    #fig = plotTwoLineOneplot(xlst, y_lst1, y_lst2, xlabel, ylabel, title_name)
    legend_labels = ['Inference_time', 'Profiling time']
    fig = plotBarWithSticker(x_lst_stickers, y_lst1, y_lst2, legend_labels, xlabel, ylabel, title_name)
    pdf.savefig(fig)
    
    pdf.close()
    '''
    

def plot_different_seg_ratio_acc(lst_x_lst_stickers_seg_ratio, lst_different_seg_time_ratio_acc, legend_labels, minAccuracy, output_plot_pdf_path):
    
    
    y_lsts = lst_different_seg_time_ratio_acc

    xlabel = "Profiling interval ratio"
    
    ylabel = 'Accuracy'

    title_name = "" # "Average accuracy of different segment time on a 600-second video"
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_plot_pdf_path)
    #fig = plotTwoLineOneplot(xlst, y_lst1, y_lst2, xlabel, ylabel, title_name)
    fig = plotStickerMultipleLines(lst_x_lst_stickers_seg_ratio, y_lsts, legend_labels, minAccuracy, xlabel, ylabel, title_name)
    pdf.savefig(fig)
    
    pdf.close()

def plot_different_seg_ratio_lag(lst_x_lst_stickers_seg_ratio, lst_different_seg_time_ratio_lag, legend_labels, minAccuracy, output_plot_pdf_path):
    
    
    y_lsts = lst_different_seg_time_ratio_lag

    xlabel = "Profiling interval ratio"
    
    ylabel = 'Delay (s)'

    title_name = ""  # "Final lag of different segment time on a 600-second video"
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_plot_pdf_path)
    #fig = plotTwoLineOneplot(xlst, y_lst1, y_lst2, xlabel, ylabel, title_name)
    fig = plotStickerMultipleLines(lst_x_lst_stickers_seg_ratio, y_lsts, legend_labels, minAccuracy,  xlabel, ylabel, title_name)
    pdf.savefig(fig)
    
    pdf.close() 
  
    
def plot_profiling_pareto_changing(segmentNo, lsts_pareto_similarity, legend_labels, output_plot_pdf_path):
    # plot pareto boundary changing
    
    xlabel = "Segment No."
    
    ylabel = 'Similarity'

    title_name = "" # "Similarity of selected config segment time on a 600-second video"
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_plot_pdf_path)
    #fig = plotTwoLineOneplot(xlst, y_lst1, y_lst2, xlabel, ylabel, title_name)
    fig = plotOnePlotMultiLine(lsts_pareto_similarity, legend_labels, xlabel, ylabel, title_name)
    pdf.savefig(fig)
    
    pdf.close() 
    
    
def executeDifferentBoundedAccPTR(data_pose_keypoint_dir, data_pickle_dir, pareto_bound_flag):
    '''
    get accuracy vs different threshold with different profiling interval ratio values
    '''
    
    outDir = data_pose_keypoint_dir + 'profiling_result/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
                
    min_acc_lst = [0.90, 0.92, 0.94, 0.96, 0.98]
            

    for minAccuracy in min_acc_lst[1:2]:  
                
        lst_segment_time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 12, 15, 20, 25]
        
        lst_profile_interval_time_ratio = [1/1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/10, 1/15, 1/20, 1/25]
        
        lst_average_acc = []
        lst_accumulated_lag = []
        x_lst_stickers = []
        
        lst_total_processing_time = []           # for one video total processing time in one ratio
        lst_total_profiling_time = []           # for one video total profiling in one ratio
        if pareto_bound_flag == 0:
            outDir2 = outDir + "simulate_result_video05_boundedAccuracy_profilingTime_segmentTime_minAcc" + str(minAccuracy) + "/"
        elif pareto_bound_flag == 1:
            outDir2 = outDir + "simulate_result_video05_boundedAccuracy_paretoBound_profilingTime_segmentTime_minAcc" + str(minAccuracy) + "/"
        elif pareto_bound_flag == 2:
            outDir2 = outDir + "simulate_result_video05_boundedAccuracy_paretoBound_and_boundedAccuracy_profilingTime_segmentTime_minAcc" + str(minAccuracy) + "/"

        if not os.path.exists(outDir2):
            os.mkdir(outDir2)
          
        
        lst_different_seg_time_ratio_acc = []
        lst_different_seg_time_ratio_lag = []
        lst_x_lst_stickers_seg_ratio = []
        
        lsts_pareto_similarity = []           # changing of pareto similarity
        
        legend_labels = []
        for segmentTime in lst_segment_time[5:6]:  # lst_segment_time:    # lst_segment_time[2:5]
            
            lst_each_seg_time_acc = []          #
            lst_each_seg_time_lag = []
            x_lst_stickers_seg_ratio = []

            legend_labels.append('seg time: ' + str(segmentTime) + 's')
            for ratio in lst_profile_interval_time_ratio:        # lst_profile_interval_time_ratio[1:5]
                
                if segmentTime * ratio < 1:
                    continue
                
                x_lst_stickers.append(str(round(segmentTime, 3)) + '-' + str(round(ratio, 3)))
                
                x_lst_stickers_seg_ratio.append(str(round(ratio, 3)))
                profile_ratio = ratio 
                
                profile_interval_time  = profile_ratio*segmentTime
               
            
                #outputPlotPdf = outDir + "simulate_result_06_boundedAccuracy" + "_result_01.pdf"    
                #pdf = matplotlib.backends.backend_pdf.PdfPages(outputPlotPdf)
        
                
                segmentNo, frameNo, predicted_seg_acc_arr, actual_seg_acc_arr, selected_config_id_arr, id_config_dict, each_frame_acc_arr, accumu_lag_seg_arr, profiling_time_arr = simBoundedAcc(data_pose_keypoint_dir, data_pickle_dir, segmentTime, profile_interval_time, minAccuracy, pareto_bound_flag)
                total_proc_time = segmentNo*(segmentTime*(1-ratio))
                
                lst_total_processing_time.append(total_proc_time)
                lst_total_profiling_time.append(np.sum(profiling_time_arr))
                #aver_acc = np.mean(actual_seg_acc_arr)
                #lst_average_acc.append(aver_acc)
                
                #accumu_lag = accumu_lag_seg_arr[-1]
                #lst_accumulated_lag.append(accumu_lag)
            

                if pareto_bound_flag == 0:
                    output_plot_pdf_path = outDir2 + 'profilingIntervalRatio-0' + str(round(profile_ratio, 3)).split('.')[-1]  + '_segmentTime-' + str(segmentTime) + '.pdf'
                    output_plot_pdf_proc_time_one_segmentTime_path = outDir2 + 'different_ratio_inference_time_one_segmentTime_minAcc_0' + str(minAccuracy).split('.')[-1] + '.pdf'

                elif pareto_bound_flag == 1:
                    output_plot_pdf_path = outDir2 + 'pareto_profilingIntervalRatio-0' + str(round(profile_ratio, 3)).split('.')[-1]  + '_segmentTime-' + str(segmentTime) + '.pdf'
                    output_plot_pdf_proc_time_one_segmentTime_path = outDir2 + 'pareto_different_ratio_inference_time_one_segmentTime_minAcc_0' + str(minAccuracy).split('.')[-1] + '.pdf'

                elif pareto_bound_flag == 2:
                    output_plot_pdf_path = outDir2 + 'pareto_boundedAcc_profilingIntervalRatio-0' + str(round(profile_ratio, 3)).split('.')[-1]  + '_segmentTime-' + str(segmentTime) + '.pdf'
                    output_plot_pdf_proc_time_one_segmentTime_path = outDir2 + 'pareto_boundedAcc_different_ratio_inference_time_one_segmentTime_minAcc_0' + str(minAccuracy).split('.')[-1] + '.pdf'

                #plot_profiling_switching_result(segmentNo, actual_seg_acc_arr, accumu_lag_seg_arr, profiling_time_arr, output_plot_pdf_path)   
                #plot_profiling_pred_actual_acc(segmentNo, predicted_seg_acc_arr, actual_seg_acc_arr, output_plot_pdf_path)
                

                lst_each_seg_time_acc.append(np.mean(actual_seg_acc_arr))
                
                lst_each_seg_time_lag.append(accumu_lag_seg_arr[-1])
                # write into file
                #df_config_selected = pd.concat(config_selected_lst, axis=1)
                #out_seg_config_selected_file = outDir + "min_acc-" + str(minAccuracy) + "_selected_config"
                #df_config_selected.to_csv(out_seg_config_selected_file, sep='\t', index=False)
                
                


            print ("length: lst_each_seg_time_acc: ", len(lst_each_seg_time_acc))
            lst_different_seg_time_ratio_acc.append(lst_each_seg_time_acc) 
            lst_different_seg_time_ratio_lag.append(lst_each_seg_time_lag) 
            lst_x_lst_stickers_seg_ratio.append(x_lst_stickers_seg_ratio)
                
            if segmentTime == 6:
                    plot_profiling_inference_profiling_time(segmentTime, x_lst_stickers_seg_ratio, lst_total_processing_time, 
                                                        lst_total_profiling_time, output_plot_pdf_proc_time_one_segmentTime_path)
    
                # get how pareto boundary changed with previous pareto boundary
                #segmentNo, frameNo, pareto_similarity_lst = get_pareto_boundary_change(data_pose_keypoint_dir, data_pickle_dir, segmentTime, profile_interval_time, minAccuracy, pareto_bound_flag)
                #lsts_pareto_similarity.append(pareto_similarity_lst)

        if pareto_bound_flag == 0:
            #output_plot_pdf_path1 = outDir2 + 'overall_acc_profilingInterval-segment-minAcc-' + '.pdf'
            #output_plot_pdf_path2 = outDir2 + 'overall_spf_profilingInterval-segment' + '.pdf'
            output_plot_pdf_proc_time_path = outDir2 + 'different_ratio_inference_time_minAcc_0' + str(minAccuracy).split('.')[-1] + '.pdf'
            output_plot_pdf_seg_ratio_acc_path = outDir2 + 'different_seg_ratio_aver_acc_minAcc_0' + str(minAccuracy).split('.')[-1] + '.pdf'
            output_plot_pdf_seg_ratio_lag_path = outDir2 + 'different_seg_ratio_lag_minAcc_0' + str(minAccuracy).split('.')[-1] + '.pdf'

        elif pareto_bound_flag == 1:
            #output_plot_pdf_path1 = outDir2 + 'overall_acc_pareto_profilingInterval-segment' + '.pdf'
            #output_plot_pdf_path2 = outDir2 + 'overall_spf_pareto_profilingInterval-segment' + '.pdf'
            
            output_plot_pdf_proc_time_path = outDir2 + 'pareto_different_ratio_inference_time_minAcc_0' + str(minAccuracy).split('.')[-1] +'.pdf'
            output_plot_pdf_seg_ratio_acc_path = outDir2 + 'pareto_different_seg_ratio_aver_acc_minAcc_0' + str(minAccuracy).split('.')[-1] + '.pdf'
            output_plot_pdf_seg_ratio_lag_path = outDir2 + 'pareto_different_seg_ratio_lag_minAcc_0' + str(minAccuracy).split('.')[-1] + '.pdf'
            
            output_plot_pdf_pareto_change_overtime_path = outDir2 + 'pareto_changing_overtime_profilingIntervalRatio_0' + str(minAccuracy).split('.')[-1] +  '.pdf'


        elif pareto_bound_flag == 2:
            output_plot_pdf_proc_time_path = outDir2 + 'pareto_boundedAcc_different_ratio_inference_time_minAcc_0' + str(minAccuracy).split('.')[-1] +'.pdf'
            output_plot_pdf_seg_ratio_acc_path = outDir2 + 'pareto_boundedAcc_different_seg_ratio_aver_acc_minAcc_0' + str(minAccuracy).split('.')[-1] + '.pdf'
            output_plot_pdf_seg_ratio_lag_path = outDir2 + 'pareto_boundedAcc_different_seg_ratio_lag_minAcc_0' + str(minAccuracy).split('.')[-1] + '.pdf'
  
        #plot_profiling_interval_segment_time_change(x_lst_stickers, lst_average_acc, lst_accumulated_lag, output_plot_pdf_path1, output_plot_pdf_path2)
        #plot_profiling_inference_profiling_time("all", x_lst_stickers, lst_total_processing_time, lst_total_profiling_time, output_plot_pdf_proc_time_path)
    
        # draw each different profiling interval ratio's avearge accuracy
        print ("lst_different_seg_time_ratio_acc: ", lst_different_seg_time_ratio_acc)
        print ("lst_different_seg_time_ratio_acc shape: ", len(lst_different_seg_time_ratio_acc), len(legend_labels), )
        #plot_different_seg_ratio_acc(lst_x_lst_stickers_seg_ratio, lst_different_seg_time_ratio_acc, legend_labels, minAccuracy, output_plot_pdf_seg_ratio_acc_path)
        #plot_different_seg_ratio_lag(lst_x_lst_stickers_seg_ratio, lst_different_seg_time_ratio_lag, legend_labels, minAccuracy, output_plot_pdf_seg_ratio_lag_path)


        #plot_profiling_pareto_changing(segmentNo, lsts_pareto_similarity, x_lst_stickers, output_plot_pdf_pareto_change_overtime_path)


def plot_profiling_overtime_delay(lsts_overtime_lag, legend_labels, output_plot_pdf_path):
    # plot the overtime delay of a video under a profiling interval ratio and segment time and minAcc
    xlabel = "Segment No."
    
    ylabel = 'Delay (s)'

    title_name = "Overtime delay of different PIR  on a 600-second video"
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_plot_pdf_path)
    #fig = plotTwoLineOneplot(xlst, y_lst1, y_lst2, xlabel, ylabel, title_name)
    fig = plotOnePlotMultiLine(lsts_overtime_lag, legend_labels, xlabel, ylabel, title_name)
    pdf.savefig(fig)
    
    pdf.close() 
    
    
    
def execute_different_bound_acc_overTime(data_pose_keypoint_dir, data_pickle_dir, pareto_bound_flag):
    '''
    after figuring out the PT,
    
    get accuracy overtime etc.
    and get the delay over time
    '''
    
    outDir = data_pose_keypoint_dir + 'profiling_result/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
                
    min_acc_lst = [0.90, 0.92, 0.94, 0.96, 0.98]
            

    for minAccuracy in min_acc_lst[1:2]:  
                
        lst_segment_time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 12, 15, 20, 25]
        
        lst_profile_interval_time_ratio = [1/2, 1/3, 1/4, 1/5, 1/6]
        
        lst_accumulated_lag = []       # final accumualted delay of a video
        
        lsts_overtime_lag = []        # overtme delay
        x_lst_stickers = []
        lst_x_lst_stickers_seg_ratio = []
                
        if pareto_bound_flag == 0:
            outDir2 = outDir + "simulate_result_video05_boundedAccuracy_profilingTime_segmentTime_minAcc" + str(minAccuracy) + "/"
        elif pareto_bound_flag == 1:
            outDir2 = outDir + "simulate_result_video05_boundedAccuracy_paretoBound_profilingTime_segmentTime_minAcc" + str(minAccuracy) + "/"
        elif pareto_bound_flag == 2:
            outDir2 = outDir + "simulate_result_video05_boundedAccuracy_paretoBound_and_boundedAccuracy_profilingTime_segmentTime_minAcc" + str(minAccuracy) + "/"

        if not os.path.exists(outDir2):
            os.mkdir(outDir2)
          
        legend_labels = []
        for segmentTime in lst_segment_time[3:4]:          # lst_segment_time
            
            x_lst_stickers_seg_ratio = []

            legend_labels.append('seg time: ' + str(segmentTime) + 's')
            for ratio in lst_profile_interval_time_ratio[2:3]:
                
                if segmentTime * ratio < 1:
                    continue
                x_lst_stickers.append(str(round(segmentTime, 3)) + '-' + str(round(ratio, 3)))
                
                x_lst_stickers_seg_ratio.append(str(round(ratio, 3)))
                profile_ratio = ratio 
                
                profile_interval_time  = profile_ratio*segmentTime
               
            
                #outputPlotPdf = outDir + "simulate_result_06_boundedAccuracy" + "_result_01.pdf"    
                #pdf = matplotlib.backends.backend_pdf.PdfPages(outputPlotPdf)
                
                segmentNo, frameNo, predicted_seg_acc_arr, actual_seg_acc_arr, selected_config_id_arr, id_config_dict, each_frame_acc_arr, accumu_lag_seg_arr, profiling_time_arr = simBoundedAcc(data_pose_keypoint_dir, data_pickle_dir, segmentTime, profile_interval_time, minAccuracy, pareto_bound_flag)
                
                # overtime delay
                #accumu_lag = accumu_lag_seg_arr[-1]
                #lst_accumulated_lag.append(accumu_lag)
                
                lsts_overtime_lag.append(accumu_lag_seg_arr)
                
                if pareto_bound_flag == 0:
                    output_plot_pdf_path = outDir2 + 'profiling_accuracy_overTime_intervalRatio_0' + str(round(profile_ratio, 3)).split('.')[-1] +  '_segmentTime-' + str(segmentTime) + 'minAcc-0' + str(minAccuracy).split('.')[-1] + '.pdf'
                elif pareto_bound_flag == 1:
                    output_plot_pdf_path = outDir2 + 'pareto_profiling_accuracy_overTime_intervalRatio_0' + str(round(profile_ratio, 3)).split('.')[-1] +  '_segmentTime-' + str(segmentTime) + 'minAcc-0' + str(minAccuracy).split('.')[-1] + '.pdf'
                elif pareto_bound_flag == 2:
                    output_plot_pdf_path = outDir2 + 'pareto_boundeAcc_profiling_accuracy_overTime_intervalRatio_0' + str(round(profile_ratio, 3)).split('.')[-1] +  '_segmentTime-' + str(segmentTime) + 'minAcc-0' + str(minAccuracy).split('.')[-1] + '.pdf'


                plot_profiling_actual_accuracy_overtime(segmentNo, actual_seg_acc_arr, minAccuracy, output_plot_pdf_path)

            lst_x_lst_stickers_seg_ratio.append(x_lst_stickers_seg_ratio)

                # write into file
                #df_config_selected = pd.concat(config_selected_lst, axis=1)
                #out_seg_config_selected_file = outDir + "min_acc-" + str(minAccuracy) + "_selected_config"
                #df_config_selected.to_csv(out_seg_config_selected_file, sep='\t', index=False)
          
        if pareto_bound_flag == 2:
            output_plot_overtime_lag_path = outDir2 + 'pareto_boundedAcc_profiling_lag_overTime_minAcc_0' + str(minAccuracy).split('.')[-1] + '.pdf'

            plot_profiling_overtime_delay(lsts_overtime_lag, x_lst_stickers, output_plot_overtime_lag_path)

        #plot_different_seg_ratio_lag(lst_x_lst_stickers_seg_ratio, lst_different_seg_time_ratio_lag, legend_labels, minAccuracy, output_plot_pdf_seg_ratio_lag_path)

    
def execute_multi_video_sim_bounded_acc():
    '''
    execute multiple video query simulation about lat threshold 
    '''
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                        'output_003_dance/', 'output_004_dance/',  \
                        'output_005_dance/', 'output_006_yoga/', \
                        'output_007_yoga/', 'output_008_cardio/', \
                        'output_009_cardio/', 'output_010_cardio/', \
                        'output_011_dance/', 'output_012_dance/']
    
    for video_dir in video_dir_lst[4:5]:
        data_pose_keypoint_dir =  dataDir3 + video_dir
        
        data_pickle_dir = dataDir3 + video_dir + 'frames_pickle_result/'
            
        # pareto_bound_flag = 0 full configuration
        # pareto_bound_flag = 1b pareto boundary
        # pareto_bound_flag = 2  pareto boundary above the bounded accuracy


        pareto_bound_flag = 2  # 0 1 2
        executeDifferentBoundedAccPTR(data_pose_keypoint_dir, data_pickle_dir, pareto_bound_flag)
        
        execute_different_bound_acc_overTime(data_pose_keypoint_dir, data_pickle_dir, pareto_bound_flag)
        
        
if __name__== "__main__": 
        
    
    execute_multi_video_sim_bounded_acc()