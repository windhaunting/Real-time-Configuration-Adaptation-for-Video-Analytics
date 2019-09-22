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
from plot import plotTwoSubPlotOneFig
from plot import plotThreeSubplots
from plot import plotFourSubplots

from blist import blist
from common import dataDir2

from profiling.common_prof import PLAYOUT_RATE
from profiling.writeIntoPickle import read_config_name_from_file

'''
rewrite all the test codes with pickle and numpy
'''


file_lst= ['acc_frame.pkl', 'spf_frame.pkl', 'acc_seg.pkl', 'acc_seg.pkl']

def readPickle():
    x = 1
    

def readProfilingResultNumpy(data_pickle_dir):
    '''
    read profiling from pickle
    the pickle file is created from the file "writeIntoPickle.py"
    '''
    
    acc_frame_arr = np.load(data_pickle_dir + file_lst[0])
    spf_frame_arr = np.load(data_pickle_dir + file_lst[1])
    #acc_seg_arr = np.load(data_pickle_dir + file_lst[2])
    #spf_seg_arr = np.load(data_pickle_dir + file_lst[3])
    
    print ("acc_frame_arr ", type(acc_frame_arr), acc_frame_arr)
    
    return acc_frame_arr, spf_frame_arr
    
    

def simBoundedAcc(data_frame_dir, data_pickle_dir, segmentTime, profile_interval_Time, minAccuracy):
    '''
    simulated bounded accuracy
    '''
    
    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)
    
    configNo = acc_frame_arr.shape[0]
    segmentNo = int(acc_frame_arr.shape[1]/(segmentTime * PLAYOUT_RATE)) #  acc_seg_arr.shape[1]
    

    print ("segmentNo: ", configNo, segmentNo)
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_frame_dir)

    print ("config_id_dict: ", config_id_dict)

    # get nmber of segments
    
    #segmentTime = 4     #4s
    segment_total_frames = int(segmentTime * PLAYOUT_RATE)
    #profile_interval_Time =  1          # segmentTime//4
    profile_interval_frames = int(profile_interval_Time * PLAYOUT_RATE)
    
    extra_time_segment = segmentTime - profile_interval_Time         # extra time in a segment not profiling 
    extra_frames_segment = int(extra_time_segment * PLAYOUT_RATE)
    
    start_proc_frm_id = 0          # start processing frame id
    end_finished_frm_id = 0         # finished proccessing  frame id
    end_streaming_frm_id = 0         # delayed frame_id, i.e. streaming frame id
    
    
    predicted_seg_acc_arr = np.zeros(segmentNo)        # the selected segment's accuracies, which is predicted
    actual_seg_acc_arr = np.zeros(segmentNo)            # actual execution's frame's acc based on configs
    
    selected_config_id_arr = np.zeros(segmentNo)
    
    each_frame_acc_arr = np.array([], dtype=np.float64)         # each frame accuracy for the video with selected config processed
    
    accumu_lag_seg_arr = np.zeros(segmentNo)
    
    profiling_time_arr = np.zeros(segmentNo)
    
    selected_config_indx = 0
    for seg in range(0, segmentNo):        # test 100 first
        
        # process the profiling period
        
        #update the end_finished_frm_id
        end_finished_frm_id = start_proc_frm_id + profile_interval_frames
        
        #get profiling used time in this segment
        
        sub_spf_arr = spf_frame_arr[:, start_proc_frm_id:end_finished_frm_id]      # # for all frames in this  profiling period
        seg_spf_arr = np.mean(sub_spf_arr, axis=1)                  # each config is the average spf for this segment
        
        sub_acc_arr = acc_frame_arr[:, start_proc_frm_id:end_finished_frm_id]         # for all frames in this profiling period
        seg_acc_arr = np.mean(sub_acc_arr, axis=1)                  # each config is the average acc for this segment

        total_time_profiling = np.sum(sub_spf_arr[selected_config_indx]) #   not considering profiling time to test now np.sum(seg_spf_arr) only segment time  # np.sum(sub_spf_arr)   # np.sum(seg_spf_arr)
        
 
        profiling_time_arr[seg] = total_time_profiling
        
        #print ("total_time_profiling: ", seg_spf_arr, seg_acc_arr, total_time_profiling)
        
        indx_config_above_minAcc = np.where(seg_acc_arr >= minAccuracy)      # the index of the config above the threshold minAccuracy
        #print("indx_config_above_minAcc: ", indx_config_above_minAcc, len(indx_config_above_minAcc[0]))
        
        cpy_minAccuracy = minAccuracy
        # in case no profiling config found satisfying the minAcc
        while len(indx_config_above_minAcc[0]) == 0:
            cpy_minAccuracy = cpy_minAccuracy - 0.05 
            indx_config_above_minAcc = np.where(seg_acc_arr >= cpy_minAccuracy)      # the index of the config above the threshold minAccuracy
            
        tmp_config_indx = np.argmax(seg_spf_arr[indx_config_above_minAcc])   # selected the minimum spf, i.e. the fastest processing speed
        
        selected_config_indx = indx_config_above_minAcc[0][tmp_config_indx]      # final selected indx from all config_indx
        #print ("selected_config_indx: ", selected_config_indx, seg_spf_arr[indx_config_above_minAcc])
        
        selected_config_id_arr[seg] = selected_config_indx
        
        
        # get predicted acc 
        predict_acc = seg_acc_arr[selected_config_indx]
        
        #print ("predict_acc: ", predict_acc)
        predicted_seg_acc_arr[seg] = predict_acc
        
        # updated streaming frame id and new start_proc_frm_id
        end_streaming_frm_id += int(total_time_profiling*PLAYOUT_RATE)
        
        if end_streaming_frm_id <= profile_interval_frames:
            end_streaming_frm_id = profile_interval_frames
            
        start_proc_frm_id = end_finished_frm_id
        
        end_finished_frm_id = start_proc_frm_id + extra_frames_segment
        # then process the rest time period extra_time_segment in this segment according to the selected config
        
        # get this config's extra processing seg's time
        tmp_extra_spf_arr = spf_frame_arr[selected_config_indx, start_proc_frm_id:end_finished_frm_id]
        process_total_extra_seg_time = np.sum(tmp_extra_spf_arr)
        
        #get this config's extra procesing seg's acc
        tmp_extra_acc_arr = acc_frame_arr[selected_config_indx, start_proc_frm_id:end_finished_frm_id]
        #process_total_extra_seg_acc = np.sum(tmp_extra_acc_arr)  
        
        # new streaming frame id updated
        #print ("process_total_extra_seg_time: ", process_total_extra_seg_time, tmp_extra_spf_arr)
        if process_total_extra_seg_time <= extra_time_segment:
            
            end_streaming_frm_id += int(extra_time_segment*PLAYOUT_RATE)
        else:
            end_streaming_frm_id += int(process_total_extra_seg_time*PLAYOUT_RATE)
            
        # get accumulated lag
        accumu_lag_seg_arr[seg] = (end_streaming_frm_id - end_finished_frm_id)/PLAYOUT_RATE
        
        print ("frame_id: ", end_finished_frm_id, end_streaming_frm_id, process_total_extra_seg_time, accumu_lag_seg_arr[seg])
        
        each_frame_acc_seg = np.concatenate((sub_acc_arr[selected_config_indx], tmp_extra_acc_arr), axis=0) # each frame_acc in this seg
        
        # actual acc
        actual_acc_seg = np.mean(each_frame_acc_seg)
        
        actual_seg_acc_arr[seg] = actual_acc_seg
        #print("aaaa: ", actual_seg_acc_arr)
        # get predicted acc
        
        
        each_frame_acc_arr = np.concatenate((each_frame_acc_arr, each_frame_acc_seg), axis=0)
        
        #print ("each_frame_acc_arr: ", each_frame_acc_arr.shape)
        # test only
        
        #print ("frame_id: ", start__proc_frm_id, end_finished_frm_id, end_streaming_frm_id)

        #break
        
    
    print ("shapes of selected_config_id_arr: ",  selected_config_id_arr.shape, predicted_seg_acc_arr.shape, actual_seg_acc_arr.shape, each_frame_acc_arr.shape)

    frameNo = len(each_frame_acc_arr)
    
    return segmentNo, frameNo, predicted_seg_acc_arr, actual_seg_acc_arr, selected_config_id_arr, id_config_dict, each_frame_acc_arr, accumu_lag_seg_arr, profiling_time_arr


def plotSimulateResult(segmentNo, frameNo,  predicted_seg_acc_arr, actual_seg_acc_arr, selected_config_id_arr, id_config_dict, each_frame_acc_arr, accumu_lag_seg_arr, profiling_time_arr
, minAccuracy, outputPlotPdf):
    '''
    plot the result of the simulation of different lags of each segment with bounded accuracy
    '''
    
    #config_id_lst = sorted(config_id_dict.items(), key=operator.itemgetter(1))
    #print ("config_id_dict config_id_dict config_id_dict: ",  config_id_dict)
    x_lst = range(0, segmentNo)
    #x_lst3 = [str(e[0].split('-')[0].split('x')[1]) + '-' + str(e[0].split('-')[1]) for e in config_id_lst]
    x_lst4 = range(0, frameNo)
    
    xlsts = [x_lst, x_lst, x_lst, x_lst4, x_lst, x_lst]
    
    y_lst11 = predicted_seg_acc_arr
    y_lst12 = actual_seg_acc_arr
    y_lst2 = selected_config_id_arr
    y_lst31 = [id_config_dict[e].split('-')[0].split('x')[1] for e in selected_config_id_arr] # [e[1] for e in config_id_lst]
    y_lst32 = [id_config_dict[e].split('-')[1] for e in selected_config_id_arr] # [e[1] for e in config_id_lst]

    y_lst4 = each_frame_acc_arr
    y_lst5 = accumu_lag_seg_arr
    y_lst6 = profiling_time_arr
    
    ylsts = [[y_lst11, y_lst12], y_lst2, [y_lst31, y_lst32], y_lst4, y_lst5, y_lst6]
        

    x_label = 'Segment no.'
    x_label4 = 'Frame no.'
    
    xlabels = [x_label, x_label, x_label, x_label4, x_label, x_label]
    
    y_label1 = 'Accuracy'
    y_label2 = 'Config_id'
    y_label31 = 'Resolution'
    y_label32 = 'Frame rate'
    y_label4 = 'Accuracy'
    y_label5 = 'Lag time (s)'
    y_label6 = 'Profiling_time (s)'
    
    ylabels = [y_label1, y_label2, [y_label31, y_label32], y_label4, y_label5, y_label6]
    
    title_name_1 = 'min_acc_threshold:' + str(minAccuracy) + \
    '--Average_pred_acc: ' + str(round(np.sum(predicted_seg_acc_arr)/len(predicted_seg_acc_arr),3)) + \
    '--Average_actual_acc: ' + str(round(np.sum(actual_seg_acc_arr)/len(actual_seg_acc_arr),3))
    title_name_2 = 'Selected configs '
    title_name_3 = 'Each Segment vs resolution and frame rate '
    title_name_4 = 'Each frame accuracy with selected config'
    title_name_5 = "Accumulated Lags: " + str(round(np.sum(accumu_lag_seg_arr)/len(accumu_lag_seg_arr),3))
    title_name_6 = 'Profiling time'
    
    title_names = [title_name_1, title_name_2, title_name_3, title_name_4, title_name_5, title_name_6]
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outputPlotPdf)
    
    for i in range(0, 6):
        
        xlst = xlsts[i]
        ylst = ylsts[i]
        xlabel = xlabels[i]
        ylabel = ylabels[i]
        title_name = title_names[i]
        if i == 0:
            fig = plotTwoLineOneplot(xlst, ylst[0], ylst[1], xlabel, ylabel, title_name)
        elif i == 2:
            ylabel1 = ylabel[0]
            ylabel2 = ylabel[1]
            y_lst1 = ylst[0]
            y_lst2 = ylst[1]
            fig = plotTwoSubPlotOneFig(x_lst, y_lst1, y_lst2, xlabel, ylabel1, ylabel2, title_name)
        else:
            #print ("i i i : ", i)
            fig = plotLineOneplot(xlst, ylst, xlabel, ylabel, title_name)
        pdf.savefig(fig)
            
    #fig = plotFiveSubplots(x_lst, y_lst11, y_lst12, x_lst, y_lst2, x_lst3, y_lst3, x_lst, y_lst4, x_lst, y_lst5,
    # x_label, x_label, x_label3, x_label, x_label,  y_label1, y_label2, y_label3, y_label4, y_label5, title_name_1, title_name_4)


    pdf.close()



def plotSimulateResult2(segmentNo, frameNo,  predicted_seg_acc_arr, actual_seg_acc_arr, selected_config_id_arr, id_config_dict, each_frame_acc_arr, accumu_lag_seg_arr, profiling_time_arr
, minAccuracy, outputPlotPdf):
    '''
    plot the result of the simulation of different lags of each segment with bounded accuracy
    '''
    
    #config_id_lst = sorted(config_id_dict.items(), key=operator.itemgetter(1))
    #print ("config_id_dict config_id_dict config_id_dict: ",  config_id_dict)
    x_lst = range(0, segmentNo)
    #x_lst3 = [str(e[0].split('-')[0].split('x')[1]) + '-' + str(e[0].split('-')[1]) for e in config_id_lst]
    x_lst3 = range(0, frameNo)
    
    xlsts = [x_lst, x_lst, x_lst3, x_lst, x_lst]
    
    y_lst11 = predicted_seg_acc_arr
    y_lst12 = actual_seg_acc_arr
    y_lst21 = selected_config_id_arr
    y_lst22 = [id_config_dict[e].split('-')[0].split('x')[1] for e in selected_config_id_arr] # [e[1] for e in config_id_lst]
    y_lst23 = [id_config_dict[e].split('-')[1] for e in selected_config_id_arr] # [e[1] for e in config_id_lst]

    y_lst3 = each_frame_acc_arr
    y_lst4 = accumu_lag_seg_arr
    y_lst5 = profiling_time_arr
    
    ylsts = [[y_lst11, y_lst12], [y_lst21, y_lst22, y_lst23], y_lst3, y_lst4, y_lst5]
        

    x_label = 'Segment no.'
    x_label3 = 'Frame no.'
    
    xlabels = [x_label, x_label, x_label3, x_label, x_label]
    
    y_label1 = 'Accuracy'
    y_label21 = 'Config_id'
    y_label22 = 'Resolution'
    y_label23 = 'Frame rate'
    y_label3 = 'Accuracy'
    y_label4 = 'Lag time (s)'
    y_label5 = 'Profiling_time (s)'
    
    ylabels = [y_label1, [y_label21, y_label22, y_label23], y_label3, y_label4, y_label5]
    
    title_name_1 = 'min_acc_thres:' + str(minAccuracy) + \
    '--Average_pred_acc: ' + str(round(np.sum(predicted_seg_acc_arr)/len(predicted_seg_acc_arr),3)) + \
    '--Average_actual_acc: ' + str(round(np.sum(actual_seg_acc_arr)/len(actual_seg_acc_arr),3))
    #title_name_2 = 'Selected configs '
    title_name_2 = 'Each Segment vs configs id,  resolution and frame rate '
    title_name_3 = 'Each frame accuracy  with selected config'
    title_name_4 = "Accumulated Lags: " + str(round(np.sum(accumu_lag_seg_arr)/len(accumu_lag_seg_arr),3))
    title_name_5 = 'Profiling time'
    
    title_names = [title_name_1, title_name_2, title_name_3, title_name_4, title_name_5]
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outputPlotPdf)
    
    for i in range(0, 5):
        
        xlst = xlsts[i]
        ylst = ylsts[i]
        xlabel = xlabels[i]
        ylabel = ylabels[i]
        title_name = title_names[i]
        if i == 0:
            fig = plotTwoLineOneplot(xlst, ylst[0], ylst[1], xlabel, ylabel, title_name)
        elif i == 1:
            ylabel1 = ylabel[0]
            ylabel2 = ylabel[1]
            ylabel3 = ylabel[2]
            y_lst1 = ylst[0]
            y_lst2 = ylst[1]
            y_lst3 = ylst[2]
            
            fig = plotThreeSubplots(x_lst, y_lst1, y_lst2, y_lst3, xlabel, ylabel1, ylabel2, ylabel3, title_name_2, "")
   
        else:
            #print ("i i i : ", i)
            fig = plotLineOneplot(xlst, ylst, xlabel, ylabel, title_name)
        pdf.savefig(fig)
            
    #fig = plotFiveSubplots(x_lst, y_lst11, y_lst12, x_lst, y_lst2, x_lst3, y_lst3, x_lst, y_lst4, x_lst, y_lst5,
    # x_label, x_label, x_label3, x_label, x_label,  y_label1, y_label2, y_label3, y_label4, y_label5, title_name_1, title_name_4)


    pdf.close()
    

    
    
def plotSimulateResult3(segmentNo, frameNo,  predicted_seg_acc_arr, actual_seg_acc_arr, selected_config_id_arr, id_config_dict, each_frame_acc_arr, accumu_lag_seg_arr, profiling_time_arr
, minAccuracy, outputPlotPdf):
    '''
    plot the result of the simulation of different lags of each segment with bounded accuracy
    '''
    
    mode_id_dict = {'cmu':0, 'mobilenet':1, 'a':2}

    #config_id_lst = sorted(config_id_dict.items(), key=operator.itemgetter(1))
    #print ("config_id_dict config_id_dict config_id_dict: ",  config_id_dict)
    x_lst = range(0, segmentNo)
    #x_lst3 = [str(e[0].split('-')[0].split('x')[1]) + '-' + str(e[0].split('-')[1]) for e in config_id_lst]
    x_lst3 = range(0, frameNo)
    
    xlsts = [x_lst, x_lst, x_lst3, x_lst, x_lst]
    
    y_lst11 = predicted_seg_acc_arr
    y_lst12 = actual_seg_acc_arr
    y_lst21 = selected_config_id_arr
    y_lst22 = [id_config_dict[e].split('-')[0].split('x')[1] for e in selected_config_id_arr] # [e[1] for e in config_id_lst]
    y_lst23 = [int(id_config_dict[e].split('-')[1]) for e in selected_config_id_arr] # [e[1] for e in config_id_lst]
    y_lst24 = [mode_id_dict[id_config_dict[e].split('-')[2]] for e in selected_config_id_arr] # [e[1] for e in config_id_lst]
    #print ("y_lst24 y_lst24 y_lst24: ",  y_lst24)
    y_lst3 = each_frame_acc_arr
    y_lst4 = accumu_lag_seg_arr
    y_lst5 = profiling_time_arr
    
    ylsts = [[y_lst11, y_lst12], [y_lst21, y_lst22, y_lst23, y_lst24], y_lst3, y_lst4, y_lst5]
        

    x_label = 'Segment no.'
    x_label3 = 'Frame no.'
    
    xlabels = [x_label, x_label, x_label3, x_label, x_label]
    
    y_label1 = 'Accuracy'
    y_label21 = 'Config_id'
    y_label22 = 'Resolution'
    y_label23 = 'Frame rate'
    y_label24 = 'Model'

    y_label3 = 'Accuracy'
    y_label4 = 'Lag time (s)'
    y_label5 = 'Profiling_time (s)'
    
    ylabels = [y_label1, [y_label21, y_label22, y_label23, y_label24], y_label3, y_label4, y_label5]
    
    title_name_1 = 'min_acc_thres:' + str(minAccuracy) + \
    '--Average_pred_acc: ' + str(round(np.sum(predicted_seg_acc_arr)/len(predicted_seg_acc_arr),3)) + \
    '--Average_actual_acc: ' + str(round(np.sum(actual_seg_acc_arr)/len(actual_seg_acc_arr),3))
    #title_name_2 = 'Selected configs '
    title_name_2 = 'Each Segment vs configs id,  resolution and frame rate '
    title_name_3 = 'Each frame accuracy  with selected config'
    title_name_4 = "Average accumulated Lags: " + str(round(np.sum(accumu_lag_seg_arr)/len(accumu_lag_seg_arr),3))
    title_name_5 = 'Profiling time'
    
    title_names = [title_name_1, title_name_2, title_name_3, title_name_4, title_name_5]
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outputPlotPdf)
    
    for i in range(0, 5):
        
        xlst = xlsts[i]
        ylst = ylsts[i]
        xlabel = xlabels[i]
        ylabel = ylabels[i]
        title_name = title_names[i]
        if i == 0:
            fig = plotTwoLineOneplot(xlst, ylst[0], ylst[1], xlabel, ylabel, title_name)
        elif i == 1:
            ylabel1 = ylabel[0]
            ylabel2 = ylabel[1]
            ylabel3 = ylabel[2]
            ylabel4 = ylabel[3]
            y_lst1 = ylst[0]
            y_lst2 = ylst[1]
            y_lst3 = ylst[2]
            y_lst4 = ylst[3]
            fig = plotFourSubplots(x_lst, y_lst1, y_lst2, y_lst3, y_lst4, xlabel, ylabel1, ylabel2, ylabel3, ylabel4, title_name_2, "")
   
        else:
            #print ("i i i : ", i)
            fig = plotLineOneplot(xlst, ylst, xlabel, ylabel, title_name)
        pdf.savefig(fig)
            
    #fig = plotFiveSubplots(x_lst, y_lst11, y_lst12, x_lst, y_lst2, x_lst3, y_lst3, x_lst, y_lst4, x_lst, y_lst5,
    # x_label, x_label, x_label3, x_label, x_label,  y_label1, y_label2, y_label3, y_label4, y_label5, title_name_1, title_name_4)


    pdf.close()
    
def executeDifferentBoundedAcc(data_frame_dir, data_pickle_dir):
    '''
    get accuracy vs different threshold
    '''
    
    segmentTime = 1.0/PLAYOUT_RATE       # 1 4
    profile_interval_Time = segmentTime   #  segmentTime/4
    
    outDir = data_pickle_dir + 'simulate_result_06_boundedAccuracy_no_profilingTIme_segmentTime_' + str(segmentTime) + 's/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    
    #outputPlotPdf = outDir + "simulate_result_06_boundedAccuracy" + "_result_01.pdf"    
    #pdf = matplotlib.backends.backend_pdf.PdfPages(outputPlotPdf)

    min_acc_lst = [0.8, 0.85, 0.9, 0.95, 1.0]
    
    #xList = min_acc_lst
    #yList = []
    for minAccuracy in min_acc_lst[1:2]:         # 
        
        segmentNo, frameNo, predicted_seg_acc_arr, actual_seg_acc_arr, selected_config_id_arr, id_config_dict, each_frame_acc_arr, accumu_lag_seg_arr, profiling_time_arr = simBoundedAcc(data_frame_dir, data_pickle_dir, segmentTime, profile_interval_Time, minAccuracy)
        outDir2 = data_pickle_dir + 'simulate_result_06_boundedAccuracy_no_profilingTIme_segmentTime_' + str(segmentTime) + 's/' + 'minAcc_' + str(minAccuracy) + '/'
        if not os.path.exists(outDir2):
            os.mkdir(outDir2)
        outputPlotPdf = outDir2 + "boundedAccuracy_" + str(minAccuracy)  + ".pdf"    
        #plotSimulateResult(segmentNo, frameNo,  predicted_seg_acc_arr, actual_seg_acc_arr, selected_config_id_arr, id_config_dict, each_frame_acc_arr, accumu_lag_seg_arr, profiling_time_arr, minAccuracy, outputPlotPdf)
        plotSimulateResult3(segmentNo, frameNo,  predicted_seg_acc_arr, actual_seg_acc_arr, selected_config_id_arr, id_config_dict, each_frame_acc_arr, accumu_lag_seg_arr, profiling_time_arr, minAccuracy, outputPlotPdf)
        
        # write into file
        #df_config_selected = pd.concat(config_selected_lst, axis=1)
        #out_seg_config_selected_file = outDir + "min_acc-" + str(minAccuracy) + "_selected_config"
        #df_config_selected.to_csv(out_seg_config_selected_file, sep='\t', index=False)

    #pdf.close()


def execute_multi_video_sim_bounded_acc():
    '''
    execute multiple video query simulation about lat threshold 
    '''
    video_dir_lst = ['output_001-dancing-10mins/', 'output_002-video_soccer-20mins/', 
                     'output_003-bike_race-10mins/', 'output_006-cardio_condition-20mins/'
                     ]
    
    for vd_dir in video_dir_lst[3:4]:
        data_pickle_dir = dataDir2 + vd_dir + 'pickle_files/'
        
        data_frame_dir = dataDir2 + vd_dir + 'frames_config_result/'
        executeDifferentBoundedAcc(data_frame_dir, data_pickle_dir)
        
if __name__== "__main__": 
        
    
    execute_multi_video_sim_bounded_acc()