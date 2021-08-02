#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:07:00 2019

@author: fubao
"""

'''
test the impact of profiling
consider the bounded accuracy, select best config to observe the lag

'''


import os
import pandas as pd
import copy
import random
import numpy as np
import copy
import operator
import math
import time

from blist import blist

from glob import glob
from collections import defaultdict
from plot import plotTwoDimensionScatter
from plot import plotTwoDimensionScatterLine
from plot import plotTwoDimensionMultiLines
from plot import plotUpsideDownTwoFigures
from plot import plotTwoSubplots
from plot import plotThreeSubplots

from common import retrieve_name
from common import cls_fifo
from common import getBufferedLag
from common import read_profile_data
from common import paddingZeroToInter

from common import dataDir2

from profiling.common_prof import PLAYOUT_RATE

import matplotlib.backends.backend_pdf


# actually online simulation
'''
simulate with a threshold of lag
and chek the accuracy can achieve?

different lag thresholds?

finally to get a plot with different average accuracy with different lags

Q1:  online simulation, does it keep within the lag in until the end of each segment?   
if not, we allow some fluctuation, why not set previous threshold higher? and  what's the point of threshold?

'''


    
    
def exec_plot_Profiling(dataDir, outPlotDir):
    
    if not os.path.exists(outPlotDir):
        os.mkdir(outPlotDir)
        
    dataFile = dataDir +  "profiling_segmentTime4_segNo47.tsv"
    df_config = read_profile_data(dataFile)
    
    x_spf = df_config.iloc[:,5].tolist()         # df.iloc[0] # first row of data frame 
    print ("x_spf: ", type(x_spf), x_spf)
    
    y_acc = df_config.iloc[:,6].tolist()
    xlabel = "Resource cost (FPS)"
    ylabel = "Accuracy"
    outputPlotPdf = outPlotDir + '/' + "profiling.pdf"
    plotTwoDimensionScatter(x_spf, y_acc, xlabel, ylabel, outputPlotPdf)


def greedilyGetConfigFromProfileBoundedAccAndProcess(df_config, df_frame_config, proc_speed_min, minAccuracy, last_buffer_lst, extra_time_segment, end_frm_ind, total_video_frame_len):
    '''
    a very simple method to select a config
    thres_acc: minimum accuracy threshold 
    select the config with the maximum processing speed satisfyng the thres_acc
    and then process the frames included in extra_time in a segment
    
    df_config is the segement's info,   acc dection speed of each config
    df_frame_config is each frame's info, acc, detection speed of each config
    
    '''
    
    curr_processed_acc_lst = blist()
    #print ("df_selected_sorted: empty? ", df_config)

    df_selected = df_config.loc[(df_config['Detection_speed_FPS'] >= proc_speed_min) & (df_config['Acc'] >= minAccuracy) ]   # , ['Config_index', 'Acc', 'Detection_speed_FPS']]        
    #print ("df_indxAccSpeed: ", type(df_indxAccSpeed))
    
    # greedily try config that has process the buffer and new streamed data in an allowed lag threshold
    
    df_selected_sorted = df_selected.sort_values(by='Detection_speed_FPS', ascending=False)
    stream_flag = False
    def processExtraSegmentWithBoundedAcc(row, last_buffer_lst, extra_time_segment, end_frm_ind, total_video_frame_len):
        # select a config that at most process all the buffered length with fasteset processing speed
        p = 0
        accumlated_proc_time = 0.0
        streaming_proc_time = 0.0
        last_buffer_lst_cpy = copy.deepcopy(last_buffer_lst) 
        buff_len_acc = last_buffer_lst_cpy.length()      # accumulated buffer length, not real length during iteration of pop
        end_frm_ind_cpy = copy.deepcopy(end_frm_ind)
        stream_flag = False
        while(p < buff_len_acc):

            #print ("buff_len_acc :", last_buffer_lst_cpy.length(), buff_len_acc)
            info = last_buffer_lst_cpy.pop()
            #print ("info :", info)
            #series_config_used = get_frame_used_config(config_selected_Seg_dict, frm_id, segment_total_frames)
            #proc_time = 1/series_config_used['Detection_speed_FPS'] 
            frm_id = info[0]
            curr_proc_time = info[1]
                        
            if curr_proc_time == -1:
                # use the current config's
                #curr_proc_time = 1.0/speed              # 1.0/curr_series_selected_config['Detection_speed_FPS'] 
                
                            # get currently selected config info from row
                cur_res = row['Resolution']
                cur_frmRate = row['Frame_rate']
                cur_Model = row['Model']
                
                #print ("cur_res, cur_frmRate, cur_Model:", cur_res, cur_frmRate, cur_Model)
                # then get the frame id's detection time and acc in this config
                #print("greedilyGetConfigFromProfileBoundedAccAndProcess curr_proc_time: ", frm_id, curr_proc_time, type(cur_res), cur_res, cur_frmRate, cur_Model)
               
                curr_proc_time_series = df_frame_config['Time_SPF'][(df_frame_config["Resolution"] == cur_res) & (df_frame_config["Model"] == cur_Model) & (df_frame_config["Image_path"].str.find(frm_id) != -1)]      # SPF
                if not curr_proc_time_series.empty:
                    
                    curr_proc_time = curr_proc_time_series.iloc[0] 
                
                else:
                    #print ("greedilyGetConfigFromProfileBoundedAccAndProcess curr_proc_time_series empty: " , type (curr_proc_time_series), curr_proc_time_series)
                    curr_proc_time = 1.0/row['Detection_speed_FPS']
                    
                #Time_SPF  df_frame_config.loc[
                #print ("greedilyGetConfigFromProfileBoundedAccAndProcess time: " , type (time), time)
                
                # because it only record frame_rate 25 in the df_frame_config
                curr_proc_time = (curr_proc_time/(math.ceil(PLAYOUT_RATE/cur_frmRate)+1))
                #print ("greedilyGetConfigFromProfileBoundedAccAndProcess time after: " , type (curr_proc_time), curr_proc_time)
            
                # get currently accuracy for each frame processed
                curr_acc_series = df_frame_config['Acc'][(df_frame_config["Resolution"] == cur_res) & (df_frame_config["Model"] == cur_Model) & (df_frame_config["Image_path"].str.find(frm_id) != -1)]      # SPF
                if not curr_acc_series.empty:
                    
                    curr_acc = curr_acc_series.iloc[0] 
                
                else:
                    #print ("greedilyGetConfigFromProfileBoundedAccAndProcess curr_proc_time_series empty: " , type (curr_proc_time_series), curr_proc_time_series)
                    curr_acc = minAccuracy
                
                curr_processed_acc_lst.append(curr_acc) 
                #x += 1
            #print ("proc_time: ",curr_proc_time )
            streaming_proc_time += curr_proc_time
            if streaming_proc_time >= 1.0/PLAYOUT_RATE:         # more frame streamed into the buffer
                end_frm_ind_cpy += 1                    # make sure  <= total_video_frame_len
                streaming_proc_time = 0.0
                if end_frm_ind_cpy <= total_video_frame_len:
                    end_frm_ind_cpy_str = paddingZeroToInter(end_frm_ind_cpy) + '.jpg'
                    last_buffer_lst_cpy.append((end_frm_ind_cpy_str, -1))         # 
                    buff_len_acc += 1
                else:
                    end_frm_ind_cpy = total_video_frame_len
                    stream_flag = True 
                    print("streaming: finished 2: ", end_frm_ind_cpy)
                
            accumlated_proc_time += curr_proc_time
            if accumlated_proc_time >= extra_time_segment:  #acchieve at the end of this current segment
                break
            p += 1
        
        return last_buffer_lst_cpy, end_frm_ind_cpy, stream_flag, curr_processed_acc_lst
    
    series_selected_config = None
    rest_lag = -1
    
    #print ("df_selected_sorted: empty? ", df_selected_sorted)
    #selected the first row with fast processing speed as selected_config
    for index, row in df_selected_sorted.iterrows():  
        #speed = row['Detection_speed_FPS']
        
        #print ("greedilyGetConfigFromProfileBoundedAccAndProcess type: ", type(row))
        t0 = time.time()

        last_buffer_lst, end_frm_ind_cpy, stream_flag, curr_processed_acc_lst = processExtraSegmentWithBoundedAcc(row, last_buffer_lst, extra_time_segment, end_frm_ind, total_video_frame_len)
        
        print ("elapsed time 2:", time.time() - t0)
        
        rest_lag = getBufferedLag(last_buffer_lst, PLAYOUT_RATE)
        
        series_selected_config = row
        #print ("greedilyGetConfigFromProfileBoundedAccAndProcess found the config: ", series_selected_config, rest_lag, lagThreshold)
        return series_selected_config, rest_lag, end_frm_ind_cpy, stream_flag, last_buffer_lst, curr_processed_acc_lst        # selected the first row
    
    return series_selected_config, rest_lag, end_frm_ind, stream_flag, last_buffer_lst, curr_processed_acc_lst
    

def getAllConfigFramesAccTime(frame_result_input_dir):
    # get each config's and its frame, it's accuracy and detection_speed

    filePathLst = sorted(glob(frame_result_input_dir + "*.tsv"))  # must read ground truth file(the most expensive config) first
    
    df_det_lst = []
    
    for fileCnt, filePath in enumerate(filePathLst):
        # read poste estimation detection result file
        df_det = read_profile_data(filePath)
        # det-> detection
        #print ("getEachSegmentProfilingAPTime filePath: ", fileCnt, filePath, df_det.columns)
        df_det_lst.append(df_det)
    
    df_all_det = pd.concat(df_det_lst)
    
    print ("df, len(df_all_det)", len(df_all_det))
    
    return df_all_det

def runVideoSimulationForAllSegmentsBoundedAccuracy(dataDir, frame_result_input_dir,  minAccuracy):
    '''
    run video simulation of bounded accuracy
 
    minAccuracy: bounded accuracy
    
    calculate the config for each config
    '''    
    df_frame_config = getAllConfigFramesAccTime(frame_result_input_dir)

    
        
    segmentTime = 4     #4s
    segment_total_frames = segmentTime * PLAYOUT_RATE
    profile_interval_Time = segmentTime//4
    profile_interval_frames = profile_interval_Time * PLAYOUT_RATE
    
    extra_time_segment = segmentTime - profile_interval_Time         # extra time in a segment not profiling 
    extra_frames_segment = extra_time_segment * PLAYOUT_RATE
    
    start = 'segNo'
    end = '.tsv'
    #filePath.split('/')[-1][filePath.split('/')[-1].find(start)+len(start):filePath.split('/')[-1].rfind(end)
    filePathLst = sorted(glob(dataDir + "profiling_segmentTime4_segNo*.tsv"), key=lambda filePath: int(filePath.split('/')[-1][filePath.split('/')[-1].find(start)+len(start):filePath.split('/')[-1].rfind(end)]))          # [:75]   5 minutes = 75 segments

    #filePathLst = sorted(glob(dataDir2 + "profiling_segmentTime4*.tsv"))          # [:75]   5 minutes = 75 segments

    segmentNo = len(filePathLst)
    
    total_video_frame_len = segmentNo * segmentTime * PLAYOUT_RATE
    #proc_speed_range = [10, 10000000]  #   150]       # [10, 50] select range around [10, 50]
    proc_speed_min = 0
    
    #define a buffer
    last_buffer_lst = cls_fifo()      # initial buffer size ;  here simulate to store  the frame id inside
    
    
    acc_seg_lst = blist()    # each seg' selected config acc
    #series_selected_config = None
    profiling_seg_time_lst = blist()
    curr_seg_lag_lst = blist()          # each segment's lag time
    last_segment_buffered_frm_id = 1          # start from 000001.jpg
    
    stream_flag = False        # finished video streaming
    
    
    config_selected_lst = blist()

    for seg_ind in range(0, segmentNo):
        minAccuracy_cp = minAccuracy
        #profile_time = 0.0
        #df_config = read_profile_data(filePathLst[i])
        #print("filePathLst: ", filePathLst[seg_ind])
        
        # profile, because we simulate, therefore we have the profiling result from the profile data
        df_config = read_profile_data(filePathLst[seg_ind])
        # get the current segment's config's accuracy
        df_config['SPF'] = 1.0/df_config['Detection_speed_FPS']
        
        profilingElapsedTime = df_config['SPF'].sum()   #df_config['Detection.sum()
        profiling_seg_time_lst.append(profilingElapsedTime)
        # how many video streamed in the cur_lag; not include profiling time
        start_frm_ind = last_segment_buffered_frm_id + profile_interval_frames             # 0 + 25 in the beginning
        
        #streamed_time = max(profilingElapsedTime - 
        #add how many frame in the buffer, i.e how many has been streamed
        added_frm_len = int((profilingElapsedTime-profile_interval_Time) * PLAYOUT_RATE) # considering profiling
        end_frm_ind = start_frm_ind + added_frm_len     # make sure  <= total_video_frame_len
        
        if end_frm_ind >= total_video_frame_len:
            end_frm_ind = total_video_frame_len
            print("streaming: finished 1: ", end_frm_ind, seg_ind)
            stream_flag = True
        for ind in range(start_frm_ind, end_frm_ind):
            ind_str = paddingZeroToInter(ind) + '.jpg'
            last_buffer_lst.append((ind_str, -1))               # -1 indicate the current selected config, when streaming, the config is not timely determined
            
        t0 = time.time()
        #print ("start_frm_ind :", start_frm_ind, added_frm_len, last_buffer_lst.length(), last_buffer_lst)
        
        last_buffer_lst_copy = copy.deepcopy(last_buffer_lst) 
        end_frm_ind_copy = end_frm_ind
        # decide which config to use for current segment        
        series_selected_config, curr_lag, end_frm_ind, stream_flag, last_buffer_lst, curr_processed_acc_lst = greedilyGetConfigFromProfileBoundedAccAndProcess(df_config, df_frame_config, proc_speed_min, minAccuracy, last_buffer_lst, extra_time_segment, end_frm_ind, total_video_frame_len)
 
        print ("time elapsed for switching and process extra segment: ", start_frm_ind, time.time()-t0)
        
        '''
        if series_selected_config is None:
            print ("can not satisfy with all config:", seg_ind)
            profiling_seg_time_lst = profiling_seg_time_lst[:-1]
            seg_ind -= 1
            break  
        '''
        
        while (series_selected_config is None):
            minAccuracy_cp -= 0.05
            print ("series_selected_config None here:", last_buffer_lst_copy.length(), end_frm_ind_copy)
            series_selected_config, curr_lag, end_frm_ind, stream_flag, last_buffer_lst, curr_processed_acc_lst = greedilyGetConfigFromProfileBoundedAccAndProcess(df_config, df_frame_config, proc_speed_min, minAccuracy_cp, last_buffer_lst_copy, extra_time_segment, end_frm_ind_copy, total_video_frame_len)
            if minAccuracy_cp <= 0:
                print ("can not satisfy with all config:", seg_ind)
                profiling_seg_time_lst = profiling_seg_time_lst[:-1]
                seg_ind -= 1
                break  
        
        curr_acc_list = blist([series_selected_config.loc['Acc']])*profile_interval_frames + curr_processed_acc_lst

        acc_seg_lst.append(sum(curr_acc_list)/len(curr_acc_list))
        
        last_segment_buffered_frm_id = end_frm_ind

        curr_lag = getBufferedLag(last_buffer_lst, PLAYOUT_RATE)
            
        #print ("last_buffer_lst len: ", seg_ind, accumlated_proc_time, extra_time_segment, last_buffer_lst.length(), curr_lag)
        # get current seg's config from buffer
        curr_seg_lag_lst.append(curr_lag)
        
        if stream_flag:
            break
        
        config_selected_lst.append(series_selected_config)
        
        #break    # debug only
    print ("curr_seg_lag_lst: ",  seg_ind, curr_seg_lag_lst, len(curr_seg_lag_lst))
    print ("acc_seg_lst: ", acc_seg_lst, len(acc_seg_lst))
    print ("profiling_seg_time_lst: ", seg_ind+1, profiling_seg_time_lst, len(profiling_seg_time_lst))
    
    #print ("config_selected_lst: ", config_selected_lst)
    
    return seg_ind+1, acc_seg_lst,  curr_seg_lag_lst, profiling_seg_time_lst, config_selected_lst




def plotSimulateResultLagsInEachSegment(segmentNo, acc_seg_lst,  curr_seg_lag_lst, profiling_seg_time_lst, minAccuracy):
    '''
    plot the result of the simulation of different lags of each segment with bounded accuracy
    '''
    x_lst = range(0, segmentNo)
    y_lst_1 = acc_seg_lst
    y_lst_2 = curr_seg_lag_lst
    y_lst_3 = profiling_seg_time_lst
    
    x_label = 'Segment no.'
    y_label_1 = 'Accuracy'
    y_label_2 = 'Lag time (s)'
    y_label_3 = 'Profiling_time (s)'
    
    title_name_1 = 'min_acc_threshold:' + str(minAccuracy) + '--Average_acc: ' + str(round(sum(acc_seg_lst)/len(acc_seg_lst),3))
    
    title_name_2 = "Average_lag: " + str(round(sum(curr_seg_lag_lst)/len(curr_seg_lag_lst),3))

    fig = plotThreeSubplots(x_lst, y_lst_1, y_lst_2, y_lst_3, x_label, y_label_1, y_label_2, y_label_3, title_name_1, title_name_2)
    return fig
    

def executeDifferentBoundedAcc(data_prof_Dir, frame_result_input_dir):
    '''
    get accuracy vs different threshold
    '''
    
    outDir = data_prof_Dir + 'simulate_result_05_boundedAccuracy/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    
    outputPlotPdf = outDir + "simulate_result_05_boundedAccuracy" + "_result_01.pdf"    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outputPlotPdf)

    min_acc_lst = [0.8, 0.85, 0.9, 0.95, 1.0]
    
    xList = min_acc_lst
    yList = []
    for minAccuracy in min_acc_lst[1:2]:         # 
        
        segmentNo, acc_seg_lst, curr_seg_lag_lst, profiling_seg_time_lst, config_selected_lst = runVideoSimulationForAllSegmentsBoundedAccuracy(data_prof_Dir, frame_result_input_dir,  minAccuracy)
        
        yList.append(sum(acc_seg_lst)/len(acc_seg_lst))
        fig = plotSimulateResultLagsInEachSegment(segmentNo, acc_seg_lst, curr_seg_lag_lst, profiling_seg_time_lst, minAccuracy)
        
        pdf.savefig(fig)
        
        # write into file
        df_config_selected = pd.concat(config_selected_lst, axis=1)
        out_seg_config_selected_file = outDir + "min_acc-" + str(minAccuracy) + "_selected_config"
        df_config_selected.to_csv(out_seg_config_selected_file, sep='\t', index=False)
    
    pdf.close()
    
    #outputPlotPdf = outDir + "simulate_result_03_lagThresholdsVSAccuracy.pdf"    
    #plotTwoDimensionScatterLine(xList, yList, 'Min Accuracy', 'Average accuracy', outputPlotPdf)
    

def execute_multi_video_sim_bounded_acc():
    '''
    execute multiple video query simulation about lat threshold 
    '''
    video_dir_lst = ['output_001-dancing-10mins/', 'output_002-video_soccer-20mins/', 'output_003-bike_race-10mins/', 'output_006-cardio_condition-20mins/'
                     ]
    
    for vd_dir in video_dir_lst[3:4]:
        dataDir = dataDir2 + vd_dir + 'profiling_result/'
        
        frame_result_input_dir = dataDir2 + vd_dir + 'frames_config_result/'
        print ("frame_result_input_dir: ",  frame_result_input_dir)
        executeDifferentBoundedAcc(dataDir, frame_result_input_dir)
        
    
    
if __name__== "__main__": 
        
    '''
    dataDir = dataDir2 + "output_006-cardio_condition-20mins/profiling_result/"           # , outPlotDir)
    outPlotDir = dataDir + "plot_result/"
    exec_plot_Profiling(dataDir, outPlotDir)
    '''
    
    
    execute_multi_video_sim_bounded_acc()
    
    