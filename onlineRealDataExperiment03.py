#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 00:17:04 2019

@author: fubao
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:00:14 2019

@author: fubao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 00:33:22 2019

@author: fubao
"""

# real data of a video of 1 minutes 


import os
import pandas as pd
import copy
import random
import numpy as np
import copy

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


# simulate without buffer to check how many accuracy we can achieve and the lag with the segment number

PLAYOUT_RATE = 25                  # 25-30  real time streaming speed (FPS)

dataDir1 = "input_output/mpii_dataset/"

dataDir2 = "input_output/diy_video_dataset/"



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


def greedilyGetConfigFromProfileLagThreshold(df_config, proc_speed_min, lagThreshold, last_buffer_lst, extra_time_segment, end_frm_ind, total_video_frame_len):
    '''
    a very simple method to select a config
    thres_acc: minimum accuracy threshold 
    select the config with the maximum processing speed satisfyng the thres_acc
    
    '''
    df_selected = df_config.loc[(df_config['Detection_speed_FPS'] >= proc_speed_min)]   # , ['Config_index', 'Acc', 'Detection_speed_FPS']]        
    #print ("df_indxAccSpeed: ", type(df_indxAccSpeed))
    
    # greedily try config that has process the buffer and new streamed data in an allowed lag threshold
    
    df_selected_sorted = df_selected.sort_values(by='Acc', ascending=False)
    stream_flag = False
    def selectConfigWithLagThreshold(speed, last_buffer_lst, extra_time_segment, end_frm_ind, total_video_frame_len):
        
        
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
                # use the current config
                curr_proc_time = 1.0/speed        # 1.0/curr_series_selected_config['Detection_speed_FPS'] 
                
            #print ("proc_time: ",curr_proc_time )
            streaming_proc_time += curr_proc_time
            if streaming_proc_time >= 1.0/PLAYOUT_RATE:         # more frame streamed into the buffer
                end_frm_ind_cpy += 1                    # make sure  <= total_video_frame_len
                streaming_proc_time = 0.0
                if end_frm_ind_cpy <= total_video_frame_len:
                    last_buffer_lst_cpy.append((end_frm_ind_cpy, -1))         # 
                    buff_len_acc += 1
                else:
                    end_frm_ind_cpy = total_video_frame_len
                    stream_flag = True 
                    print("streaming: finished 2: ", end_frm_ind_cpy,)
                
            accumlated_proc_time += curr_proc_time
            if accumlated_proc_time >= extra_time_segment:  #acchieve the end of this current segment
                break
            p += 1
        
        return last_buffer_lst_cpy, end_frm_ind_cpy, stream_flag
    
    series_selected_config = None
    rest_lag = -1
    for index, row in df_selected_sorted.iterrows():  
        speed = row['Detection_speed_FPS']
        #print ("greedilyGetConfigFromProfileLagThreshold type: ", type(row), speed, row['Acc'])
        last_buffer_lst, end_frm_ind_cpy, stream_flag = selectConfigWithLagThreshold(speed, last_buffer_lst, extra_time_segment, end_frm_ind, total_video_frame_len)
        
        rest_lag = getBufferedLag(last_buffer_lst, PLAYOUT_RATE)
        if rest_lag <= lagThreshold:       # because sorted accuracy
            series_selected_config = row
            #print ("greedilyGetConfigFromProfileLagThreshold found the config: ", series_selected_config, rest_lag, lagThreshold)
            return series_selected_config, rest_lag, end_frm_ind_cpy, stream_flag, last_buffer_lst
    
    return series_selected_config, rest_lag, end_frm_ind, stream_flag, last_buffer_lst
    

    
def runVideoSimulationForAllSegments(dataDir, lagThreshold):
    '''
    run video simulation
    find the config to satisfy the goal that acchieve max accuracy within the lag threshold in each segment
    '''    
    
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
    proc_speed_min = 10
    
    #define a buffer
    last_buffer_lst = cls_fifo()      # initial buffer size ;  here simulate to store  the frame id inside
    
    
    acc_seg_lst = []    # each seg' selected config acc
    #series_selected_config = None
    profiling_seg_time_lst = []
    curr_seg_lag_lst = []          # each segment's lag time
    last_segment_buffered_frm_id = 0
    
    stream_flag = False        # finished video streaming
    for seg_ind in range(0, segmentNo):
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
        added_frm_len = int((profilingElapsedTime-profile_interval_Time) * PLAYOUT_RATE)
        end_frm_ind = start_frm_ind + added_frm_len     # make sure  <= total_video_frame_len
        
        if end_frm_ind >= total_video_frame_len:
            end_frm_ind = total_video_frame_len
            print("streaming: finished 1: ", end_frm_ind, seg_ind)
            stream_flag = True
        for ind in range(start_frm_ind, end_frm_ind):
            last_buffer_lst.append((ind, -1))              # -1 indicate the current selected config, when streaming, the config is not timely determined
            
        #print ("start_frm_ind :", start_frm_ind, added_frm_len, last_buffer_lst.length())
        
        # decide which config to use for current segment        
        series_selected_config, curr_lag, end_frm_ind, stream_flag, last_buffer_lst = greedilyGetConfigFromProfileLagThreshold(df_config, proc_speed_min, lagThreshold, last_buffer_lst, extra_time_segment, end_frm_ind, total_video_frame_len)
 
        if series_selected_config is None:
            print ("can not satisfy with all config:", seg_ind)
            profiling_seg_time_lst = profiling_seg_time_lst[:-1]
            break            
        acc_seg_lst.append(series_selected_config.loc['Acc'])

        last_segment_buffered_frm_id = end_frm_ind

        curr_lag = getBufferedLag(last_buffer_lst, PLAYOUT_RATE)
            
        #print ("last_buffer_lst len: ", seg_ind, accumlated_proc_time, extra_time_segment, last_buffer_lst.length(), curr_lag)
        # get current seg's config from buffer
        curr_seg_lag_lst.append(curr_lag)
        
        if stream_flag:
            break
    
        #break
    print ("curr_seg_lag_lst: ",  seg_ind, curr_seg_lag_lst, len(curr_seg_lag_lst))
    print ("acc_seg_lst: ", acc_seg_lst, len(acc_seg_lst))
    print ("profiling_seg_time_lst: ", seg_ind+1, profiling_seg_time_lst, len(profiling_seg_time_lst))
    
    return seg_ind+1, acc_seg_lst,  curr_seg_lag_lst, profiling_seg_time_lst



def plotSimulateResultMaxAccuracyInEachSegment(segmentNo, acc_seg_lst,  curr_seg_lag_lst, profiling_seg_time_lst, lagThre):
    '''
    plot the result of the simulation
    '''
    x_lst = range(0, segmentNo)
    y_lst_1 = acc_seg_lst
    y_lst_2 = curr_seg_lag_lst
    y_lst_3 = profiling_seg_time_lst
    
    x_label = 'Segment no.'
    y_label_1 = 'Accuracy'
    y_label_2 = 'Lag time (s)'
    y_label_3 = 'Profiling_time (s)'
    
    title_name_1 = 'Lag_Threshold:' + str(lagThre) + 's--Average_acc: ' + str(round(sum(acc_seg_lst)/len(acc_seg_lst),3))
    
    title_name_2 = "Average_lag: " + str(round(sum(curr_seg_lag_lst)/len(curr_seg_lag_lst),3))

    fig = plotThreeSubplots(x_lst, y_lst_1, y_lst_2, y_lst_3, x_label, y_label_1, y_label_2, y_label_3, title_name_1, title_name_2)
    return fig
    

def executeLagThresholds(dataDir):
    '''
    get accuracy vs different threshold
    '''
    
    outDir = dataDir + 'simulate_result_03_lagThreshold/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    
    outputPlotPdf = outDir + "simulate_result_03_lagThreshold" + "_result_01.pdf"    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outputPlotPdf)

    lagThresholds = [0, 1, 5, 10, 20, 30, 40, 50, 80, 100, 200, 300]
    
    xList = lagThresholds
    yList = []
    for lagThres in lagThresholds:
        
        segmentNo, acc_seg_lst, curr_seg_lag_lst, profiling_seg_time_lst = runVideoSimulationForAllSegments(dataDir, lagThres)
        
        yList.append(sum(acc_seg_lst)/len(acc_seg_lst))
        fig = plotSimulateResultMaxAccuracyInEachSegment(segmentNo, acc_seg_lst, curr_seg_lag_lst, profiling_seg_time_lst, lagThres)
        
        pdf.savefig(fig)
    pdf.close()
    
    outputPlotPdf = outDir + "simulate_result_03_lagThresholdsVSAccuracy.pdf"    
    plotTwoDimensionScatterLine(xList, yList, 'Lag threshold', 'Average accuracy', outputPlotPdf)
    
    
    
if __name__== "__main__": 
    
    '''
    
    dataDir = dataDir2 + '001_output_video_dancing_01/profiling_result/'
    outPlotDir = dataDir + "one_profiling_result_AccVSResources.pdf"
    exec_plot_Profiling(dataDir, outPlotDir)
        
    '''

    '''
    dataDir = dataDir2 + '001_output_video_dancing_01/profiling_result/'
    executeLagThresholds(dataDir)
    
    '''
    
    dataDir = dataDir2 + '002_output_video_soccer_01/profiling_result/'
    executeLagThresholds(dataDir)
    
    
    '''
    dataDir = dataDir2 + '003-output_bike_race-20mins_01/profiling_result/'
    executeLagThresholds(dataDir)
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    