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
from glob import glob
from collections import defaultdict
from plot import plotTwoDimensionScatter
from plot import plotTwoDimensionMultiLines
from plot import plotUpsideDownTwoFigures
from plot import plotTwoSubplots
from plot import plotThreeSubplots

from common import retrieve_name
from common import cls_fifo
from common import getBufferedLag

import matplotlib.backends.backend_pdf


# actually online simulation
'''
idea 1: considering profiling, when profiling segment, there is one question: 
    1) when doing profiling of 1 second frames, it takes about 6s, the streaming has already 6s, so there would be 5 second video clips missing
        if there is  no buffer available? then we directly process later frames to catch up the real streaming speed in 3s (extra segment time)?:
            we can make a stat about how many frames are missing?
            
    2) if we consider buffer when only doing profiling?  does it make sense? 

idea 2: no considering profiling, does it make sense?  does not make sense
'''


# simulate without buffer to check how many accuracy we can achieve and the lag with the segment number

PLAYOUT_RATE = 25                  # 25-30  real time streaming speed (FPS)

dataDir1 = "input_output/mpii_dataset/output_01_mpii/profiling_result/segment_result/"

dataDir2 = "input_output/diy_video_dataset/output_video_dancing_01/profiling_result/"


def read_profile_data(dataFile):
    '''
    read the synthesized profile data
    '''
    df_config = pd.read_csv(dataFile, delimiter='\t', index_col=False)
    
    #print (df_config.columns)
    
    return df_config



def getConfigFromProfileRealtime(df_config, min_speed_thres):
    '''
    a very simple method to select a config
    select the config with the maximum accuarcy speed that also can achieve realtime playout rate
    i.e. select a config above PLAYOUT_RATE
    '''
    
    
    df_select = df_config.loc[df_config['Detection_speed_FPS'] >= min_speed_thres]       
    #print ("df_indxAccSpeed: ", type(df_indxAccSpeed))
            
    #
    series_selected_config = df_select.ix[df_select['Acc'].idxmax()]   #select the whole rows with maximum processing speed
    #print ("selected_config: ",  series_selected_config, type(series_selected_config))
    
    return series_selected_config
    

def nobuffer_simulateStreaming_profiling(dataDir):
    '''
    idea 1. 1) considering profiling,   no buffer online simulation
    
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
    
    #  proc_speed_minimum = 10  #   150]       # [10, 50] select range around [10, 50]
    
    
    missing_frames_num_lst = []         # no buffer missed frame number due to profiling time
    missing_streamed_time_lst = []
    
    profiling_seg_time_lst = []    # each seg' selected config index
    #series_selected_config = None
    acc_seg__lst = []       # each segment time
    
    stream_flag = False        # finished video streaming
    for seg_ind in range(0, segmentNo):
        
        df_config = read_profile_data(filePathLst[seg_ind])
        # get the current segment's config's accuracy
        df_config['SPF'] = 1.0/df_config['Detection_speed_FPS']
        profilingElapsedTime = df_config['SPF'].sum()   #df_config['Detection.sum()
                
        profiling_seg_time_lst.append(profilingElapsedTime)
        missed_frm_len = int((profilingElapsedTime-profile_interval_Time) * PLAYOUT_RATE)
        missing_frames_num_lst.append(missed_frm_len)
        
        missed_stream_time = int((profilingElapsedTime-profile_interval_Time))
        missing_streamed_time_lst.append(missed_stream_time)
        
        
        # select a config to process for the rest of frames?
        curr_series_selected_config = getConfigFromProfileRealtime(df_config, PLAYOUT_RATE)

        #print ("curr_series_selected_config: ", curr_series_selected_config.loc['Acc'])
        
        acc_seg__lst.append(curr_series_selected_config.loc['Acc'])
        
        
    print ("acc_seg__lst: ",  acc_seg__lst)
    print ("missing_frames_num_lst: ",  missing_frames_num_lst)
    print ("missing_streamed_time_lst: ",  missing_streamed_time_lst)
    
    return segmentNo, acc_seg__lst,  missing_streamed_time_lst, profiling_seg_time_lst
        

def nobuffer_simulateStreaming_profilingTime_bufferOnly(dataDir):
    '''
    idea 1. 2) considering profiling only buffer,  online simulation
    
    
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
    
    #  proc_speed_minimum = 10  #   150]       # [10, 50] select range around [10, 50]
    
    
    missing_frames_num_lst = []         # no buffer missed frame number due to profiling time
    missing_streamed_time_lst = []
    
    profiling_seg_time_lst = []    # each seg' selected config index
    #series_selected_config = None
    acc_seg__lst = []       # each segment time
    stream_flag = False        # finished video streaming
    last_segment_buffered_frm_id = 0
    for seg_ind in range(0, segmentNo):
        
        df_config = read_profile_data(filePathLst[seg_ind])
        # get the current segment's config's accuracy
        df_config['SPF'] = 1.0/df_config['Detection_speed_FPS']
        profilingElapsedTime = df_config['SPF'].sum()   #df_config['Detection.sum()
                        
        end_frm_ind = last_segment_buffered_frm_id + int((profilingElapsedTime-profile_interval_Time) * PLAYOUT_RATE)    # make sure  <= total_video_frame_len
        
        if end_frm_ind >= total_video_frame_len:
            end_frm_ind = total_video_frame_len
            print("streaming: finished 1: ", end_frm_ind, seg_ind)
            stream_flag = True        # finished video streaming
        
        if not stream_flag:
        # also plus segment extra time
            end_frm_processed_ind = end_frm_ind + extra_frames_segment 
        

        # process this frames in real time
        min_speed_thres = (end_frm_processed_ind - last_segment_buffered_frm_id)/(extra_time_segment)
        
        #print ("min_speed_thres: ", end_frm_processed_ind - last_segment_buffered_frm_id, profilingElapsedTime-profile_interval_Time + extra_time_segment, min_speed_thres)

        profiling_seg_time_lst.append(profilingElapsedTime)
        
        
        # select a config to process for the rest of frames?
        curr_series_selected_config = getConfigFromProfileRealtime(df_config, min_speed_thres)

        #print ("curr_series_selected_config: ", curr_series_selected_config.loc['Acc'])
        
        acc_seg__lst.append(curr_series_selected_config.loc['Acc'])
        
        last_segment_buffered_frm_id = end_frm_ind

        if stream_flag:
            break
                    
    print ("acc_seg__lst: ",  acc_seg__lst)
    print ("profiling_seg_time_lst: ",  profiling_seg_time_lst)
    
    return seg_ind+1, acc_seg__lst, profiling_seg_time_lst

def plotSimulateResNobufferInEachSegment(segmentNo, acc_seg__lst, missing_streamed_time_lst, profiling_seg_time_lst, outputPlotPdf):
    '''
    plot the result of the simulation
    '''
    x_lst = range(0, segmentNo)
    
    y_lst_1 = acc_seg__lst
    y_lst_2 = missing_streamed_time_lst
    y_lst_3 = profiling_seg_time_lst
    
    x_label = 'Segment no.'
    y_label_1 = 'Accuracy '
    y_label_2 = 'Missed streaming time '
    y_label_3 = 'Profiling time (s)'
    
    title_name = 'Average acc: ' + str( sum(acc_seg__lst)/len(acc_seg__lst))

    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outputPlotPdf)

    fig = plotThreeSubplots(x_lst, y_lst_1, y_lst_2, y_lst_3, x_label, y_label_1, y_label_2, y_label_3, title_name, outputPlotPdf)
    pdf.savefig(fig)

    pdf.close()


def plotSimulateResbufferOnlyProfilingInEachSegment(segmentNo, acc_seg__lst, profiling_seg_time_lst, outputPlotPdf):
    '''
    plot the result of the simulation
    '''
    x_lst = range(0, segmentNo)
    
    y_lst_1 = acc_seg__lst
    y_lst_2 = profiling_seg_time_lst
    
    x_label = 'Segment no.'
    y_label_1 = 'Accuracy '
    y_label_2 = 'Profiling time (s)'
    
    title_name = 'Average acc: ' + str( sum(acc_seg__lst)/len(acc_seg__lst))

    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outputPlotPdf)

    fig = plotTwoSubplots(x_lst, y_lst_1, y_lst_2, x_label, y_label_1, y_label_2, title_name, outputPlotPdf)
    pdf.savefig(fig)

    pdf.close()


if __name__== "__main__": 
    
    '''
    outDir = dataDir2 + 'simulate_result_02_no_buffer/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
        
    outputPlotPdf = outDir + "simulate_dancing_no_buffer_result_01.pdf"    
    
    segmentNo, acc_seg__lst, missing_streamed_time_lst, profiling_seg_time_lst = nobuffer_simulateStreaming_profiling(dataDir2)
    plotSimulateResNobufferInEachSegment(segmentNo, acc_seg__lst, missing_streamed_time_lst, profiling_seg_time_lst, outputPlotPdf)
    
    '''
    
    outDir = dataDir2 + 'simulate_result_02_buffer_onlyProfiling/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
        
    outputPlotPdf = outDir + "simulate_dancing_bufferOnlyProfiling_result_01.pdf"    
    
    segmentNo, acc_seg__lst,  profiling_seg_time_lst = nobuffer_simulateStreaming_profilingTime_bufferOnly(dataDir2)
    plotSimulateResbufferOnlyProfilingInEachSegment(segmentNo, acc_seg__lst, profiling_seg_time_lst, outputPlotPdf)
    
    
    