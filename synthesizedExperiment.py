#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:17:06 2019

@author: fubao
"""

# synthesize to verify the greedy heuristic method 01

import os
import pandas as pd
import copy

from plot import plotTwoDimensionScatter
from plot import plotTwoDimensionMultiLines

from common import retrieve_name
from common import cls_fifo

dataDir = "input_output/"

STANDARD_FPS = 30                  # 25-30  real time streaming speed (FPS)

outPlotDir = dataDir + 'synthesize_experiment_plot'


def exec_plot_Profiling():
    
    if not os.path.exists(outPlotDir):
        os.mkdir(outPlotDir)
        
    dataFile = dataDir +  "synthesize_profiling.tsv"
    df_config = read_profile_data(dataFile)
    
    x_spf = df_config.iloc[:,5].tolist()         # df.iloc[0] # first row of data frame 
    print ("x_spf: ", type(x_spf), x_spf)
    
    y_acc = df_config.iloc[:,4].tolist()
    xlabel = "Resource cost (FPS)"
    ylabel = "Accuracy (AP50) "
    outputPlotPdf = outPlotDir + '/' + "profiling.pdf"
    plotTwoDimensionScatter(x_spf, y_acc, xlabel, ylabel, outputPlotPdf)
    
    
def read_profile_data(dataFile):
    '''
    read the synthesized profile data
    '''
    df_config = pd.read_csv(dataFile, delimiter='\t')
    
    print (df_config.columns)
    
    return df_config


def plot_streaming_buffer(x_segments, y_buffer_lsts, flagChange, changeLengendLst, fixedVar1, fixedVar2):
    '''
    plot buffer size vs segment
    flag indicate : which variable change . e.g buffer size, or ACC_MIN or seg_time_len changed
    '''
    
    if not os.path.exists(outPlotDir):
        os.mkdir(outPlotDir)
        
        
    xlabel = "Segment"
    ylabel = "Lag (second)"
    subDir = outPlotDir + '/' +  flagChange + '_change'
    if not os.path.exists(subDir):
        os.mkdir(subDir)
            
    outputPlotPdf = subDir + '/' + retrieve_name(fixedVar1) + '-' + str(fixedVar1) + '_' + retrieve_name(fixedVar2) + '-' + str(fixedVar2)  + '_' + flagChange + '_change_' + 'buffer_vs_segment.pdf'

    plotTwoDimensionMultiLines(x_segments, y_buffer_lsts,  xlabel, ylabel, changeLengendLst, outputPlotPdf)
        

def plot_streaming_acc(x_segments, y_acc_lsts, flagChange, changeLengendLst,  fixedVar1, fixedVar2):
    '''
    plot acc vs segment
    flag indicate : which variable change . e.g buffer size, or ACC_MIN or seg_time_len changed
    '''
    
    if not os.path.exists(outPlotDir):
        os.mkdir(outPlotDir)
        
        
    xlabel = "Segment"
    ylabel = "Accuracy (AP50)"
    subDir = outPlotDir + '/' +  flagChange + '_change'
    if not os.path.exists(subDir):
        os.mkdir(subDir)
            
    outputPlotPdf = subDir + '/' + retrieve_name(fixedVar1) + '-' + str(fixedVar1) + '_' + retrieve_name(fixedVar2) + '-' + str(fixedVar2)  + '_' + flagChange + '_change_' + 'acc_vs_segment.pdf'

    print(" fixedVar1, fixedVar2: ", fixedVar1, fixedVar2, outputPlotPdf)   
    plotTwoDimensionMultiLines(x_segments, y_acc_lsts, xlabel, ylabel, changeLengendLst, outputPlotPdf)
    
    
def getBufferLag(ans_current_buffer):
    '''
    calculate how many lags (s) from how many frames stored in the buffer
    according to STANDARD_FPS
    '''
    buf_len = ans_current_buffer.length()
    lag = buf_len/STANDARD_FPS
    
    return lag
    

#def writeAverageBufferAcc(ACC_MIN, BUFF_FRM_MAX, seg_time_len, buffer_size_lst, acc_lst):
#    '''
#    write the accuracy and buffer into file
#    '''
    
    
def video_streaming_simulation(df_config):
    '''
    simulation video streaming with greedy heuristic
    '''
    
    
    
    ACC_MIN_LST = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]                  # 0.8, 0.9 lower bound of accuracy 

    BUFF_FRM_MAX_LST =  [i*STANDARD_FPS for i in range(0, 11)]       # 1, 2, 3s, 5s and 8s, lower bound Buffer size of frames

    print ("BUFF_FRM_MAX_LST: ", BUFF_FRM_MAX_LST)
    #mLst = [1, 2, 3]      # 20, 30, number of m segment  to do a statistic
    
    seg_time_len_lst = [3, 4, 6, 8, 10, 12, 16]        #  4s, 6s,8s, 10s, 12s, 16s  segment time length
    
    
    # simulate video streaming to switch configuration

    m = 31   # segment
    
    '''
    #1st consider  fixed set_time_len  and BUFF_FRM_MAX for each plt
    for seg_time_len in seg_time_len_lst:
        for BUFF_FRM_MAX in BUFF_FRM_MAX_LST:
            
            y_buffer_lsts_diff_ACC_MIN = []          # y axis is buffer
            y_acc_lsts_diff_ACC_MIN = []             # y axis is acc

            for ACC_MIN in ACC_MIN_LST:
                last_buffer_lst = cls_fifo()      # initial buffer size = 0
                segIndex = 1
                seg_Index_lst = []
                buffer_size_lst = []        # each index is a segment index
                acc_lst = []         # accumulated accuracy
    
                while (True):           # video streaming
                            
                    #print ("ACC_MIN: ", ACC_MIN)
                    ans_config_index, ans_current_buffer, ans_acc = greedy_heuristic_select_config(ACC_MIN, BUFF_FRM_MAX, seg_time_len, df_config, last_buffer_lst)
                    
                    #print ("ans_config_index, ans_current_buffer: ", seg_time_len, BUFF_FRM_MAX, ACC_MIN, ans_config_index, last_buffer_lst.length(), ans_current_buffer.length(), ans_acc)
                    
                    #buffer_size_lst.append(ans_current_buffer.length())
                    buffer_size_lst.append(getBufferLag(ans_current_buffer))
                    acc_lst.append(ans_acc)
                    
                    seg_Index_lst.append(segIndex)

                    last_buffer_lst = ans_current_buffer             # update buffer 

                    segIndex += 1
                    if segIndex >= m:  # test only
                        break
                    
                y_buffer_lsts_diff_ACC_MIN.append(buffer_size_lst)           
                                        
                y_acc_lsts_diff_ACC_MIN.append(acc_lst)
                    
                #print ("final buffer size:", seg_time_len, BUFF_FRM_MAX, ACC_MIN, buffer_size_lst, seg_Index_lst, acc_lst)         
            
            #print ("final buffer size:",seg_time_len, BUFF_FRM_MAX, ACC_MIN, y_acc_lsts_diff_ACC_MIN)

            flagChange = 'ACC_MIN'
            #plot buffer vs sgement
            plot_streaming_buffer(seg_Index_lst, y_buffer_lsts_diff_ACC_MIN, flagChange, ACC_MIN_LST, seg_time_len, BUFF_FRM_MAX)
            
            #plot acc vs sgement
            plot_streaming_acc(seg_Index_lst, y_acc_lsts_diff_ACC_MIN, flagChange, ACC_MIN_LST, seg_time_len, BUFF_FRM_MAX)                
    '''
    
    
    #2nd consider  fixed set_time_len  and ACC_MIN for each plt
    for seg_time_len in seg_time_len_lst:        
        for ACC_MIN in ACC_MIN_LST:
    
            y_buffer_lsts_diff_BUFF_FRM_MAX = []          # y axis is buffer
            y_acc_lsts_diff_BUFF_FRM_MAX = []             # y axis is acc
            for BUFF_FRM_MAX in BUFF_FRM_MAX_LST:
                last_buffer_lst = cls_fifo()      # initial buffer size = 0
                segIndex = 1               # set segment index
                seg_Index_lst = []
                buffer_size_lst = []        # each index is a segment index
                acc_lst = []                # accumulated accuracy
                
                while (True):               # video streaming
                            
                    #print ("ACC_MIN: ", ACC_MIN)
                    ans_config_index, ans_current_buffer, ans_acc = greedy_heuristic_select_config(ACC_MIN, BUFF_FRM_MAX, seg_time_len, df_config, last_buffer_lst)
                    
                    #print ("ans_config_index: ", seg_time_len, ACC_MIN, BUFF_FRM_MAX, ans_config_index, ans_current_buffer_size, ans_acc)
                    
                    #buffer_size_lst.append(ans_current_buffer.length())
                    buffer_size_lst.append(getBufferLag(ans_current_buffer))
                    acc_lst.append(ans_acc)
                    
                    seg_Index_lst.append(segIndex)
                    
                    last_buffer_lst = ans_current_buffer             # update buffer 
                    
                    segIndex += 1
                    if segIndex >= m:       # test only
                        break
                
                    
                y_buffer_lsts_diff_BUFF_FRM_MAX.append(buffer_size_lst)           
                                        
                y_acc_lsts_diff_BUFF_FRM_MAX.append(acc_lst)
                    
                #print ("final buffer size:", seg_time_len, BUFF_FRM_MAX, ACC_MIN, buffer_size_lst, seg_Index_lst, acc_lst)         
            
            #print ("final buffer size:",seg_time_len, BUFF_FRM_MAX, ACC_MIN, y_acc_lsts_diff_BUFF_FRM_MAX)
            #print ("final seg_time_len, ACC_MIN, BUFF_FRM_MAX :",seg_time_len, ACC_MIN, BUFF_FRM_MAX, len(y_buffer_lsts_diff_BUFF_FRM_MAX), y_buffer_lsts_diff_BUFF_FRM_MAX, y_acc_lsts_diff_BUFF_FRM_MAX)

            flagChange = 'BUFF_FRM_MAX'
            #plot buffer vs sgement
            changeLengend = [i/STANDARD_FPS for i in BUFF_FRM_MAX_LST]
            plot_streaming_buffer(seg_Index_lst, y_buffer_lsts_diff_BUFF_FRM_MAX, flagChange, changeLengend, seg_time_len, ACC_MIN)
            
            #plot acc vs sgement
            plot_streaming_acc(seg_Index_lst, y_acc_lsts_diff_BUFF_FRM_MAX, flagChange, changeLengend, seg_time_len, ACC_MIN)                
    
    '''
    #3rd consider  fixed set_time_len  and ACC_MIN for each plt
    for BUFF_FRM_MAX  in BUFF_FRM_MAX_LST:        
        for ACC_MIN in ACC_MIN_LST:
    
            y_buffer_lsts_diff_BUFF_FRM_MAX = []          # y axis is buffer
            y_acc_lsts_diff_BUFF_FRM_MAX = []             # y axis is acc
            for seg_time_len in seg_time_len_lst:
                last_buffer_lst = cls_fifo()      # initial buffer size = 0
                segIndex = 1               # set segment index
                seg_Index_lst = []
                buffer_size_lst = []        # each index is a segment index
                acc_lst = []                # accumulated accuracy
                
                while (True):               # video streaming
                            
                    #print ("ACC_MIN: ", ACC_MIN)
                    ans_config_index, ans_current_buffer, ans_acc = greedy_heuristic_select_config(ACC_MIN, BUFF_FRM_MAX, seg_time_len, df_config, last_buffer_lst)
                    
                    #print ("ans_config_index: ", seg_time_len, ACC_MIN, BUFF_FRM_MAX, ans_config_index, ans_current_buffer_size, ans_acc)
                    
                    #buffer_size_lst.append(ans_current_buffer.length())
                    buffer_size_lst.append(getBufferLag(ans_current_buffer))
                    acc_lst.append(ans_acc)
                    
                    seg_Index_lst.append(segIndex)
                    
                    last_buffer_lst = ans_current_buffer             # update buffer 
                    
                    segIndex += 1
                    if segIndex >= m:       # test only
                        break
                
                    
                y_buffer_lsts_diff_BUFF_FRM_MAX.append(buffer_size_lst)           
                                        
                y_acc_lsts_diff_BUFF_FRM_MAX.append(acc_lst)
                    
                #print ("final buffer size:", seg_time_len, BUFF_FRM_MAX, ACC_MIN, buffer_size_lst, seg_Index_lst, acc_lst)         
            
            #print ("final buffer size:",seg_time_len, BUFF_FRM_MAX, ACC_MIN, y_acc_lsts_diff_BUFF_FRM_MAX)
            #print ("final seg_time_len, ACC_MIN, BUFF_FRM_MAX :",seg_time_len, ACC_MIN, BUFF_FRM_MAX, len(y_buffer_lsts_diff_BUFF_FRM_MAX), y_buffer_lsts_diff_BUFF_FRM_MAX, y_acc_lsts_diff_BUFF_FRM_MAX)

            flagChange = 'SEG_TIME_LEN'
            #plot buffer vs sgement
            plot_streaming_buffer(seg_Index_lst, y_buffer_lsts_diff_BUFF_FRM_MAX, flagChange, seg_time_len_lst, BUFF_FRM_MAX, ACC_MIN)
            
            #plot acc vs sgement
            plot_streaming_acc(seg_Index_lst, y_acc_lsts_diff_BUFF_FRM_MAX, flagChange, seg_time_len_lst, BUFF_FRM_MAX, ACC_MIN)                
    '''
    

def processBufferedFrames(current_buffer_lst, seg_time_len):
    '''
    process buffered frames in last buffer lst and current buffer lst 
    total time a video seg_time_len
    '''
    p = 0
    
    total_size = current_buffer_lst.length()
    
    acc_time = 0
    
    #print ("processBufferedFrames, current_buffer_lst: ", current_buffer_lst.data, total_size)
    while (p < total_size):
        
        info = current_buffer_lst.pop()
        time = 1/info[1]         # each frame process time
        acc_time += time
        if acc_time >= seg_time_len:
            break
        p += 1
    
    return current_buffer_lst

    
    
def greedy_heuristic_select_config(ACC_MIN, BUFF_FRM_MAX, seg_time_len, df_config, last_buffer_lst):
    '''
    select a config
    last_buffer_size: last segment buffer size
    '''
        
    cols = df_config.columns
    # sort the config by accuarcy
    df_config = df_config.sort_values(by = cols[4], ascending = False)
    
    #print (df_config.values)
    
    ans_config_index =  0        # the config index
    ans_current_buffer = cls_fifo()
    ans_acc = 0
    
    current_buffer_lst = cls_fifo() 
    for row in df_config.itertuples(index=False, name='Pandas'):
        #print (getattr(row, "c1"), getattr(row, "c2"))
        #print ("row: ", row[0])
        #get config_index, frame_rate, resolution, model, acc, speed
        confIndex = row[0]
        #frate = row[1]
        #reso = row[2]
        #model = row[3]
        acc = row[4]
        spf = row[5]
        
        #print ("rows: ", confIndex,  acc, spf)
        
        if acc < ACC_MIN:     # directly exit, because it's in non-descending order
            break
        
        # calculate buffer size
        #current_buffer_size = max(seg_time_len * STANDARD_FPS - seg_time_len * spf + last_buffer_size, 0)
        #current_buffer_size = current_seg_extra
        # assume we put all the frames in the video segment into the buffer queue
        
        current_buffer_lst = copy.deepcopy(last_buffer_lst)       # copy into current buffer 
        #print (" greedy_heuristic_select_config current_buffer_lst here: ", current_buffer_lst.length())
        for i in range (seg_time_len*STANDARD_FPS):
            current_buffer_lst.append((i, spf))        
        # process buffer frame
        current_buffer_lst = processBufferedFrames(current_buffer_lst, seg_time_len)
        
        
        #print (" greedy_heuristic_select_config current_buffer_size: ", confIndex, ACC_MIN, BUFF_FRM_MAX, seg_time_len, last_buffer_lst.length(), current_buffer_lst.length())
        
        ans_config_index = confIndex
        ans_current_buffer = current_buffer_lst
        ans_acc = acc
        
        if current_buffer_lst.length() > BUFF_FRM_MAX:
            continue
        else:
            #print ("configIndex: ", acc, confIndex, current_buffer_size)        
            ans_config_index = confIndex
            ans_current_buffer = current_buffer_lst
            ans_acc = acc
            #print ("greedy_heuristic_select_config enter here,  find suitable config: ", ACC_MIN, BUFF_FRM_MAX, seg_time_len, ans_config_index, last_buffer_lst.length(), ans_current_buffer.length(), ans_acc)
            break             # find the answer

   
    # use the first config with the minimum accuracy satisfying accuracy requirement, but not buffer requirement
    return ans_config_index, ans_current_buffer, ans_acc

            

def getMostExpensiveConfig():
    '''
    get the most expensive config
    most expensive
    '''
    x = 1
    
def exec_simulation_video_analytic():
    '''
    main entry to execute video streaming
    '''
    
    dataFile = dataDir +  "synthesize_profiling.tsv"
    df_config = read_profile_data(dataFile)
    video_streaming_simulation(df_config)
    
    
    
if __name__== "__main__": 
    
    #exec_plot_Profiling()
    
    exec_simulation_video_analytic()

    