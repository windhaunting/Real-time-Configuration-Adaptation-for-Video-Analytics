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


PLAYOUT_RATE = 25                  # 25-30  real time streaming speed (FPS)

dataDir1 = "input_output/mpii_dataset/output_01_mpii/profiling_result/segment_result/"

dataDir2 = "input_output/diy_video_dataset/"

def read_profile_data(dataFile):
    '''
    read the synthesized profile data
    '''
    df_config = pd.read_csv(dataFile, delimiter='\t', index_col=False)
    
    #print (df_config.columns)
    
    return df_config


def plotEachConfigOvertime(dataDir):
    '''
    draw the different configurations speed
    and accuracy overtime of the video,
    each line is a config overtime profiling_segmentTime4_segNo123
    '''
    start = 'segNo'
    end = '.tsv'
    #filePath.split('/')[-1][filePath.split('/')[-1].find(start)+len(start):filePath.split('/')[-1].rfind(end)
    filePathLst = sorted(glob(dataDir + "profiling_segmentTime4*.tsv"), key=lambda filePath: int(filePath.split('/')[-1][filePath.split('/')[-1].find(start)+len(start):filePath.split('/')[-1].rfind(end)]))          # [:75]   5 minutes = 75 segments

    
    #select a config according to minimal threshold
    selected_config_indexs_plots = []
    
    aver_accuracy_min = 0.8
    aver_detection_speed_min = 10
    aver_detect_speed_max = 50  
    
    config_indexs_plots = range(1, 61)  # [3] #range(1, 51)   # [31]
    
    # plot x and y
    outDir = dataDir + 'plot_result/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
        
    outputPlotPdf = outDir +"all_configs_-prior_accuracy_speed_20mins.pdf"  # "all_configs_-prior_accuracy_speed_9mins.pdf"
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outputPlotPdf)

    flag_plotAll = True        # True false
    for config_ind_plot in config_indexs_plots:
        x_lst = list(range(0, len(filePathLst)))
        
        speed_y_lst = []
        acc_y_lst = []
        
        
        for fileCnt, filePath in enumerate(filePathLst):
    
            df_config = read_profile_data(filePath)
            
            config_str = df_config.loc[df_config['Config_index'] == config_ind_plot].iloc[:, 0:4].to_string(header=False,
                  index=False).split('\n')
            #print ("config_str: ", config_str)
            
            #print ("config_ind_plot: ", filePath, config_ind_plot)
            df_speed = df_config.loc[df_config['Config_index'] == config_ind_plot, 'Detection_speed_FPS']
            if df_speed.empty:
               speed = 0
            else:
                speed = df_speed.item()
            speed_y_lst.append(speed)
            
            df_acc = df_config.loc[df_config['Config_index'] == config_ind_plot, 'Acc']
            if df_acc.empty:
               acc = 0
            else:
                acc = df_acc.item()
            acc_y_lst.append(acc)
            
            
 
        #print ("speed: ",config_ind_plot, speed_y_lst, acc_y_lst)
    
        # get average accuracy
        aver_acc = np.mean(acc_y_lst)
        
        aver_speed = np.mean(speed_y_lst)
        
        if flag_plotAll == True:
            selected_config_indexs_plots.append(config_ind_plot)
            x_label = "Video segment"
            y_label_1 = "Detection Speed (FPS)"
            y_label_2 = "ACC"
            outputPlotPdf = outDir + "config-" + str(config_str) + "-prior_accuracy_speed.pdf"
            #plotUpsideDownTwoFigures(x_lst, speed_y_lst, acc_y_lst, x_label, y_label_1, y_label_2, outputPlotPdf)
            #title_name = config_str
            fig = plotTwoSubplots(x_lst, speed_y_lst, acc_y_lst, x_label, y_label_1, y_label_2, config_str, outputPlotPdf)
            pdf.savefig(fig)
            
        elif aver_acc >= aver_accuracy_min and aver_speed >= aver_detection_speed_min and aver_speed <=aver_detect_speed_max:
            
            selected_config_indexs_plots.append(config_ind_plot)
            x_label = "Video segment"
            y_label_1 = "Detection Speed (FPS)"
            y_label_2 = "ACC"
            outputPlotPdf = outDir + "config-" + str(config_str) + "-prior_accuracy_speed.pdf"
            #plotUpsideDownTwoFigures(x_lst, speed_y_lst, acc_y_lst, x_label, y_label_1, y_label_2, outputPlotPdf)
            #title_name = config_str
            fig = plotTwoSubplots(x_lst, speed_y_lst, acc_y_lst, x_label, y_label_1, y_label_2, config_str, outputPlotPdf)
            pdf.savefig(fig)
        
    pdf.close()

    

def getConfigFromProfile(df_config, thres_acc, proc_speed_min):
    '''
    a very simple method to select a config
    thres_acc: minimum accuracy threshold 
    select the config with the maximum processing speed satisfyng the thres_acc
    
    '''
    df_select = df_config.loc[df_config['Detection_speed_FPS'] >= proc_speed_min].loc[df_config.Acc >= thres_acc]      
    #print ("df_indxAccSpeed: ", type(df_indxAccSpeed))
            
    if df_select.empty:
        series_selected_config = df_config.ix[df_config['Detection_speed_FPS'].idxmax()] 
            
        print ("df_select: ", df_select)
    else:
        series_selected_config = df_select.ix[df_select['Detection_speed_FPS'].idxmax()]   #select the whole rows with maximum processing speed
    #print ("selected_config: ",  series_selected_config, type(series_selected_config))
    
    return series_selected_config
    

    
def getMaxAccuracyInEachSegment(dataDir):
    '''
    assume we want to achieve maximum accuracy 1, plot the accumulated lags in each video segment    
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
    thres_acc = 1.0
    proc_speed_min = 10  #   150]       # [10, 50] select range around [10, 50]
    
    
    #define a buffer
    last_buffer_lst = cls_fifo()      # initial buffer size ;  here simulate to store  the frame id inside
    
    
    config_selected_seg_indx_lst = []    # each seg' selected config index
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
        curr_series_selected_config = getConfigFromProfile(df_config, thres_acc, proc_speed_min)
        
 
        # store segment's selected config               
        
        config_selected_seg_indx_lst.append(curr_series_selected_config.loc['Config_index'].astype(int))
        #curr_proc_time = 1.0/curr_series_selected_config['Detection_speed_FPS'] 
        #print ("config_selected_seg_indx_lst :", config_selected_seg_indx_lst)
        # for the extra of time in the segment except from profile  use the config
        #calculate how many frames have been processed in this time
        #curr_buffer_lst = copy.deepcopy(last_buffer_lst)       # copy into current buffer 
        
        p = 0
        accumlated_proc_time = 0.0
        streaming_proc_time = 0.0
        buff_len_acc = last_buffer_lst.length()      # accumulated buffer length, not real length during iteration of pop
        
        last_segment_buffered_frm_id = 0
        while(p < buff_len_acc):

            #print ("buff_len_acc :", last_buffer_lst.length(), buff_len_acc)
            info = last_buffer_lst.pop()
            #print ("info :", info)
            #series_config_used = get_frame_used_config(config_selected_Seg_dict, frm_id, segment_total_frames)
            #proc_time = 1/series_config_used['Detection_speed_FPS'] 
            frm_id = info[0]
            curr_proc_time = info[1]
            
            if curr_proc_time == -1:
                # use the current config
                curr_proc_time = 1.0/curr_series_selected_config['Detection_speed_FPS'] 
                
            #print ("proc_time: ",curr_proc_time )
            streaming_proc_time += curr_proc_time
            if streaming_proc_time >= 1.0/PLAYOUT_RATE:         # more frame streamed into the buffer
                end_frm_ind = end_frm_ind + 1                    # make sure  <= total_video_frame_len
                streaming_proc_time = 0.0
                if end_frm_ind <= total_video_frame_len:
                    last_buffer_lst.append((end_frm_ind, -1))         # 
                    buff_len_acc += 1
                else:
                    end_frm_ind = total_video_frame_len
                    print("streaming: finished 2: ", end_frm_ind, seg_ind)
                    stream_flag = True
                
            accumlated_proc_time += curr_proc_time
            if accumlated_proc_time >= extra_time_segment:  #acchieve the end of this current segment
                break
            p += 1
        
        last_segment_buffered_frm_id = end_frm_ind

        curr_lag = getBufferedLag(last_buffer_lst, PLAYOUT_RATE)
            
        #print ("last_buffer_lst len: ", seg_ind, accumlated_proc_time, extra_time_segment, last_buffer_lst.length(), curr_lag)
        # get current seg's config from buffer
        curr_seg_lag_lst.append(curr_lag)
        
        if stream_flag:
            break
    
        #break
    print ("curr_seg_lag_lst: ",  curr_seg_lag_lst, len(curr_seg_lag_lst))
    print ("config_selected_seg_indx_lst: ", config_selected_seg_indx_lst, len(config_selected_seg_indx_lst))
    print ("profiling_seg_time_lst: ", seg_ind+1, profiling_seg_time_lst, len(profiling_seg_time_lst))
    
    return seg_ind+1, curr_seg_lag_lst, config_selected_seg_indx_lst, profiling_seg_time_lst



def baselineOneConfigSimulate(dataDir):
    '''
    baseline: use one config all the way for all segment
    use one config all the way to
    '''
    
    baseline_config_indx = 21      # 10?
    
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
    thres_acc = 1.0
    
    
    #define a buffer
    last_buffer_lst = cls_fifo()      # initial buffer size ;  here simulate to store  the frame id inside
    
    
    config_selected_seg_indx_lst = []    # each seg' selected config index
    #series_selected_config = None
    profiling_seg_time_lst = []
    curr_seg_lag_lst = []          # each segment's lag time
    last_segment_buffered_frm_id = 0
    
    stream_flag = False        # flag for "finished video streaming"
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
            stream_flag = True        # finished video streaming
        for ind in range(start_frm_ind, end_frm_ind):
            last_buffer_lst.append((ind, -1))              # -1 indicate the current selected config, when streaming, the config is not timely determined
            
        #print ("start_frm_ind :", start_frm_ind, added_frm_len, last_buffer_lst.length())
        
        # decide which config to use for current segment
        #curr_series_selected_config = getConfigFromProfile(df_config, thres_acc, proc_speed_range)
        
        #print ("df_config: ", df_config)
        curr_series_selected_config = df_config.iloc[baseline_config_indx-1, :]
        
        # store segment's selected config               
        config_selected_seg_indx_lst.append(baseline_config_indx)        # curr_series_selected_config.loc['Config_index']
        #curr_proc_time = 1.0/curr_series_selected_config['Detection_speed_FPS'] 
        #print ("config_selected_seg_indx_lst :", config_selected_seg_indx_lst)
        # for the extra of time in the segment except from profile  use the config
        #calculate how many frames have been processed in this time
        #curr_buffer_lst = copy.deepcopy(last_buffer_lst)       # copy into current buffer 
        
        p = 0
        accumlated_proc_time = 0.0
        streaming_proc_time = 0.0
        buff_len_acc = last_buffer_lst.length()      # accumulated buffer length, not real length during iteration of pop
        
        last_segment_buffered_frm_id = 0
        while(p < buff_len_acc):

            #print ("buff_len_acc :", last_buffer_lst.length(), buff_len_acc)
            info = last_buffer_lst.pop()
            #print ("info :", info)
            #series_config_used = get_frame_used_config(config_selected_Seg_dict, frm_id, segment_total_frames)
            #proc_time = 1/series_config_used['Detection_speed_FPS'] 
            frm_id = info[0]
            curr_proc_time = info[1]
            
            if curr_proc_time == -1:
                # use the current config
                curr_proc_time = 1/curr_series_selected_config['Detection_speed_FPS'] 
                
            #print ("proc_time: ",curr_proc_time )
            streaming_proc_time += curr_proc_time
            if streaming_proc_time >= 1.0/PLAYOUT_RATE:         # more frame streamed into the buffer
                end_frm_ind = end_frm_ind + 1                    # make sure  <= total_video_frame_len
                streaming_proc_time = 0.0
                if end_frm_ind <= total_video_frame_len:
                    last_buffer_lst.append((end_frm_ind, -1))         # 
                    buff_len_acc += 1
                else:
                    end_frm_ind = total_video_frame_len
                    print("streaming: finished2: ", end_frm_ind, seg_ind)
                    stream_flag = True        # finished video streaming
                
            accumlated_proc_time += curr_proc_time
            if accumlated_proc_time >= extra_time_segment:  #acchieve the end of this current segment
                break
            p += 1
        
        last_segment_buffered_frm_id = end_frm_ind

        curr_lag = getBufferedLag(last_buffer_lst, PLAYOUT_RATE)
            
        #print ("last_buffer_lst len: ", seg_ind, accumlated_proc_time, extra_time_segment, last_buffer_lst.length(), curr_lag)
        # get current seg's config from buffer
        curr_seg_lag_lst.append(curr_lag)
        
        if stream_flag:
            break        

    print ("curr_seg_lag_lst: ",  curr_seg_lag_lst)
    print ("config_selected_seg_indx_lst: ", config_selected_seg_indx_lst)
    print ("profiling_seg_time_lst: ", profiling_seg_time_lst)
    
    return seg_ind+1, curr_seg_lag_lst, config_selected_seg_indx_lst, profiling_seg_time_lst

    
    
def plotSimulateResultMaxAccuracyInEachSegment(segmentNo, curr_seg_lag_lst, config_selected_seg_indx_lst, profiling_seg_time_lst, outputPlotPdf):
    '''
    plot the result of the simulation
    '''
    x_lst = range(0, segmentNo)
    
    y_lst_1 = curr_seg_lag_lst
    y_lst_2 = config_selected_seg_indx_lst
    y_lst_3 = profiling_seg_time_lst
    
    x_label = 'Segment no.'
    y_label_1 = 'Lag time (s)'
    y_label_2 = 'Config_index'
    y_label_3 = 'Profiling_time (s)'
    
    title_name = ''
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outputPlotPdf)

    fig = plotThreeSubplots(x_lst, y_lst_1, y_lst_2, y_lst_3, x_label, y_label_1, y_label_2, y_label_3, title_name, outputPlotPdf)
    pdf.savefig(fig)

    pdf.close()



if __name__== "__main__": 
    
    #plotEachConfigOvertime(dataDir1)
    
    '''
    dataDir = dataDir2 + "001_output_video_dancing_01/profiling_result/"
    plotEachConfigOvertime(dataDir)
    '''

    dataDir = dataDir2 + "002_output_video_soccer_01/profiling_result/"
    plotEachConfigOvertime(dataDir)

    '''
    dataDir = dataDir2 + "001_output_video_dancing_01/profiling_result/"
    outDir = dataDir + 'simulate_result_01_maxSpeed_no_limit/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
        
    outputPlotPdf = outDir + "simulate_result_01_maxSpeed_no_limit.pdf"       
    segmentNo, curr_seg_lag_lst, config_selected_seg_indx_lst, profiling_seg_time_lst = getMaxAccuracyInEachSegment(dataDir)
    plotSimulateResultMaxAccuracyInEachSegment(segmentNo, curr_seg_lag_lst, config_selected_seg_indx_lst, profiling_seg_time_lst, outputPlotPdf)
    '''
    
    '''
    dataDir = dataDir2 + "002_output_video_soccer_01/profiling_result/"
    outDir = dataDir + 'simulate_result_01_maxSpeed_no_limit/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)
        
    outputPlotPdf = outDir + "simulate_result_01_maxSpeed_no_limit.pdf"       
    segmentNo, curr_seg_lag_lst, config_selected_seg_indx_lst, profiling_seg_time_lst = getMaxAccuracyInEachSegment(dataDir)
    plotSimulateResultMaxAccuracyInEachSegment(segmentNo, curr_seg_lag_lst, config_selected_seg_indx_lst, profiling_seg_time_lst, outputPlotPdf)
    '''
    