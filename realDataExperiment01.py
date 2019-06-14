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

from plot import plotTwoDimensionScatter
from plot import plotTwoDimensionMultiLines
from plot import plotUpsideDownTwoFigures
from plot import plotTwoSubplots

from common import retrieve_name
from common import cls_fifo

import matplotlib.backends.backend_pdf

dataDir = "input_output/"

STANDARD_FPS = 25                  # 25-30  real time streaming speed (FPS)


dataDir1 = "input_output/mpii_dataset/output_01_mpii/profiling_result/segment_result/"

dataDir2 = "input_output/diy_video_dataset/output_video_dancing_01/profiling_result/"

def read_profile_data(dataFile):
    '''
    read the synthesized profile data
    '''
    df_config = pd.read_csv(dataFile, delimiter='\t')
    
    #print (df_config.columns)
    
    return df_config


def plotEachConfigOvertime(dataDir):
    '''
    draw the different configurations speed
    and accuracy overtime of the video,
    each line is a config overtime
    '''
    filePathLst = sorted(glob(dataDir + "profiling_segmentTime*.tsv"))          # [:75]   5 minutes = 75 segments

    
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
        
    outputPlotPdf = outDir +"selected_configs_-prior_accuracy_speed_9mins.pdf"  # "all_configs_-prior_accuracy_speed_9mins.pdf"
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(outputPlotPdf)

    flag_plotAll = False        # True false
    for config_ind_plot in config_indexs_plots:
        x_lst = list(range(0, len(filePathLst)))
        
        speed_y_lst = []
        acc_y_lst = []
        
        
        for fileCnt, filePath in enumerate(filePathLst):
    
            df_config = read_profile_data(filePath)
            
            config_str = df_config.loc[df_config['Config_index'] == config_ind_plot].iloc[:, 0:4].to_string(header=False,
                  index=False).split('\n')
            #print ("config_str: ", config_str)
            
            speed = df_config.loc[df_config['Config_index'] == config_ind_plot, 'Detection_speed_FPS'].item()
            speed_y_lst.append(speed)
            
            acc = df_config.loc[df_config['Config_index'] == config_ind_plot, 'Acc'].item()
            
            acc_y_lst.append(acc)
            
            
 
        print ("speed: ",config_ind_plot, speed_y_lst, acc_y_lst)
    
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

        

if __name__== "__main__": 
    
    #plotEachConfigOvertime(dataDir1)
    
    plotEachConfigOvertime(dataDir2)
