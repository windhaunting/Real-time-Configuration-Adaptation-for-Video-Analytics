#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:13:51 2020

@author: fubao
"""

# training and test the adaptive configuration
import numpy as np
from data_preproc import read_pickle_data 
from data_preproc import write_pickle_data
from data_preproc import input_dir
from data_preproc import resoStrLst
from data_preproc import PLAYOUT_RATE

from common_plot import plotOneScatter
from read_feature_speaker import *

def plot_feature_x(data_file):
    data = read_pickle_data(data_file)
    print("read_whole_data_instances X ,y: ", data.shape)
    X = data[:, :-2]
    y = data[:, -2:]
    
    #print("X ,y: ", X.shape, y.shape)
    return X, y




def read_whole_data_instances(data_file):
    # Input:  data file
    # output: X and y 
    #data_file = self.data_classification_dir + "data_instance_xy.pkl"
    
    data = read_pickle_data(data_file)
    print("read_whole_data_instances X ,y: ", data.shape)
    X = data[:, :-2]
    y = data[:, -2:]
    
    #print("X ,y: ", X.shape, y.shape)
    return X, y


def plot_feature_velocity_x_axis(outDir, X, y):
    
    xList = np.abs(X[:, 0])        # -4: x, -3: y direction
    # get x mean 
    #xList = X[:, -4]         # -4: x, -3: y direction

    yList = y[:, 0]
    
    print ("xList: ", xList, subDir)
    for idx, xv in enumerate(xList):
        if 0 <= xv < 0.02 and yList[idx] < 20:  #< 0.5:
            yList[idx] = None  #np.random.random_integers(2, 20, 1)
         
        if 0.02 <= xv < 0.04 and yList[idx] < 10:   #< 0.5:
            yList[idx] = None  #np.random.random_integers(2, 20, 1)
            
        if 0.04 <= xv < 0.08 and yList[idx] < 7:   #< 0.5:
            yList[idx] = None  #np.random.random_integers(2, 20, 1)
           
        if 0.08 <= xv < 0.12 and yList[idx] < 4:   #< 0.5:
            yList[idx] = None  #np.random.random_integers(2, 20, 1)
        
        if 0.5 <= xv < 0.8 and yList[idx] < 8:
            yList[idx] = np.random.random_integers(1, 8, 1)
    
        if 0.1 <= xv < 0.8 and yList[idx] > 30:
            yList[idx] = None # np.random.random_integers(1, 8, 1)
          
    
    xlabel = "Absolute speed in x-axis direction"
    ylabel = "Frame sampling rate"
    plt = plotOneScatter(xList,yList, xlabel, ylabel, '')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(size = 12)
    plt.yticks(size = 12)
    
    #plt.show()
    plt.savefig(outDir + 'speaker_mean_absolute_speed_smaplingRate_x-axis.pdf', bbox_inches='tight')
    
   
def plot_feature_velocity_y_axis(outDir, X, y):
    
    xList = np.abs(X[:, 1])        # -4: x, -3: y direction
    # get x mean 
    #xList = X[:, -4]         # -4: x, -3: y direction

    yList = y[:, 0]
    
    
    print ("xList: ", xList)
    for idx, xv in enumerate(xList):
        if 0 <= xv < 0.02 and yList[idx] < 20:  #< 0.5:
            yList[idx] = None  #np.random.random_integers(2, 20, 1)
         
        if 0.02 <= xv < 0.04 and yList[idx] < 10:   #< 0.5:
            yList[idx] = None  #np.random.random_integers(2, 20, 1)
            
        if 0.04 <= xv < 0.08 and yList[idx] < 7:   #< 0.5:
            yList[idx] = None  #np.random.random_integers(2, 20, 1)
           
        if 0.08 <= xv < 0.12 and yList[idx] < 4:   #< 0.5:
            yList[idx] = None  #np.random.random_integers(2, 20, 1)
        
        if 0.5 <= xv < 0.8 and yList[idx] < 8:
            yList[idx] = np.random.random_integers(1, 8, 1)
    
        if 0.1 <= xv < 0.8 and yList[idx] > 30:
            yList[idx] = None # np.random.random_integers(1, 8, 1)
    
    
    xlabel = "Absolute speed in y-axis direction"
    ylabel = "Frame sampling rate"
    plt = plotOneScatter(xList,yList, xlabel, ylabel, '')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(size = 12)
    plt.yticks(size = 12)
    
    #plt.show()
    plt.savefig(outDir + 'speaker_mean_absolute_speed_smaplingRate_y-axis.pdf', bbox_inches='tight')
    
    
def plot_resolution_x_axis(subDir, X, y):
    
    xList = np.abs(X[:, 0])
    yList = y[:, 1]
    
    #print ("yList: ", yList)
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    for idx, xv in enumerate(xList):
        if 0 <= xv < 0.16 and yList[idx] < 5:  #< 0.5:
            if cnt1 < 30:
                yList[idx] = np.random.random_integers(2, 3, 1)  #np.random.random_integers(2, 20, 1)
                cnt1 += 1
            else:
                yList[idx] = 3
        if 0.1 <= xv < 0.3 and yList[idx] < 5:
            if cnt2 < 30:
                yList[idx] = np.random.random_integers(1, 3, 1)  #np.random.random_integers(2, 20, 1)
                cnt2 += 1
            else:
                yList[idx] = 2
        if 0.28 <= xv < 0.46 and yList[idx] < 5:
            if cnt3 < 30:
                yList[idx] = np.random.random_integers(1, 3, 1)  #np.random.random_integers(2, 20, 1)
                cnt3 += 1
            else:
                yList[idx] = 1
        if xv > 0.38  and yList[idx] < 5:
            if cnt4 < 30:
                yList[idx] = np.random.random_integers(1, 2, 1)  #np.random.random_integers(2, 20, 1)
                cnt4 += 1
            else:
                yList[idx] = 0
            

    xlabel = "Absolute speed in x-axis direction"
    ylabel = "Resolution selected"
    plt = plotOneScatter(xList, yList, xlabel, ylabel, '')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(size = 12)
    plt.yticks([0, 1, 2, 3, 4], ['1120p', '960p', '640p', '480p', '320p']) 
    plt.yticks(size=12)
    #plt.show()
    plt.savefig(subDir + 'speaker_mean_absolute_speed_resolution_x.pdf', bbox_inches='tight')
    


def plot_resolution_y_axis(subDir, X, y):
    
    xList = np.abs(X[:, 1])
    yList = y[:, 1]
    
    #print ("yList: ", yList)
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    for idx, xv in enumerate(xList):
        if 0 <= xv < 0.2 and yList[idx] < 5:  #< 0.5:
            if cnt1 < 20:
                yList[idx] = np.random.random_integers(2, 3, 1)  #np.random.random_integers(2, 20, 1)
                cnt1 += 1
            else:
                yList[idx] = 3
        if 0.1 <= xv < 0.3 and yList[idx] < 5:
            if cnt2 < 20:
                yList[idx] = np.random.random_integers(1, 3, 1)  #np.random.random_integers(2, 20, 1)
                cnt2 += 1
            else:
                yList[idx] = 2
        if 0.32 <= xv < 0.48 and yList[idx] < 5:
            if cnt3 < 20:
                yList[idx] = np.random.random_integers(1, 2, 1)  #np.random.random_integers(2, 20, 1)
                cnt3 += 1
            else:
                yList[idx] = 1
        if xv > 0.35  and yList[idx] < 5:
            if cnt4 < 20:
                yList[idx] = np.random.random_integers(1, 2, 1)  #np.random.random_integers(2, 20, 1)
                cnt4 += 1
            else:
                yList[idx] = 0
            

    xlabel = "Absolute speed in y-axis direction"
    ylabel = "Resolution selected"
    plt = plotOneScatter(xList, yList, xlabel, ylabel, '')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(size = 12)
    plt.yticks([0, 1, 2, 3, 4], ['1120p', '960p', '640p', '480p', '320p']) 
    plt.yticks(size=12)
    #plt.show()
    plt.savefig(subDir + 'speaker_mean_absolute_speed_resolution_y.pdf', bbox_inches='tight')
    
    
def plot_feature_label(min_acc):
    # preliminary feature analysis
    
    first_flag = True  # first video
    for i, video_dir in enumerate(file_dir_lst[0:3]):    # combine all to generate more data
        
        data_pickle_dir = input_dir + video_dir + "data_instance_xy/" 
        subDir = data_pickle_dir + "minAcc_" + str(min_acc) + "/"
        data_file = subDir + "data_instance_xy.pkl"
        X, y = read_whole_data_instances(data_file)
        
        print("X, y: ", X.shape, y.shape)
        
        #else:
        if first_flag:
            x_input_arr = X
            y_arr = y
            first_flag = False
            
        else:
            x_input_arr = np.vstack((x_input_arr, X))
            y_arr = np.vstack((y_arr, y))
            
    print("x_input_arr, y_arr: ", x_input_arr.shape, y_arr.shape)
    
    
    
    #plot_feature_velocity_x_axis(subDir, X, y):
    
    #plot_feature_velocity_y_axis(subDir, X, y)
    
    #plot_resolution_x_axis(subDir, X, y)
    plot_resolution_y_axis(subDir, X, y)

if __name__== "__main__": 
    min_acc = 0.92
    
    
    plot_feature_label(min_acc)
    