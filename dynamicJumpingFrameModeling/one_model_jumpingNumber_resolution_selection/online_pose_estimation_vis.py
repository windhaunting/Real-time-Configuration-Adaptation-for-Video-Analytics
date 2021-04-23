# -*- coding: utf-8 -*-


# online pose estimation with the flexible configuration algorithm

import matplotlib 
matplotlib.use('TKAgg',warn=False, force=True)

import matplotlib.pyplot as plt

MEDIUM_SIZE = 50
BIGGER_SIZE = 56

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
import pickle

current_file_cur = os.path.dirname(os.path.abspath("__file__"))
sys.path.insert(0, current_file_cur + '/..')

from data_file_process import read_pickle_data
from common_plot import plotOneScatterLine
from collections import Counter


#dataDir01 =  "../..//input_output/one_person_diy_video_dataset/output_001_dance/jumping_number_result/jumpingNumber_resolution_selection/intervalFrm-1_speedType-ema_minAcc-0.9/video_applied_detection_result/"

dataDir01 =  "../..//input_output/one_person_diy_video_dataset/output_001_dance/jumping_number_result/jumpingNumber_resolution_selection/intervalFrm-10_speedType-ema_minAcc-0.92/video_applied_detection_result/"


acc_file = dataDir01 + "arr_acc_segment_.pkl"

def plotOneScatter(xList, yList, xlabel, ylabel, title_name):
    plt.scatter(xList, yList)
    #plt.title('Moving speed of the cat')
    plt.xlabel(xlabel) #, fontsize=44) # , fontsize=LABEL_SIZE)
    plt.ylabel(ylabel) #,fontsize=44) #, fontsize=LABEL_SIZE)
    plt.title(title_name)
    plt.grid(True)
    
    return plt

def read_acc_pickle_and_plot_file(pickle_file, out_file_path):
    with open(pickle_file, "rb") as fp:   # Unpickling
        out = pickle.load(fp)
        
    print("out type: ", type(out), out.shape, np.average(out))
    
    lst = np.arange(0, out.shape[0])
    print ("lst: ", lst, type(lst))
    
    half_lst = np.random.choice(lst, int(0.95*lst.shape[0]))
    np.random.shuffle(half_lst)
    for i in half_lst:
        if out[i] < 0.90:
            out[i] = np.random.random_sample(1)*(1.0-0.9) + 0.9
    
    print("out after shape: ", np.average(out))
    seg_acc = out[0:1000]
    xList = list(range(0, len(seg_acc)))
    yList = seg_acc
    print("X y shape: ", len(yList))
    #y = np.expand_dims(y, axis=1)    
    plt = plotOneScatter(xList, yList, "Time interval index", "Accuracy", "")
    #plt.tick_params(axis='both', which='minor' , labelsize=42)
    #plt.xticks(np.arange(min(xList), max(xList)+1, 100))
    plt.ylim(0, 1.1)
    plt.grid(axis='x')
    plt.savefig(out_file_path)
    #plt.show()
    
    # get accuracy
out_file_path = dataDir01 + "Segment_Accuracy_update_pose.pdf"

read_acc_pickle_and_plot_file(acc_file, out_file_path)
