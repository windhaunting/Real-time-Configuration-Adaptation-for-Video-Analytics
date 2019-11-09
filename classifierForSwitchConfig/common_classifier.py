#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:12:00 2019

@author: fubao
"""

# common file for classification of switching config

import re
import sys
import os
import csv
import cv2

import numpy as np
from collections import OrderedDict
from collections import defaultdict
from glob import glob
from blist import blist
from matplotlib import pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.preprocessing import OneHotEncoder
from common_plot import plotTwoLinesOneFigure
from common_plot import plotTwoSubplots

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import dataDir3
from profiling.common_prof import frameRates

COCO_KP_NUM = 17      # total 17 keypoints


'''
0	nose
1	leftEye
2	rightEye
3	leftEar
4	rightEar
5	leftShoulder
6	rightShoulder
7	leftElbow
8	rightElbow
9	leftWrist
10	rightWrist
11	leftHip
12	rightHip
13	leftKnee
14	rightKnee
15	leftAnkle
16	rightAnkle

'''



def drawHuman(npimg, est_arr):
    '''
    draw the skeleton according to the estimation point
    
    # only one person here
    '''
    dict_part_id = {'nose' : 0, 'leftEye': 1, 'rightEye': 2, 'leftEar': 3, 'rightEar': 4, 'leftShoulder': 5,
                 'rightShoulder': 6, 'leftElbow' : 7, 'rightElbow': 8, 'leftWrist': 9, 
                 'rightWrist': 10, 'leftHip': 11, 'rightHip':12, 'leftKnee': 13, 'rightKnee': 14,
                 'leftAnkle': 15, 'rightAnkle': 16}
    
    pairs_parts = [('leftShoulder', 'leftElbow'), ('leftElbow', 'leftWrist'), ('leftShoulder', 'leftHip'),
                  ('leftHip','leftKnee'),  ('leftKnee', 'leftAnkle'), ('leftShoulder', 'rightShoulder'),
                  ('leftHip','rightHip'), ('rightShoulder', 'rightHip'), ('rightHip', 'rightKnee'),
                  ('rightKnee','rightAnkle'), ('rightShoulder', 'rightElbow'), ('rightElbow', 'rightWrist')]
    
    image_h, image_w = npimg.shape[:2]

    kp_arr = getPersonEstimation(est_arr)

    centers = [None]*COCO_KP_NUM


    # draw point
    point_num = kp_arr.shape[0]
    for i in range(0, point_num):
        body_part = kp_arr[i]
        center = (int(body_part[0] * image_w + 0.5), int(body_part[1] * image_h + 0.5))
        
        #print ("center: ", center, body_part[0])
        centers[i] = center
        cv2.circle(npimg, center, 3, (0, 255, 255), thickness=3, lineType=8, shift=0)
        cv2.putText(npimg, str(i), center, 2, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    #print ("kp_arr: ", kp_arr.shape, )
    
    #dict_id_part = {v:k for k, v in dict_part_id.items()}
    for pr in pairs_parts:
        
        #print ("pr: ", pr, dict_part_id)
        pt1 = centers[dict_part_id[pr[0]]]
        pt2 = centers[dict_part_id[pr[1]]]
        cv2.line(npimg, pt1, pt2, (0, 255, 255), 3)

    return npimg
    
def getPersonEstimation(est_res):
    '''
    analyze the personNo's pose estimation result with highest confidence score
    input: 
        [500, 220, 2, 514, 214, 2, 498, 210, 2, 538, 232, 2, 0, 0, 0, 562, 308, 2, 470, 304, 2, 614, 362, 2, 420, 362, 2, 674, 398, 2, 372, 394, 2, 568, 468, 2, 506, 468, 2, 596, 594, 2, 438, 554, 2, 616, 696, 2, 472, 658, 2],1.4246317148208618;
        [974, 168, 2, 988, 162, 2, 968, 158, 2, 1004, 180, 2, 0, 0, 0, 1026, 244, 2, 928, 250, 2, 1072, 310, 2, 882, 302, 2, 1112, 360, 2, 810, 346, 2, 1016, 398, 2, 948, 396, 2, 1064, 518, 2, 876, 482, 2, 1060, 630, 2, 900, 594, 2],1.4541109800338745;
        [6, 172, 2, 16, 164, 2, 6, 162, 2, 48, 182, 2, 0, 0, 0, 68, 256, 2, 10, 254, 2, 112, 312, 2, 0, 0, 0, 168, 328, 2, 0, 0, 0, 70, 412, 2, 28, 420, 2, 108, 600, 2, 0, 0, 0, 144, 692, 2, 0, 0, 0],1.283275842666626;
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 162, 2, 0, 0, 0, 92, 208, 2, 42, 206, 2, 134, 250, 2, 0, 0, 0, 176, 284, 2, 0, 0, 0, 118, 334, 2, 86, 334, 2, 126, 396, 2, 80, 396, 2, 148, 548, 2, 0, 0, 0],0.9429177641868591
        
        
    output:
        a specific person's detection result arr (17x2)
    '''
    
    #print ("est_res: ", est_res)
    strLst = re.findall(r'],\d.\d+', est_res)
    person_score = [re.findall(r'\d.\d+', st) for st in strLst]
    
    personNo = np.argmax(person_score)
    
    
    est_res = est_res.split(';')[personNo]
    
    lst_points = [[float(t[0]), float(t[1]), float(t[2])] for t in re.findall(r'(0(?:\.\d*)?), (0(?:\.\d*)?), (\d\.[0]|[0123])', est_res)]

    kp_arr = np.array(lst_points)
    
    #print ("kp_arr: ", kp_arr.shape, kp_arr)
    return kp_arr

    
def load_data_all_features(data_examples_dir, xfile, yfile):
    '''
    data the data for traing and test with all the features, 
    without feature selection
    '''
    
    x_input_arr = np.load(data_examples_dir + xfile)
    y_out_arr = np.load(data_examples_dir + yfile).astype(int)
    
    print ("y_out_arr:",x_input_arr.shape, y_out_arr.shape)

    #y_out_arr = y_out_arr.reshape(-1, 1)
    #print ("y_out_arr before:", np.unique(y_out_arr),  y_out_arr.shape, y_out_arr[:2])

    # reshape to 2-dimen for input
    x_input_arr = x_input_arr.reshape((x_input_arr.shape[0], -1))
    
    #output config to one hot encoder for classification
    #onehot_encoder = OneHotEncoder(sparse=False)
    #y_out_arr = onehot_encoder.fit_transform(y_out_arr)
    
    print ("y_out_arr after:", y_out_arr.shape)
    #return 
    return x_input_arr, y_out_arr          # debug only 1000 first


def parseToResolution(cls_id, id_config_dict):
    #
    
    cls_id = int(cls_id)
    config = id_config_dict[cls_id]
    
    reso = int(config.split("-")[0].split("x")[1])
    
    return reso
    
def checkCorrelationPlot(data_pose_keypoint_dir, input_x_arr, out_y_arr, id_config_dict):
    '''
    visualize correlation
    '''
    outDir = data_pose_keypoint_dir + "classifier_result/"
    
    pdf_pages = PdfPages( outDir + 'correlation_vis_all_config.pdf')


    print ("out_y_arr: ", input_x_arr.shape, out_y_arr.shape)
    out_y_arr = out_y_arr.reshape((-1, 1))
    #output y class transfer to resolution
    reso_out_y_arr = np.apply_along_axis(parseToResolution, 1, out_y_arr, id_config_dict)
    
    print ("reso_out_y_arr: ", reso_out_y_arr)
    instance_num = input_x_arr.shape[0]
    feature_num = input_x_arr.shape[1]
    
    
    xList = range(0, instance_num)
    for i in range(0, feature_num):
        
        yList1 = input_x_arr[:, i]
        yList2 =  reso_out_y_arr
    # Create a figure instance (ie. a new page)
        xlabel = 'Instance number'
        ylabel1 = 'Feature value'
        ylabel2 = 'Ouput_class(Resolution)'
        title_name = ''
        #fig = plotTwoLinesOneFigure(xList, yList1, yList2, xlabel, ylabel, title_name)
        
        fig = plotTwoSubplots(xList, yList1, yList2, xlabel, ylabel1, ylabel2, title_name)
        # Plot whatever you wish to plot
     
        # Done with the page
        pdf_pages.savefig(fig)
    pdf_pages.close()
    

    
    
def feature_selection(data_examples_dir, history_frame_num, max_frame_example_used):
    '''
    transfer input output into a csv and run it in weka 
    '''                 
        
    x_input_arr = np.load(data_examples_dir + "X_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl")
    y_out_arr = np.load(data_examples_dir + "Y_data_features_config-history-frms" + str(history_frame_num) + "-sampleNum" + str(max_frame_example_used) + ".pkl")
    #x_input_arr = x_input_arr.reshape((x_input_arr.shape[0], x_input_arr.shape[1]*x_input_arr.shape[2]))
    y_out_arr = y_out_arr.reshape((y_out_arr.shape[0], 1))

    print ("x_input_arr y_out_arr shape:",x_input_arr.shape, y_out_arr.shape)
    
    data_example_arr= np.hstack((x_input_arr, y_out_arr))
    header_lst = ','.join([str(i) for i in range(0, x_input_arr.shape[1])]) + ',Config'
    np.savetxt( data_examples_dir + "data_example_history_frms1.csv", data_example_arr, delimiter=",", header = header_lst)
    
 
def get_cmu_model_config_acc_spf(data_pickle_dir, data_pose_keypoint_dir):
        intervalFlag = 'sec'
        acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir, intervalFlag)
                
        #old_acc_frm_arr = acc_frame_arr
        #print ("getOnePersonFeatureInputOutput01 acc_frame_arr: ", acc_frame_arr.shape)
    
        # save to file to have a look
        #outDir = data_pose_keypoint_dir + "classifier_result/"
        #np.savetxt(outDir + "accuracy_above_threshold" + str(minAccuracy) + ".tsv", acc_frame_arr[:, :5000], delimiter="\t")
        
        
        #config_ind_pareto = getParetoBoundary(acc_frame_arr[:, 0], spf_frame_arr[:, 0])
        #resolution_set = ["1120x832", "960x720", "640x480",  "480x352", "320x240"]   # for openPose models [720, 600, 480, 360, 240]   # [240] #     # [240]       # [720, 600, 480, 360, 240]    #   [720]     # [720, 600, 480, 360, 240]  #  [720]    # [720, 600, 480, 360, 240]            #  16: 9
        resolution_set = ["1120x832", "960x720", "640x480",  "480x352", "320x240"]  #["640x480"]  #["1120x832"]  #, "960x720", "640x480",  "480x352", "320x240"]   # for openPose models [720, 600, 480, 360, 240]   # [240] #     # [240]       # [720, 600, 480, 360, 240]    #   [720]     # [720, 600, 480, 360, 240]  #  [720]    # [720, 600, 480, 360, 240]            #  16: 9
        frame_set = [25, 15, 10, 5, 2, 1]     #  [25, 10, 5, 2, 1]    # [30],  [30, 10, 5, 2, 1] 
        model_set = ['cmu']   #, 'mobilenet_v2_small']
    
        lst_id_subconfig, id_config_dict = extract_specific_config_name_from_file(data_pose_keypoint_dir, resolution_set, frame_set, model_set)
    
        print ("getOnePersonFeatureInputOutput01 lst_id_subconfig: ", lst_id_subconfig)
        acc_frame_arr = acc_frame_arr[lst_id_subconfig, :]
        spf_frame_arr =  spf_frame_arr[lst_id_subconfig, :]
    
        id_config_dict = OrderedDict(sorted(id_config_dict.items(), key=lambda t: t[0]))

        return acc_frame_arr, spf_frame_arr, id_config_dict

def getParetoBoundary(acc_arr, spf_arr):
    #acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)
    
    
    # given a config of accuracy and spf
    num_confg = acc_arr.shape[0]
    
    config_ind_pareto = []
    print ("acc_shape: ", num_confg)
    for i in range(0, num_confg):
        found_flag = True
        for j in range(0, num_confg):
            if i == j:
                continue
            if acc_arr[j] > acc_arr[i] and spf_arr[j] < spf_arr[i]:
                found_flag = False
                break
            
        if found_flag:
            config_ind_pareto.append(i)
        
    return config_ind_pareto
    
def getNewconfig(reso, model):
    '''
    get new config from all available frames
    the config file name has only frame rate 25;  1 frame do not have frame rate 
    so we get more models from frames
    '''
    config_lst = blist()
    for frmRt in frameRates:
        config_lst.append(reso + '-' + str(frmRt) + '-' + model)
        
    return config_lst


def read_all_config_name_from_file(data_pose_keypoint_dir, write_flag):
    '''
    read config info and order based on resolution*frame rate and then order them in descending order
    and make it a dictionary
    '''
    config_lst = blist()
    # get config_id
    
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*result*.tsv"))  # must read ground truth file(the most expensive config) first
    #resoFrmRate = blist()
    for fileCnt, filePath in enumerate(filePathLst):
        #if '1120x832' in filePath and 'cmu' in filePath:        # neglect the most expensive config as ground truth for caluclating accuracy and resource cost
        #    continue
        # get the resolution, frame rate, model
        # input_output/diy_video_dataset/output_006-cardio_condition-20mins/frames_config_result/1120x832_25_cmu_frame_result.tsv
        filename = filePath.split('/')[-1]
        #print ("filename: ", filename)
        reso = filename.split('_')[0]
        #res_right = reso.split('x')[1]
        #frm_rate = filename.split('_')[1]
        
        model = filename.split('_')[2]
                
        #print ("reso: ", reso)
        
        config_lst += getNewconfig(reso, model)     # add more configs
        
        #resoFrmRate.append(res_frame_multiply)  random.sort(key=lambda e: e[1])

        
    #model_resoFrm_dict = dict(zip(config_lst, resoFrmRate))
    #sort by resolution*frame_rate  e.g. 720px25
    config_lst.sort(key = lambda ele: int(ele.split('-')[0].split('x')[1])* int(ele.split('-')[1]), reverse=True)
    config_id_dict = OrderedDict(zip(config_lst,range(0, len(config_lst))))
        
    id_config_dict = {v:k for k, v in config_id_dict.items()} #    error dict(zip(range(0, len(config_lst)), config_lst))

    #print ("model_resoFrm_dict: ", id_config_dict, len(id_config_dict), config_id_dict)
    
    if write_flag:
        pickle_dir = data_pose_keypoint_dir 
        with open(pickle_dir + 'config_to_id.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Config_name", "Id_in_order"])
            for key, value in config_id_dict.items():
                writer.writerow([key, value])
    
    return config_id_dict, id_config_dict



def extract_specific_config_name_from_file(data_pose_keypoint_dir, resolution_set, frame_set, model_set):
    '''
    extract the specific config from a certain resolution, frame_rate, and/or model
    
    resolution_lst, frame_lst,  model_lst
    
    '''
    write_flag = False
    config_id_dict, id_config_dict = read_all_config_name_from_file(data_pose_keypoint_dir, write_flag)
    lst_id_subconfig = []     # include the original id    
    for conf, idx in config_id_dict.items():
        
        reso = conf.split("-")[0]
        frmRt = int(conf.split("-")[1])
        model = conf.split("-")[2]
        
        if reso in resolution_set and frmRt in frame_set and model in model_set:
            lst_id_subconfig.append(idx)
        
    #print ("lst_id_subconfig: ", lst_id_subconfig, config_id_dict)
    
    
    # get new id map based on pareto boundary/'s result
    new_id_config_dict = defaultdict()
    for i, ind in enumerate(lst_id_subconfig):
        new_id_config_dict[i] = id_config_dict[ind]
    id_config_dict = new_id_config_dict
    
    #print ("config_id_dict: ", len(id_config_dict), id_config_dict)
    
    return lst_id_subconfig, id_config_dict



def read_poseEst_conf_frm(data_pickle_dir):
    '''
    read profiling conf's pose of each frame from pickle 
    the pickle file is created from the file "writeIntoPickleConfigFrameAccSPFPoseEst.py"
    
    '''
    
    confg_est_frm_arr = np.load(data_pickle_dir + 'config_estimation_frm.pkl', allow_pickle=True)
    #acc_seg_arr = np.load(data_pickle_dir + file_lst[2])
    #spf_seg_arr = np.load(data_pickle_dir + file_lst[3])
    
    print ("confg_est_frm_arr ", type(confg_est_frm_arr), confg_est_frm_arr.shape)
    
    return confg_est_frm_arr


def readProfilingResultNumpy(data_pickle_dir, intervalFlag):
    '''
    read profiling from pickle
    the pickle file is created from the file "writeIntoPickle.py"
    
    '''
    if intervalFlag == 'frame':
        acc_frame_arr = np.load(data_pickle_dir + 'config_acc_frm.pkl', allow_pickle=True)          # frame by frame but it also calculate the 1sec interval with each frame starting
    elif intervalFlag == 'sec':
        #acc_frame_arr = np.load(data_pickle_dir + 'config_acc_interval_1sec.pkl')
        acc_frame_arr = np.load(data_pickle_dir + 'config_oks_interval_1sec.pkl', allow_pickle=True)
#        
    spf_frame_arr = np.load(data_pickle_dir + 'config_spf_frm.pkl', allow_pickle=True)
    #acc_seg_arr = np.load(data_pickle_dir + file_lst[2])
    #spf_seg_arr = np.load(data_pickle_dir + file_lst[3])
    
    #print ("acc_frame_arr ", type(acc_frame_arr), acc_frame_arr.shape)
    
    return acc_frame_arr, spf_frame_arr


def read_config_name_resolution_frmRate(data_pose_keypoint_dir, write_flag):
    '''
    read config info and order based on resolution and then order them in descending order
    and make it a dictionary
    
    read config info and order based on frame_rate and then order them in descending order
    and make it a dictionary
    
    '''
    
    config_lst = blist()
    # get config_id
    
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    #resoFrmRate = blist()
    for fileCnt, filePath in enumerate(filePathLst):
        #if '1120x832' in filePath and 'cmu' in filePath:        # neglect the most expensive config as ground truth for caluclating accuracy and resource cost
        #    continue
        # get the resolution, frame rate, model
        # input_output/diy_video_dataset/output_006-cardio_condition-20mins/frames_config_result/1120x832_25_cmu_frame_result.tsv
        filename = filePath.split('/')[-1]
        #print ("filename: ", filename)
        reso = filename.split('_')[0]
        #res_right = reso.split('x')[1]
        #frm_rate = filename.split('_')[1]
        
        model = filename.split('_')[2]
                
        #print ("reso: ", reso)
        
        config_lst += getNewconfig(reso, model)     # add more configs
        
    
    config_tuple_lst = blist()
    for config in config_lst:
        reso = int(config.split('-')[0].split('x')[1])
        frm_rate = int(config.split('-')[1])
        model = config.split('-')[2]

        config_tuple_lst.append((reso, frm_rate, model))
    
    print ("config_tuple_lst: ", config_tuple_lst)

    config_tuple_lst.sort(key = lambda ele: ele[0], reverse=True)
    
    config_tuple_id_dict = dict(zip(config_tuple_lst,range(0, len(config_tuple_lst))))

    id_config_tuple_dict = dict(zip(range(0, len(config_tuple_lst)), config_tuple_lst))

    print ("config_tuple_dict: ", id_config_tuple_dict)

    if write_flag:
        pickle_dir = data_pose_keypoint_dir 
        with open(pickle_dir + 'config_tuple_to_id.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Config_name", "Id_in_order"])
            for key, value in config_tuple_id_dict.items():
                writer.writerow([key, value])



def read_config_name_resolution_only(data_pose_keypoint_dir, write_flag):
    '''
    read config info and order based on resolution and then order them in descending order
    and make it a dictionary
    
    read config info and order based on frame_rate and then order them in descending order
    and make it a dictionary
    
    '''
    
    config_lst = set()
    # get config_id
    
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    #resoFrmRate = blist()
    for fileCnt, filePath in enumerate(filePathLst):
        #if '1120x832' in filePath and 'cmu' in filePath:        # neglect the most expensive config as ground truth for caluclating accuracy and resource cost
        #    continue
        # get the resolution, frame rate, model
        # input_output/diy_video_dataset/output_006-cardio_condition-20mins/frames_config_result/1120x832_25_cmu_frame_result.tsv
        filename = filePath.split('/')[-1]
        #print ("filename: ", filename)
        reso = filename.split('_')[0]
        #res_right = reso.split('x')[1]
        #frm_rate = filename.split('_')[1]
        
        model = filename.split('_')[2]
                
        #print ("reso: ", reso)
        
        #config_lst += getNewconfig(reso, model)     # add more configs
        config_lst.add((int(reso.split('x')[1])))
    
   
    print ("config_lst: ", config_lst)

    config_lst = list(config_lst)
    config_lst.sort(reverse=True)
    
    config_reso_id_dict = dict(zip(config_lst,range(0, len(config_lst))))

    id_config_reso_dict = dict(zip(range(0, len(config_lst)), config_lst))

    print ("config_reso_id_dict: ", config_reso_id_dict)

    if write_flag:
        pickle_dir = data_pose_keypoint_dir 
        with open(pickle_dir + 'config_to_id_resoOnly.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Config_name", "Id_in_order"])
            for key, value in config_reso_id_dict.items():
                writer.writerow([key, value])
                
    return config_reso_id_dict, id_config_reso_dict


def calculateDifferenceSumFrmRate(y_test, y_pred, id_config_dict):
    '''
    get the predicted config abs difference sum of frame rate overall
    #2nd get the difference of predicted false
    '''
    # get the frame rate list
    lst1 = [int(id_config_dict[int(x)].split("-")[1]) for x in y_test]       # frame rate
    lst2 = [int(id_config_dict[int(x)].split("-")[1]) for x in y_pred]
    
    overall_diff = [abs(lst1[i]-lst2[i]) for i in range(0, len(lst1))]
    print ("calculateDifferenceSumFrmRate: overall_diff ", len(overall_diff))
    
    overall_diffSum = sum(overall_diff)/len(overall_diff)
    
    lst1_sub = []   # the instance that predicted wrong
    lst2_sub = []
    for i in range(0, len(lst1)):
        if lst1[i] != lst2[i]:
            lst1_sub.append(lst1[i])
            lst2_sub.append(lst2[i])
       
        
    sub_diff = [abs(lst1_sub[i]-lst2_sub[i]) for i in range(0, len(lst1_sub))]

    sub_diffSum = sum(sub_diff)/len(sub_diff)

    return overall_diffSum, sub_diffSum


def paddingZeroToInter(ind):
    '''
    padding to 6digits at most,  1 -> 000001, 10->000010
    
    '''
    
    if ind < 10:
        ind_str = '00000' + str(ind)
    elif ind < 100:
        ind_str = '0000' + str(ind)
    elif ind < 1000:
        ind_str = '000' + str(ind)
    elif ind < 10000:
        ind_str = '00' + str(ind)
    elif ind < 100000:
        ind_str = '0' + str(ind)
    else:    
        ind_str = str(ind)
        
    return ind_str

if __name__== "__main__": 

    #data_pose_keypoint_dir =  dataDir2 + 'output_006-cardio_condition-20mins/'

    #read_config_name_from_file(data_pose_keypoint_dir, True)
    #read_config_name_resolution_frmRate(data_pose_keypoint_dir, True) 
    #read_config_name_resolution_only(data_pose_keypoint_dir, True)
    
    
    # feature selection
    #video_dir_lst = ['output_001-dancing-10mins/', 'output_006-cardio_condition-20mins/', 'output_008-Marathon-20mins/']   
    
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/']
    
    for video_dir in video_dir_lst[0:7]:  #[0:1]: 
        history_frame_num = 1  #1          # 
        max_frame_example_used = 10000 # 8000 # 20000 #8025   # 8000
        
        data_examples_dir =  dataDir3 + video_dir + 'data_examples_files/'
        #data_examples_dir =  dataDir2 + 'output_006-cardio_condition-20mins/' + 'data_examples_files_resoFR_tuple/'
        #data_examples_dir =  dataDir2 + 'output_006-cardio_condition-20mins/' + 'data_examples_files_resolutionOnly/'
        
        feature_selection(data_examples_dir, history_frame_num, max_frame_example_used)