import os
import pandas as pd
import copy
import numpy as np
import copy
import operator
import math
import time
import sys
import csv

import numpy as np
import pickle

from blist import blist

from glob import glob

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur)

from common_prof import dataDir3
from common_prof import PLAYOUT_RATE
from common_prof import frameRates
from common_prof import computeOKSAP


# actually online simulation
'''
simulate with a threshold of lag
and chek the accuracy can achieve?
different lag thresholds?
finally to get a plot with different average accuracy with different lags
Q1:  online simulation, does it keep within the lag in until the end of each segment?   
if not, we allow some fluctuation, why not set previous threshold higher? and  what's the point of threshold?
'''



def getEachFrameProfilingAPTtime(inputDir,  outDir):
    '''
    #get each config, each frame's accuracy and detection speed/frame
    in total each config corresponds a file
    '''
      
    filePathLst = sorted(glob(inputDir + "*estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    for fileCnt, filePath in enumerate(filePathLst):
       # read poste estimation detection result file
       
        df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)         # det-> detection
        print ("getEachSegmentProfilingAPTime filePath: ", fileCnt, filePath, df_det.columns)
                   
        
        #config_index = df_det.iat[0, 0]
        resolution = df_det.iat[1, 1]
        sample_rate = df_det.iat[1, 2]
        model = df_det.iat[1, 3]
        #print ("resolution: ", type(resolution), resolution)
        img_w = int(resolution.split('x')[0])
        img_h = int(resolution.split('x')[1])
            
        if fileCnt == 0:      # 384x288_a_cpn or '1120x832_25_cmu' in fileName:  # this file has the most expensive config result
            # get the ground truth for each image_path
            # pose estimation result is ground truth, accuracy is 1, add a new column "Acc"
            # get ground truth dictionary with estimation result
            gtDic = dict(zip(df_det.Image_path, df_det.Estimation_result))
            #df_det['Acc'] = 1             
            #gt_w = img_w             # ground truth image's width
            #gt_h = img_h            # ground truth image's heigth 
            #continue                 # ground truth not need to calculate accuracy? or still use it as "1"
        # get the accuracy of each frame
        #acc = 0.0
        list_acc_cur = []
        for indx, row in df_det.iterrows():
            #print ("getEachSegmentAPTime row: ", indx, type(row), row['Estimation_result'])
            #get the estimation_result for this frame
            img = row['Image_path']
                    
            if img not in gtDic:
                acc = 0
                print ("getEachSegmentProfilingAPTime img not found in ground truth", img)
                list_acc_cur.append(acc)
                continue
            gt_result_curr_frm = gtDic[img]
               
            curr_frm_est_res  = row['Estimation_result']
            #print ("getEachSegmentProfilingAPTime curr_frm_est_res", curr_frm_est_res)
            #acc = computeOKSAP(curr_frm_est_res, gt_result_curr_frm, img, img_w, img_h, gt_w, gt_h)
            if str(curr_frm_est_res) == 'nan':
                list_acc_cur.append(acc)
                continue
            acc = computeOKSAP(curr_frm_est_res, gt_result_curr_frm, img)
            list_acc_cur.append(acc)
        
        #print ("acc: ", len(list_acc_cur), len(df_det), df_det.count(),  df_det.shape)
        
        
        df_det['Acc'] = list_acc_cur
        
        # write into file
        del df_det['Estimation_result']         # delete the column file
        del df_det['numberOfHumans']         # delete the column file
        out_file = outDir + str(resolution) + '_' + str(sample_rate) + '_' + str(model) + '_frame_result.tsv'
        df_det.to_csv(out_file, sep='\t', index=False)
         


def execute_frame_performance():
    '''
    after profiling
    calculate each frame's performance accuracy, detection speed for each config
    
    '''
    #lst_input_video_frms_dir = ['001-dancing_10mins_frames/', '002-soccer-20mins-frames/', \
    #                    '003-bike_race-20mins_frames/', '004-Marathon-20mins_frames/',  \
    #                    '006-cardio_condition-20mins_frames/', ]
    
    lst_input_video_frms_dir = ['output_001_dance/', 'output_002_dance/', \
                    'output_003_dance/', 'output_004_dance/',  \
                    'output_005_dance/', 'output_006_yoga/', \
                    'output_007_yoga/', 'output_008_cardio/', \
                    'output_009_cardio/', 'output_010_cardio/']
      
    
    for input_frm_dir in lst_input_video_frms_dir[4:5]:  # [4::]:       # run 006 first
        
        #transfer to accuracy and detection speed in each segment
        inputDir = dataDir3 + input_frm_dir
        out_dir = inputDir + 'frames_config_result/' 
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        getEachFrameProfilingAPTtime(inputDir, out_dir)

def modelToInt(model):
    if model == 'cmu':
        return 0
    elif model == 'a':
        return 2
    else: 
        return 1
    
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
    
 
def read_config_name_from_file(data_frame_dir):
    '''
    read config info and order based on resolution*frame rate and then order them in descending order
    and make it a dictionary
    '''
    #print ("bbbb read_config_name_from_file: ", data_frame_dir)
    config_lst = blist()
    # get config_id
    
    filePathLst = sorted(glob(data_frame_dir + "*frame_result*.tsv"))  # must read ground truth file(the most expensive config) first
    #resoFrmRate = blist()
    for fileCnt, filePath in enumerate(filePathLst):
        # get the resolution, frame rate, model
        # input_output/diy_video_dataset/output_006-cardio_condition-20mins/frames_config_result/1120x832_25_cmu_frame_result.tsv
        filename = filePath.split('/')[-1]
        #print ("aaaaa filename: ", filename)
        reso = filename.split('_')[0]
        #res_right = reso.split('x')[1]
        #frm_rate = filename.split('_')[1]
        
        model = filename.split('_')[2]
        
        #720 720 25 mobilenet
        #res_frame_multiply = int(res_right) * int(frm_rate)    # order of the resolution
        
        #print ("reso: ", reso)
        
        config_lst += getNewconfig(reso, model)     # add more configs
        
        #resoFrmRate.append(res_frame_multiply)  random.sort(key=lambda e: e[1])

        
    #model_resoFrm_dict = dict(zip(config_lst, resoFrmRate))
    #sort by resolution*frame_rate  e.g. 720px25
    config_lst.sort(key = lambda ele: int(ele.split('-')[0].split('x')[1])* int(ele.split('-')[1]), reverse=True)
    config_id_dict = dict(zip(config_lst,range(0, len(config_lst))))
        
    id_config_dict = dict(zip(range(0, len(config_lst)), config_lst))

    print ("model_resoFrm_dict aaa: ", len(config_id_dict), len(id_config_dict))
    
    '''
    pickle_dir = dataDir2 + 'output_006-cardio_condition-20mins/' + 'pickle_files/'
    with open(pickle_dir + 'config_to_id.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Config_name", "Id_in_order"])
        for key, value in config_id_dict.items():
            writer.writerow([key, value])
    '''

    
    return config_id_dict, id_config_dict



def read_config_name_resolution_frm_rate(data_pose_keypoint_dir, write_flag):
    '''
    read config info and order based on resolution and then order them in descending order
    and make it a dictionary
    
    read config info and order based on frame_rate and then order them in descending order
    and make it a dictionary
    
    '''
    
    config_lst = blist()
    # get config_id
    
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*frame_result*.tsv"))  # must read ground truth file(the most expensive config) first
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
        config_tuple_lst.append((reso, frm_rate, modelToInt(model)))
    
    #print ("config_tuple_lst: ", config_tuple_lst)

    config_tuple_lst.sort(key = lambda ele: ele[0], reverse=True)
    
    config_tuple_id_dict = dict(zip(config_tuple_lst,range(0, len(config_tuple_lst))))

    id_config_tuple_dict = dict(zip(range(0, len(config_tuple_lst)), config_tuple_lst))

    #print ("config_tuple_dict: ", len(config_tuple_lst),  len(config_tuple_id_dict), len(id_config_tuple_dict))

    if write_flag:
        pickle_dir = data_pose_keypoint_dir 
        with open(pickle_dir + 'config_tuple_to_id.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Config_name", "Id_in_order"])
            for key, value in config_tuple_id_dict.items():
                writer.writerow([key, value])
         
    return config_tuple_id_dict, id_config_tuple_dict



def read_config_name_resolution_only(data_pose_keypoint_dir, write_flag):
    '''
    read config info and order based on resolution and then order them in descending order
    and make it a dictionary
    
    read config info and order based on frame_rate and then order them in descending order
    and make it a dictionary
    
    '''
    
    config_lst = set()
    # get config_id
    
    
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*frame_result*.tsv"))  # must read ground truth file(the most expensive config) first
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


                
                
        
def getconfigSPFAcc(reso, model, time_spf, acc, config_id_dict, frm_id, confg_frm_spf_arr, confg_frm_acc_arr):
    '''
    get new config's tiem_spf for the current frame ,  because the input is for PLAYOUT rate 25
    
    '''
    for frmRt in frameRates:
        frmInter = math.ceil(PLAYOUT_RATE/frmRt)          # frame rate sampling frames in interval, +1 every other
        
        #new_spf = time_spf/frmInter
        
        config = reso + '-' + str(frmRt) + '-' + model.split('_')[0]
        #config = (int(reso.split('x')[1]), int(frmRt), modelToInt(model.split('_')[0]))   #
        
        #config = int(reso.split('x')[1])   #
        
        cfg_id = config_id_dict[config]
        
        # confg_frm_spf_arr[cfg_id, frm_id-1] = new_spf
        confg_frm_acc_arr[cfg_id, frm_id-1] = acc
        #print ("cfg_id: ", cfg_id)
    #print ("confg_frm_acc_arr 2222: ",  confg_frm_acc_arr)
    return confg_frm_acc_arr


def read_frame_result_config_numpy(data_frame_dir, out_frm_spf_pickle_file, out_frm_acc_pickle_file):
    '''
    read frame profiling result into numpy array
    
    
    config_id:  reso --frame -- model
    matrix 1: accuracy value
                image_id
    config_id     0.9
    
    matrix 2: detection  computation cost value  SPF
                imag_id
    config_id   0.05 s
    
    '''
    
    config_id_dict, id_config_dict = read_config_name_from_file(data_frame_dir)
    
    #config_id_dict, id_config_dict = read_config_name_resolution_frm_rate(data_frame_dir, False)
    #config_id_dict, id_config_dict = read_config_name_resolution_only(data_frame_dir, False)

    #config_lst = blist()
    # 1120x832_25_cmu_frame_result
    filePathLst = sorted(glob(data_frame_dir + "*frame_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    config_num = len(config_id_dict)  # len(config_id_dict)
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection
    frame_num = len(df_det)+7000     # because maybe some frame_id is missing
    #create a numpy array
    confg_frm_spf_arr = np.zeros((config_num, frame_num)) # array of time_spf with config vs frame_Id
    confg_frm_acc_arr = np.zeros((config_num, frame_num)) # array of acc with config vs frame_Id
    
    print ("config_id_dict: ", config_id_dict,  len(config_id_dict), frame_num)
    
    for fileCnt, filePath in enumerate(filePathLst):
        df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)         # det-> detection
        

        print ("numy shape: ", confg_frm_spf_arr.shape, filePath)
        
        for index, row in df_det.iterrows():  
            #print ("index, row: ", index, row)
            reso = row['Resolution']
            #frm_rate = row['Frame_rate']
            model = row['Model']
            time_spf = row['Time_SPF']
            acc = row['Acc']
            frm_id = int(row['Image_path'].split('/')[-1].split('.')[0])
            
            #config_spf, config_acc = getconfigSPFAcc(reso, model, time_spf, acc)
            
            #print ("config_lst: ", config_spf)
            #print ("id: ", config_id_dict[config_lst[1]])
            #confg_frm_spf_arr[fileCnt*len(frameRates):fileCnt*len(frameRates)+len(frameRates), frm_id-1] = config_spf
           
            #confg_frm_acc_arr[fileCnt*len(frameRates):fileCnt*len(frameRates)+len(frameRates), frm_id-1] = config_acc
            confg_frm_acc_arr = getconfigSPFAcc(reso, model, time_spf, acc, config_id_dict, frm_id, confg_frm_spf_arr, confg_frm_acc_arr)
            
            #print ("confg_frm_spf_arr: ", confg_frm_spf_arr, frm_id-1)
            
            #if index == 1:
            #    break   # debug only
                
        #if fileCnt == 1:
        #    break   # debug only

    print ("confg_frm_spf_arr: ", confg_frm_spf_arr.shape)
    
    with open(out_frm_spf_pickle_file,'wb') as fs:
        pickle.dump(confg_frm_spf_arr, fs)
        
    with open(out_frm_acc_pickle_file,'wb') as fa:
        pickle.dump(confg_frm_acc_arr, fa)
        

if __name__== "__main__":
    execute_frame_performance()
    