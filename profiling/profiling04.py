#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:07:08 2019

@author: fubao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:53:39 2019

@author: fubao
"""

# profiling  04
# consider another pose estimation model cpn  https://github.com/chenyilun95/tf-cpn
# modify the code about the profiling to read frame by frame not config by config
# read every frame of pose estimation with all its positions, and then write into each file, then read the next frame again and again
# so that for a long video, we can get part of video frames' results, no need to wait the whole video finish if necessary.

# every other 1 minute to get 
# for frame rate less than maximum frame rate, use current extracted frame to replace the rest of frames unextracted



# get a profiling result and store into csv with dataframe
import sys 
import os 
import re
import cv2

import pandas as pd
import numpy as np

from glob import glob
from collections import defaultdict
from common_prof import *

#sys.path.insert(0, '../openPoseEstimation')

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

#from poseEstimation.pytorch_Realtime_Multi_Person_Pose_Estimation.openPose_interface import openPose_estimation_one_image

from tensorflow.python.client import device_lib

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


from poseEstimation.tf_pose_estimation.tf_openPose_interface import tf_open_pose_inference
from poseEstimation.tf_pose_estimation.tf_openPose_interface import load_openPose_model


from poseEstimation.tf_cpn.models.COCO_res50_384x288_CPN import tf_cpn_interface_res50_384  # import load_objectdetection_model
from poseEstimation.tf_cpn.models.COCO_res50_256x192_CPN import tf_cpn_interface_res50_256  # import load_cpn_pose_estimation_model


'''
modify the profiling frame by frame not config by config
'''


# define a class including each clip's profile result
#class cls_profile_video(object):
    # cameraNo is the camera no. for multiple camera streaming. 
    # queryNo is the query type, currently human pose estimation
    # 'frameStartNo', 
#    __slots__ = ['cameraNo', 'queryNo', 'resolution', 'frameRate','modelMethod', 'accuracy', 'costFPS']


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def profiling_Video_MaxFrameRate_OpenPose(inputDir, outDir):
    '''
    profiling frame by frame first of openPose models
    2 models with 5 resolutions, 6 frame rate
    output profiling result:
    'Configuration_C_index' + '\t' + 'Resolution' +'\t' + 'Frame_rate' +'\t' + 'Model' +   \
             '\t' + 'Image_path' +'\t' + 'Estimation_result' + '\t' + 'numberOfHumans' + '\t'+ 'Time_SPF' + '\n'  # '\t' + 'image_Id'+ '\t' + 'category_Id' +'\t' + 'detection_keyPoint' + '\t' + 'scores' + '\n'
        
    Where Estimation_result is the coco format with detection result.
    '''
    model_openPose_dict = defaultdict()         # open pose model 
    
    configIndex = 1
    #create the file first each file is a config of  the video
    for mod in modelMethods_openPose:
        for res in resoStrLst_OpenPose:    # resolutions
            fr = frameRates[0]        # use only maximum frame rate is enough
            #'resolution', 'frameRate','modelMethod',
            # each file is a config of  the video
            out_file = outDir + str(res)+ '_' + str(fr) + '_' + str(mod) + '_estimation_result.tsv'
                
            headStr = 'Configuration_C_index' + '\t' + 'Resolution' +'\t' + 'Frame_rate' +'\t' + 'Model' +   \
             '\t' + 'Image_path' +'\t' + 'Estimation_result' + '\t' + 'numberOfHumans' + '\t'+ 'Time_SPF' + '\n'  # '\t' + 'image_Id'+ '\t' + 'category_Id' +'\t' + 'detection_keyPoint' + '\t' + 'scores' + '\n'
                     
            configIndex += 1
            # load model
            e, w, h = load_openPose_model(mod, res)
            model_openPose_dict[str(mod) + '_' + str(res)] = (e, w, h)
            
            with open(out_file, 'w') as f:  # write the head for each file
                f.write(headStr)
                


    img_arr_dict = defaultdict()          # load into memory first
    frmCnt = 1                              # frame count
    #print ("inputDir: ", inputDir)
    filePathLst = sorted(glob(inputDir + "*.jpg"))        # test only [:2]
    #print ("filePathLst: ", filePathLst[0])
    for imgPath in filePathLst:      # iterate each subdirect's frames and compose a video    # profiling frame by frame
        
    
        #create the file first each file is a config of  the video
        for res in resoStrLst_OpenPose:    # resolutions
            fr = frameRates[0]        # use only maximum frame rate is enough
            for mod in modelMethods_openPose:
                # get the file name 
                out_file = outDir + str(res)+ '_' + str(fr) + '_' + str(mod) + '_estimation_result.tsv'
                with open(out_file, 'a') as f:
                    #get pose estimation model
                    paramTriple = model_openPose_dict[str(mod) + '_' + str(res)]
                    e = paramTriple[0]
                    w = paramTriple[1]
                    h = paramTriple[2]
                    
                    key_img = imgPath + str(res)
                    #print ('imgPath: ', imgPath)
                    if key_img in img_arr_dict:     # estimate human poses from a single image !
                        img_arr = img_arr_dict[key_img]
                    
                    elif verify_bad_image(imgPath):
                        print ('verify_image: error', imgPath)
                        img_arr_dict[key_img] = None
                        human_no = 0
                        humans_poses = ""
                        elapsedTime = 0
                    else:
                        img_arr = read_imgfile(imgPath, w, h)
                        if img_arr is None:
                            print ('Image can not be read, path=%s')
                            img_arr_dict[key_img] = None
                            human_no = 0
                            humans_poses = ""  
                            elapsedTime = 0
                        else:
                            img_arr_dict[key_img] = img_arr
                    
                    #print ('img_arr_dict[key_img]: ', img_arr_dict[key_img].shape)
                    if img_arr_dict[key_img] is not None:
                        # estimate pose of this frame  call the open_pose etc. interface
                        #output_lst, elapsedTime = openPose_estimation_one_image(imgPath, res) #mulitple person
                        humans_pose_lst, elapsedTime = tf_open_pose_inference(img_arr, e, w, h)
                        #print ("xxxx humans_pose_lst: ",res, fr, mod,  humans_pose_lst, imgPath)
                        
                        if humans_pose_lst == [] or humans_pose_lst is None:
                            human_no = 0
                            humans_poses = ""
                        else:
                            human_no = len(humans_pose_lst)
                            humans_poses = ";".join(human for human in humans_pose_lst)
    
                    if frmCnt % (1000) == 0:        # every other 4s to print only
                        print ("profiling_Video_MaxFrameRate_OpenPose 00 frames finished inference result: ", imgPath, mod, res, elapsedTime)
                    
                    #save into file
                    #if humans_poses is None or human_no == 0:
                    #    continue
                    
                    preWriteStr = str(configIndex) + '\t' + str(res) +'\t' + str(fr) +'\t' + str(mod) + '\t' + \
                        imgPath + '\t' + str(humans_poses)  + '\t'  + str(human_no) + '\t' + str(elapsedTime) + '\n'      
                    f.write(preWriteStr)
        
        frmCnt += 1
                    


def profilingOneVideoMaxFrameRateFrameByFrame_CPN(inputDir, outDir):
    '''
    profiling frame by frame first with CPN models
    two resolutions available only,  1 model, use 6 frame_rates later
    '''    
    model_cpn_pose_dict = defaultdict()          # cpn pose  
    configIndex = 1
    #create the file first each file is a config of  the video
    for mod in modelMethods_cpn:
            
        MODEL_NAME = "../poseEstimation/tf_cpn/object_detection_models/object_detection_model_ssd_mobilenet_v1_fpn/"   # object_detection_model_rcfn/"
        object_detection_graph, category_index = tf_cpn_interface_res50_384.load_objectdetection_model(MODEL_NAME)
            
        # loading weights are the same
        #if res == "384x288":
        test_cpn_pose_model_path = "../poseEstimation/tf_cpn/models/COCO_res50_384x288_CPN/model_graph/snapshot_350.ckpt"
        tester_cpn = tf_cpn_interface_res50_384.load_cpn_pose_estimation_model(test_cpn_pose_model_path)                
        #elif res == "256x192":
        #object_detection_graph, category_index = tf_cpn_interface_res50_256.load_objectdetection_model(MODEL_NAME)          # no need to reload here
        #test_cpn_pose_model_path = "../poseEstimation/tf_cpn/models/COCO_res50_256x192_CPN/model_graph/snapshot_350.ckpt"
        #tester_cpn = tf_cpn_interface_res50_256.load_cpn_pose_estimation_model(test_cpn_pose_model_path)   
            
        for res in resoStrLst_cpn:    # resolutions
            fr = frameRates[0]        # use only maximum frame rate is enough
            #'resolution', 'frameRate','modelMethod',
            # each file is a config of  the video
            out_file = outDir + str(res)+ '_' + str(fr) + '_' + str(mod) + '_estimation_result.tsv'
                
            headStr = 'Configuration_C_index' + '\t' + 'Resolution' +'\t' + 'Frame_rate' +'\t' + 'Model' +   \
             '\t' + 'Image_path' +'\t' + 'Estimation_result' + '\t' + 'numberOfHumans' + '\t'+ 'Time_SPF' + '\n'  # '\t' + 'image_Id'+ '\t' + 'category_Id' +'\t' + 'detection_keyPoint' + '\t' + 'scores' + '\n'
                     
            configIndex += 1

            model_cpn_pose_dict[mod + '_' + str(res)] = (object_detection_graph, category_index, tester_cpn)
            
            with open(out_file, 'w') as f:  # write the head for each file
                f.write(headStr)
    
    
    img_arr_dict = defaultdict()          # load into memory first
    frmCnt = 1        # fram step size
                
    filePathLst = sorted(glob(inputDir + "*.jpg"));      # test only [:2]
    #print ("filePathLst: ", filePathLst)
    for imgPath in filePathLst:      # iterate each subdirect's frames and compose a video    # profiling frame by frame
        #create the file first each file is a config of  the video
        for res in resoStrLst_cpn:    # resolutions
            fr = frameRates[0]        # use only maximum frame rate is enough
            for mod in modelMethods_cpn:
                # get the file name 
                out_file = outDir + str(res)+ '_' + str(fr) + '_' + str(mod) + '_estimation_result.tsv'
                with open(out_file, 'a') as f:
                    #get pose estimation model
                    paramTriple = model_cpn_pose_dict[mod + '_' + str(res)]
                    object_detection_graph = paramTriple[0]
                    category_index = paramTriple[1]
                    tester_cpn = paramTriple[2]
                    
                    w, h = map(int, res.split('x'))
                      
                    
                    key_img = imgPath + str(res)
                    if key_img in img_arr_dict:     # estimate human poses from a single image !
                        img_arr = img_arr_dict[key_img]
                        
                    elif verify_bad_image(imgPath):
                        print ('verify_image: error2', imgPath)
                        img_arr_dict[key_img] = None
                        human_no = 0
                        humans_poses = ""
                        elapsedTime = 0
                    else:
                        img_arr = cv2.imread(imgPath, cv2.IMREAD_COLOR)
                        if img_arr is None:
                            print ('Image can not be read2, path=%s')
                            img_arr_dict[key_img] = None
                            human_no = 0
                            humans_poses = ""   
                            elapsedTime = 0
                        else:
                            img_arr_dict[key_img] = img_arr
                    
                    if img_arr_dict[key_img] is not None:
                        
                        # call cpn model to detect pose for all the humans in the image
                        if res == "384x288":
                            humans_pose_lst, elapsedTime = tf_cpn_interface_res50_384.tf_cpn_inference_pose(img_arr, (w, h), object_detection_graph, category_index, tester_cpn)
                        
                        elif res == "256x192":
                            humans_pose_lst, elapsedTime = tf_cpn_interface_res50_256.tf_cpn_inference_pose(img_arr, (w, h), object_detection_graph, category_index, tester_cpn)
                        
                        if humans_pose_lst == [] or humans_pose_lst is None:
                            human_no = 0
                            humans_poses = ""
                        else:
                            human_no = len(humans_pose_lst)
                            humans_poses = ";".join(human for human in humans_pose_lst)               
                            
    
                    if frmCnt % (1000) == 0:        # every other 4s to print only
                        print ("profilingOneVideoMaxFrameRateFrameByFrame_CPN 00 frames finished inference result: ", imgPath, mod, res, elapsedTime)
                    
                    #save into file
                    preWriteStr = str(configIndex) + '\t' + str(res) +'\t' + str(fr) +'\t' + str(mod) + '\t' + \
                        imgPath + '\t' + str(humans_poses)  + '\t'  + str(human_no) + '\t' + str(elapsedTime) + '\n'      
                    f.write(preWriteStr)
        
        frmCnt += 1




    
def execute_profiling(segment_time):
    
    print ("gpus devices: ", device_lib.list_local_devices())
    get_available_gpus()
    
    lst_input_video_frms_dir = ['001-dancing-10mins_frames/', '002-soccer-20mins_frames/', \
                        '003-bike_race-20mins_frames/', '004-Marathon-20mins_frames/',  \
                        '006-cardio_condition-20mins_frames/', '008-Marathon-20mins_frames/', \
                        '009-Marathon-20mins_frames/', '010-baseball_static-20mins_frames/']
    
    
    for input_frm_dir in lst_input_video_frms_dir[7:8]:   # [1:2]:      # [4:5][0:1]:         #  # [5:6]:       # run 006 first
        input_dir = dataDir2 + input_frm_dir
        
        out_dir = dataDir2 + 'output_' + '_'.join(input_frm_dir.split('_')[:-1]) +'/'      # 004-output_Marathon-20mins_01/' 
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        profiling_Video_MaxFrameRate_OpenPose(input_dir, out_dir)
        profilingOneVideoMaxFrameRateFrameByFrame_CPN(input_dir, out_dir)
        

def execute_profiling_one_person():
    
    print ("gpus devices: ", device_lib.list_local_devices())
    get_available_gpus()
    
    lst_input_video_frms_dir = ['001_dance_frames/', '002_dance_frames/', \
                        '003_dance_frames/', '004_dance_frames/',  \
                        '005_dance_frames/', '006_yoga_frames/', \
                        '007_yoga_frames/', '008_cardio_frames/',
                        '009_cardio_frames/', '010_cardio_frames/']
    

    for input_frm_dir in lst_input_video_frms_dir[0:1]:   # [1:2]:      # [4:5][0:1]:         #  # [5:6]:       # run 006 first
        input_dir = dataDir3 + input_frm_dir
        
        out_dir = dataDir3 + 'output_' + '_'.join(input_frm_dir.split('_')[:-1]) +'/'      # 004-output_Marathon-20mins_01/' 
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        profiling_Video_MaxFrameRate_OpenPose(input_dir, out_dir)
        profilingOneVideoMaxFrameRateFrameByFrame_CPN(input_dir, out_dir)
        

    
if __name__== "__main__":
    
    #execute_profiling()
    execute_profiling_one_person()