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
import pandas as pd
import numpy as np

from glob import glob
from collections import defaultdict
from common_prof import computeOKSAP

#sys.path.insert(0, '../openPoseEstimation')

current_file_cur = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, current_file_cur + '/..')

#from poseEstimation.pytorch_Realtime_Multi_Person_Pose_Estimation.openPose_interface import openPose_estimation_one_image

from tensorflow.python.client import device_lib

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


from poseEstimation.tf_pose_estimation.tf_openPose_interface import tf_open_pose_inference
from poseEstimation.tf_pose_estimation.tf_openPose_interface import load_openPose_model


from poseEstimation.tf_cpn.models.COCO_res50_384x288_CPN import tf_cpn_interface_res50_384  # import load_objectdetection_model
from poseEstimation.tf_cpn.models.COCO_res50_256x192_CPN import tf_cpn_interface_res50_256  # import load_cpn_pose_estimation_model


dataDir1 = '../input_output/mpii_dataset/'
dataDir2 = '../input_output/diy_video_dataset/'

PLAYOUT_RATE = 25


'''
modify the profiling frame by frame not config by config
'''

# define a class including each clip's profile result
#class cls_profile_video(object):
    # cameraNo is the camera no. for multiple camera streaming. 
    # queryNo is the query type, currently human pose estimation
    # 'frameStartNo', 
#    __slots__ = ['cameraNo', 'queryNo', 'resolution', 'frameRate','modelMethod', 'accuracy', 'costFPS']


frameRates = [25, 15, 10, 5, 2, 1]    # test only [25, 10, 5, 2, 1]   # [5]   #          #  [25]    #  [25, 10, 5, 2, 1]    # [30],  [30, 10, 5, 2, 1] 
resoStrLst_OpenPose = ["1120x832", "960x720", "640x480",  "480x352", "320x240"]   # for openPose models [720, 600, 480, 360, 240]   # [240] #     # [240]       # [720, 600, 480, 360, 240]    #   [720]     # [720, 600, 480, 360, 240]  #  [720]    # [720, 600, 480, 360, 240]            #  16: 9
resoStrLst_cpn = ["384x288", "256x192"]   # for cpn models, only two resolutions pretrained available

modelMethods_openPose = ['cmu', 'mobilenet_v2_small']
# a_cpn,   "a" is just to make it alphabetically order first, to make it as ground truth conviniently for programming
modelMethods_cpn = ['a_cpn']  #  'cmu']   # , 'mobilenet_v2_small'] # ['a_cpn']   #     ['a_cpn', 'cmu', 'mobilenet_v2_small']  #  ['mobilenet_v2_small']      # ['mobilenet_thin']  # ['cmu']  #  ["openPose"]


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

    frmCnt = 1                              # frame count
                
    filePathLst = sorted(glob(inputDir + "*.jpg"))        # test only [:2]
    #print ("filePathLst: ", filePathLst)
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
                        
                    # estimate pose of this frame  call the open_pose etc. interface
                    #output_lst, elapsedTime = openPose_estimation_one_image(imgPath, res) #mulitple person
                    humans_pose_lst, elapsedTime = tf_open_pose_inference(imgPath, e, w, h)
                
                    human_no = len(humans_pose_lst)
                    
                    humans_poses = ";".join(human for human in humans_pose_lst)

                    if frmCnt % (1) == 0:        # every other 4s to print only
                        print ("profilingOneVideoWithMaxFrameRate 00 frames finished inference result: ", imgPath, mod, res, elapsedTime)
                    #save into file
                    if humans_poses is None or human_no == 0:
                        continue
                    
                    preWriteStr = str(configIndex) + '\t' + str(res) +'\t' + str(fr) +'\t' + str(mod) + '\t' + \
                        imgPath + '\t' + str(humans_poses)  + '\t'  + str(human_no) + '\t' + str(elapsedTime) + '\n'      
                    f.write(preWriteStr)
        
        frmCnt += 1
                    


def profilingOneVideoMaxFrameRateFrameByFrame_CPN(inputDir, outDir):
    '''
    profiling frame by frame first with CPN models
    two resolutions available only,  1 models, use 6 frame rate later
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
                      
                    # call cpn model to detect pose for all the humans in the image
                    if res == "384x288":
                        humans_poses_array, elapsedTime = tf_cpn_interface_res50_384.tf_cpn_inference_pose(imgPath, (w, h), object_detection_graph, category_index, tester_cpn)
                    
                    elif res == "256x192":
                        humans_poses_array, elapsedTime = tf_cpn_interface_res50_256.tf_cpn_inference_pose(imgPath, (w, h), object_detection_graph, category_index, tester_cpn)
                    
                    human_no = len(humans_poses_array)
                        
                    humans_poses = ";".join(human for human in humans_poses_array)
                        

                    if frmCnt % (1) == 0:        # every other 4s to print only
                        print ("profilingOneVideoWithMaxFrameRate 00 frames finished inference result: ", imgPath, mod, res, elapsedTime)
                    #save into file
                    if humans_poses is None or human_no == 0:
                        continue
                    
                    preWriteStr = str(configIndex) + '\t' + str(res) +'\t' + str(fr) +'\t' + str(mod) + '\t' + \
                        imgPath + '\t' + str(humans_poses)  + '\t'  + str(human_no) + '\t' + str(elapsedTime) + '\n'      
                    f.write(preWriteStr)
        
        frmCnt += 1

def getEachSegmentProfilingAPTime(inputDir, segment_time,  outDir):
    '''
    get each frames's avearge precision(AP) and detection time
    
    read each file's detection result of each config, compare with the most expensive configuration and
    calculate the AP and time

    outProfFile:
        [resolution', 'frameRate','modelMethod', 'segment_no', 'costFPS', 'accuracy']

    '''
    
    segment_frames = PLAYOUT_RATE*segment_time  # 25*4 = 100   
    profilingTime = segment_time//4       
    profiling_frames = int(PLAYOUT_RATE*profilingTime)    # frames in the profiling time
    

    gtDic = defaultdict(list)        # ground truth detection file for each frame
    
    
    out_file = outDir + 'test_profiling_segment_time' + str(segment_time)+ '.tsv'
    
    headStr = 'Resolution' +'\t' + 'Frame_rate' +'\t' + 'Model' +   \
             '\t' + 'Segment_no' +'\t' + 'Detection_speed_FPS'  + '\t' + 'Acc' +'\n'   # '\t' + 'image_Id'+ '\t' + 'category_Id' +'\t' + 'detection_keyPoint' + '\t' + 'scores' + '\n'
             
        
    print ("inputDir: ", inputDir)
    filePathLst = sorted(glob(inputDir + "*.tsv"))  # must read ground truth file(the most expensive config) first
    with open(out_file, 'w') as f:
        f.write(headStr)
        
        for fileCnt, filePath in enumerate(filePathLst):
           # read poste estimation detection result file
           
            df_det = pd.read_csv(filePath, delimiter='\t', index_col=False)         # det-> detection
            print ("filePath: ", fileCnt, filePath, df_det.columns)
                       

            framesTotalNum = df_det.shape[0]
    
            segment_no = 1
            startFrmCnt = 1
            endFrmCnt = segment_frames
    
            #config_index = df_det.iat[0, 0]
            resolution = df_det.iat[1, 1]
            #print ("resolution: ", type(resolution), resolution)
            img_w = int(resolution.split('x')[0])
            img_h = int(resolution.split('x')[1])

            model = df_det.iat[0, 3]
                            
            if fileCnt == 0:      # 384x288_a_cpn or '1120x832_25_cmu' in fileName:  # this file has the most expensive config result
                # get the ground truth for each image_path
                # pose estimation result is ground truth, accuracy is 1, add a new column "Acc"
                # get ground truth dictionary with estimation result
                gtDic = dict(zip(df_det.Image_path, df_det.Estimation_result))
                #df_det['Acc'] = 1             
                gt_w = img_w             # ground truth image's width
                gt_h = img_h            # ground truth image's heigth 
            while (startFrmCnt < framesTotalNum):

                for frmRate in frameRates:
                    profileFrmInter = int(PLAYOUT_RATE//frmRate)          # frame rate sampling frames in interval
                    
                    #if frmRate == frameRates[0]:     # PLAYOUT_RATE  # ground truth
                    #    gtDic = dict(zip(df_det.Image_path, df_det.Estimation_result))     # this should be cpn model as ground truth
                        #df_det['Acc'] = 1 
                        #print ("profileFrmInter: ",profiling_frames, profileFrmInter, type( df_prof.iloc[startFrmCnt:profiling_frames]))
                        #get det_speed in the profiling time
                    
                    # no else: also calculate the segment's accuracy for ground truth scenario
                    #print ("det_speed_seg: ", df_det.iloc[startFrmCnt:startFrmCnt+profiling_frames].iloc[::profileFrmInter, 7].astype(float).values)
                    #calculate  detection speed for this segment based on current frame rate
                    det_speed_seg= sum(df_det.iloc[startFrmCnt:startFrmCnt+profiling_frames].iloc[::profileFrmInter, 7].astype(float).values)/PLAYOUT_RATE
               
                    det_speed_seg = round(1/det_speed_seg, 2)
                    # accuracy use the extracted frame estimation result for the result of frame, tha a little complicated,
                    #df_extracted_det_est = df_prof.iloc[startFrmCnt:startFrmCnt+profiling_frames].iloc[::profileFrmInter, 4, 5]
                    df_extracted_det_est = df_det.iloc[startFrmCnt:startFrmCnt+profiling_frames]   # Image_path and Estimation_result
                    
                    iterFrmRateIndx = 0
                    acc_seg = 0.0
                    for indx, row in df_extracted_det_est.iterrows():
                        #print ("getEachSegmentAPTime row: ", indx, type(row), row['Estimation_result'])
                        #get the estimation_result for this frame
                        img = row['Image_path']
                        gt_result_curr_frm = gtDic[img]
                        
                        if img not in gtDic:
                            acc_seg += 0
                            print ("img not found in ground truth")
                            continue
            
                        if iterFrmRateIndx % profileFrmInter == 0:
                            lead_frame_est_res  = row['Estimation_result']
                            #get accuracy
                            acc_curr_frm = computeOKSAP(lead_frame_est_res, gt_result_curr_frm, img, img_w, img_h, gt_w, gt_h)
                            acc_seg += acc_curr_frm
                        else:
                            curr_frm_est_res = lead_frame_est_res
                            acc_curr_frm = computeOKSAP(curr_frm_est_res, gt_result_curr_frm, img, img_w, img_h, gt_w, gt_h)
                            acc_seg += acc_curr_frm
                            
                        iterFrmRateIndx += 1
                    acc_seg /= profiling_frames
                    acc_seg = round(acc_seg, 3)
                    #print ("iterFrmRateIndx: ", iterFrmRateIndx)
                    #break       # test only
                
                    rowStr =  str(resolution) + '\t' + str(frmRate) + '\t' + str(model) +   \
                        '\t'  + str(segment_no) + '\t' + str(det_speed_seg) + '\t' + str(acc_seg) + '\n'
                    f.write(rowStr)
                    
                startFrmCnt = endFrmCnt
                endFrmCnt += segment_frames
                if endFrmCnt > framesTotalNum:   # not enough
                    endFrmCnt = framesTotalNum
                    
                segment_no += 1
                


    df_prof_segment = pd.read_csv(out_file, delimiter='\t', index_col=False)  # det-> detection

    #df_prof_segment['Config_index'] = df_prof_segment.index
    
    # get each segment and store into file
    
    #df_prof_segment.to_csv(out_file, sep='\t', index=False)


    for seg_no in range(1, segment_no):
        out_seg_no_file = outDir + 'profiling_segmentTime' + str(segment_time)+ '_segNo' + str(seg_no) + '.tsv'
        df_seg_no = df_prof_segment.loc[df_prof_segment['Segment_no'] == seg_no]
        print ("df_seg_no shape: ", df_seg_no.shape, type(df_seg_no))
        df_seg_no.insert(0, column="Config_index", value = range(1, len(df_seg_no) + 1))
        df_seg_no.to_csv(out_seg_no_file, sep='\t', index=False)
        


    
def execute_profiling(segment_time):
    
    print ("gpus devices: ", device_lib.list_local_devices())
    get_available_gpus()
    
    lst_input_video_frms_dir = ['001-dancing_10mins_frames/', '002-soccer-20mins-frames/', \
                        '003-bike_race-20mins_frames/', '004-Marathon-20mins_frames/',  \
                        '006-cardio_condition-20mins_frames/', ]
    
    for input_frm_dir in lst_input_video_frms_dir[4::]:       # run 006 first
        input_dir = dataDir2 + input_frm_dir
        
        out_dir = dataDir2 + 'output_' + '_'.join(input_frm_dir.split('_')[:-1]) +'/'      # 004-output_Marathon-20mins_01/' 
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        #profilingOneVideoMaxFrameRateFrameByFrame_CPN(input_dir, out_dir)
        #profiling_Video_MaxFrameRate_OpenPose(input_dir, out_dir)
        
        
        #transfer to accuracy and detection speed in each segment
        '''
        input_dir = out_dir
        out_dir = input_dir + 'profiling_result/' 
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        getEachSegmentProfilingAPTime(input_dir, segment_time, out_dir)
        '''
    
        
  
    
if __name__== "__main__":
    
    segment_time = 4
    execute_profiling(segment_time)
    