#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:50:44 2020

@author: fubao
"""

# video_process common file
# read video and frames, draw the pose estimation on the frame and video for display


import sys
import glob
import os
import cv2
import time
import skvideo.io

import numpy as np
from blist import blist
from data_file_process import read_pickle_data

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from profiling.common_prof import dataDir3


def drawHuman(npimg, kp_arr):
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


    centers = [None]*len(dict_part_id)


    # draw points
    point_num = kp_arr.shape[0]
    for i in range(0, point_num):
        body_part = kp_arr[i]
        center = (int(body_part[0] * image_w + 0.5), int(body_part[1] * image_h + 0.5))
        
        #print ("center: ", center, body_part[0])
        centers[i] = center
        cv2.circle(npimg, center, 3, (0, 204, 255), thickness=5, lineType=8, shift=0)
        cv2.putText(npimg, str(i), center, 2, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    #print ("kp_arr: ", kp_arr.shape, )
    
    #dict_id_part = {v:k for k, v in dict_part_id.items()}
    for pr in pairs_parts:
        
        #print ("pr: ", pr, dict_part_id)
        pt1 = centers[dict_part_id[pr[0]]]
        pt2 = centers[dict_part_id[pr[1]]]
        cv2.line(npimg, pt1, pt2, (0, 204, 255), 8)

    return npimg


def read_frame_jpg_file_dir(input_dir):
    # read frame file from directory
    #input: directory
    # ouput: array list
    frame_jpg_list = []
    for root, dirnames, filenames in os.walk(input_dir):
        for filename in sorted(filenames):
            if filename.endswith(('jpg')):
                frame_jpg_list.append(os.path.join(root, filename))
    return frame_jpg_list

def draw_video_frames(pose_est_frm, frame_jpg_list, out_data_dir):
    # read a series of frames and draw the pose estimation on the video
    # input: a video's keypoint pose estimation result and the video's frame directory
    # output the vi
    
    curr_start_frm_path = frame_jpg_list[0]
    isFile = os.path.isfile(curr_start_frm_path) 
    #print("isFile  isFile: ", curr_start_frm_path)
    assert(isFile == True)       # make sure it is a frame
    
    #print("curr_start_frm_path  shape: ", curr_start_frm_path)
    try:
        im = cv2.imread(curr_start_frm_path, cv2.IMREAD_COLOR)
    except:
        print ("img read failure" )
        raise
    #print("im  shape: ", im.shape)
    height, width, layers = im.shape
    video_name = out_data_dir + 'out_pose_video.mp4'   
    
    #print("draw_video_frames video_name here: ", video_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #('m','p','4','v') #  Be sure to use lower case
    out_video = cv2.VideoWriter(video_name, fourcc, 25.0, (width, height))
    
    #out_video =  np.empty([126, height, width, 3], dtype = np.uint8)
    
    frm_indx = 0
    frame_num = pose_est_frm.shape[0]
    
    while(frm_indx < 2500):  #part of video clips for test; frame_num):  # frame_num):
        curr_frm_path = frame_jpg_list[frm_indx]
        # print("curr_frm_path: ", curr_frm_path)
        isFile = os.path.isfile(curr_start_frm_path) 
        assert(isFile == True)       # make sure it is a frame
        
        try:
            im = cv2.imread(curr_frm_path, cv2.IMREAD_COLOR)
        except:
            print ("img read failure" )
            raise
            
        npimg = drawHuman(im, pose_est_frm[frm_indx])
        #print("img_name npimg : ", frm_indx, pose_est_frm[frm_indx])
        frm_indx += 1
        
        cv2.imwrite(out_data_dir + str(frm_indx) + ".jpg", npimg)
        out_video.write(npimg)
        
        #out_video[frm_indx] = npimg
        
    #print("draw_video_frames end here: ", video_name)
    #skvideo.io.vwrite("test.mp4", out_video)

    out_video.release()
    cv2.destroyAllWindows()

def read_pose_estimation(data_file):

    pose_est_frm = read_pickle_data(data_file)
        
    #print("pose shape: ", pose_est_frm[1])
    #y = np.expand_dims(y, axis=1)    
    return pose_est_frm  


def execute_draw_video():
    
    video_dir_lst = ['output_001_dance/', 'output_002_dance/', \
                        'output_003_dance/', 'output_004_dance/',  \
                        'output_005_dance/', 'output_006_yoga/', \
                        'output_007_yoga/', 'output_008_cardio/', \
                        'output_009_cardio/', 'output_010_cardio/']
            
    for video_dir in video_dir_lst[4:5]:    # [2:3]:   #[1:2]:  # [1:2]:  #[0:1]:        #[1:2]:
        
        data_pickle_dir = dataDir3 + video_dir + 'jumping_number_result/'
        subDir = data_pickle_dir + "intervalFrm-1_speedType-ema_minAcc-0.95/"
        
        # read pose estimation result
        data_file =  subDir + "pose_est_frm.pkl"
        frm_input_dir = dataDir3 + "005_dance_frames_bakup/"
        
        # from training data
        pose_est_detected_frm = read_pose_estimation(data_file)
        
        # get image jpg path
        frame_jpg_list = read_frame_jpg_file_dir(frm_input_dir)
        
        draw_video_frames(pose_est_detected_frm, frame_jpg_list, subDir)
        

def video_to_frame(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length, input_loc, output_loc)
    count = 0
    print ("Converting video..\n",  cap.isOpened())
    # Start converting the video
    
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break


def extract_video_frames(input_dir):
    
    lst_input_video_loc = sorted(glob.glob(input_dir + '*.mp4'))
    for input_video_loc in lst_input_video_loc:
        #input_video_loc = '/'.join(input_dir.split('/')[:-1]) + '/' + 'car_traffic_01.mp4'
        output_loc = '/'.join(input_dir.split('/')[:-1]) + '/' + input_video_loc.split('/')[-1].split('.')[0] + '_frames'
        video_to_frame(input_video_loc, output_loc)
        
if __name__== "__main__": 
    
    execute_draw_video()
    
    