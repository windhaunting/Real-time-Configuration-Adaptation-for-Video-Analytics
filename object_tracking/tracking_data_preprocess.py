#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:34:54 2021

@author: fubao
"""
import pickle
import glob
import subprocess
import json
import cv2
import time
import os
import csv

import numpy as np
from collections import defaultdict

# video or file preprocess; process video and decode into different resolution with FFMPEG

#  "/media/fubao/TOSHIBAEXT/research_bakup/data_video_analytics/input_output/vechicle_tracking/proceesing_videos/"    # "../input_output/vechicle_tracking/"   # /var/fubao/videoAnalytics_poseEstimation/input_output/vechicle_tracking "/media/fubao/TOSHIBAEXT/research_bakup/data_video_analytics/input_output/vechicle_tracking/"


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
        

def video_resize_process(input_video_dir):
    # resize_video
    
    video_path_files = sorted(glob.glob(input_video_dir + '*.mp4'))

    print ("video_path_files: ", video_path_files)

    reso_list = ["1120x832", "960x720", "640x480",  "480x352", "320x240"] 
    
    for video_input in video_path_files:
        
        video_name = video_input.split('/')[-1].split('.')[0]    # without extension
        
        for reso in reso_list:
            video_output =  input_video_dir + video_name + "_" + reso  + ".mp4"
            
            #print ("video_output: ", video_output)
            command = "ffmpeg -i {video_input} -s {reso} -c:a copy {video_output}".format(video_input=video_input, reso=reso, video_output=video_output)
            subprocess.call(command,shell=True)
                        
    # ffmpeg -i car_traffic_08.mp4 -s 480x352 -c:a copy car_traffic_08_480x352.mp4

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def write_pickle_data(data_instance, out_pickle_file):
    
    with open(out_pickle_file, "wb") as fp:   #Pickling
        pickle.dump(data_instance, fp)


def read_pickle_data(pickle_file):
    with open(pickle_file, "rb") as fp:   # Unpickling
        out = pickle.load(fp)
        
    return out

def write_numpy_into_file(arr, out_file_path):
    
    with open(out_file_path, 'wb') as f:
        np.save(f, arr)
    
    

def read_numpy(npy_file):
    arr = np.load(npy_file)
    return arr


def write_lst_to_csv(columnName_lst, val_lst, fileName):
    # write into csv file;  one column assumed
    with open(fileName, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(columnName_lst)
        for val in val_lst:
            wr.writerow([val])
    
def read_json_dir(one_video_input_dir):
    # read json to a dictionary or numpy
    # output: dictionary out of json
    
    json_file_list = sorted(glob.glob(one_video_input_dir + '*.json'))
    print("json_file_list: ", one_video_input_dir, json_file_list)
    
    dict_detection_reso_video = defaultdict()   # multiple resolution
    
    for json_file in json_file_list:
        
        #print ("json_file :", json_file)
        #detection_res_json = json.loads(json_file)
        reso = json_file.split("/")[-1].split("_")[-1].split(".")[0]
        
        with open(json_file, 'r') as j:
            dict_detection_res_frames_ = json.loads(j.read())
        
        #print ("dict_detection_res_frames_: ", dict_detection_res_frames_['1']['spf'])
        
        dict_detection_reso_video[reso] = dict_detection_res_frames_
        
    #print("dict_detection_reso_video: ", dict_detection_reso_video.keys())

    return dict_detection_reso_video
    


if __name__ == '__main__':
    
    #input_video_dir = "/var/fubao/videoAnalytics_objectTracking/input_output/vehicle_tracking/"
    
    input_video_dir = "/var/fubao/videoAnalytics_objectTracking/input_output/object_tracking/sample_video_out/sample_video_json_03/"
    #video_resize_process(input_video_dir)
    
    extract_video_frames(input_video_dir)