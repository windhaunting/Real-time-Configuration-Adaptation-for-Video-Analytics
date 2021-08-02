#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:32:35 2020

@author: fubao
"""


import numpy as np
import glob
import json
import pickle
import os
import cv2
import time
# speaker detection analysis

# read speaker bounding box and put into numpy



PLAYOUT_RATE = 25
COCO_KP_NUM = 17
input_dir = "../input_output/speaker_video_dataset/sample_video_out/"


resoStrLst = ["1120x832", "960x720", "640x480",  "480x352", "320x240"]   # 0 1 2 3 4

# bounding_box: [y1, x1, y2, x2]  vertical height is y, horiztonal width is x

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
    print ("Converting video..\n", cap.isOpened())
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
        

def extract_video_frames(input_video_loc):
    #input_video_loc = '/'.join(input_dir.split('/')[:-1]) + '/' + file_name # 'sample_05.mp4'
    output_loc = '/'.join(input_video_loc.split('/')[:-1]) + '/' + input_video_loc.split('/')[-1].split('.')[0] + '_frames'
    video_to_frame(input_video_loc, output_loc)


def read_pickle_data(pickle_file):
    with open(pickle_file, "rb") as fp:   # Unpickling
        out = pickle.load(fp)
        
    return out

def write_pickle_data(data_instance, out_pickle_file):
    
    with open(out_pickle_file, "wb") as fp:   #Pickling
        pickle.dump(data_instance, fp)
        
        
def normalize_bounding_box(resolution, box):
    
    # bounding_box: [x1, y1, x2, y2]  vertical height is x, horiztonal width is y
    width = int(resolution.split("x")[0])
    height = int(resolution.split("x")[1])
    
    yA = box[0]/height        # width
    xA = box[1]/width         # height
    yB = box[2]/height
    xB = box[3]/width
    
    return [yA, xA, yB, xB]
    

def get_accuracy_time_py(data_dir):
    # read the pickle into numpy
    # input: data pickle all 5 resolution pickle
    # output: detection bounding_box numpy,  resolution number x frame number x 4,  value is bounding box
            # time numpy, resolution number x frame number, value is time

    speaker_box_arr = [0] * len(resoStrLst)   # 0
    
    
    spf_arr = [0] * len(resoStrLst)      # spf each frame detection time
    
    #print ("get_accuracy_time_py data_dir: ", data_dir)
    for json_file in glob.glob(data_dir + "*.json"):

        file_name = json_file.split("/")[-1]
        
        #print ("get_accuracy_time_py file_name: ", file_name)
        for i, reso in enumerate(resoStrLst):
            if reso in file_name:
                #print ("pickle_lst: ",i, json_file)                
                
                # read the json file: format: "168": {"time(s)": 0.2048, "person": [[125, 255, 196, 300]], "screen": [[13, 5, 185, 165]]}, "169": {"time(s)": 0.2022, "person": [[127, 255, 192, 300]], "screen": [[13, 5, 185, 165]]}, "170": {"time(s)": 0.2038, "person": [[128, 254, 196, 300]], "screen": [[13, 5, 185
                                                # , 165]]}, "171": {"time(s)": 0.2031, "person": [[128, 257, 193, 300]], "screen": [[13, 5, 185, 165]]},

                with open(json_file) as f:
                    detect_res = json.load(f)
                    frame_len = len(detect_res)
                    
                    
                    #print("ddd: ", frame_len, detect_res["0"], detect_res["0"]["time(s)"], detect_res["0"]["person"])
                    
                    speaker_box_lst = [normalize_bounding_box(resoStrLst[i], detect_res[str(j)]["person"][0]) if detect_res[str(j)]["person"] != [] else [0, 0, 0, 0] for j in range(0, frame_len)]
                    
                    
                    spf_lst = [detect_res[str(j)]["time(s)"] for j in range(0, frame_len)]
                    
                    #print ("speaker_box_lst: ", len(speaker_box_lst), len(spf_lst))
                    
                    speaker_box_arr[i] = speaker_box_lst
                    
                    spf_arr[i] = spf_lst
                    
                    
    speaker_box_arr = np.asarray(speaker_box_arr)
    spf_arr = np.asarray(spf_arr)
    print ("get_accuracy_time_py speaker_box_arr: ", speaker_box_arr.shape, spf_arr.shape)
                
    return speaker_box_arr, spf_arr


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


def apply_func_get_acc(element_arr):
    
    boxA = element_arr[0:4]
    boxB = element_arr[4:8]
    
    #print ("boxA: ", element_arr, boxA, boxB)
    return bb_intersection_over_union(boxA, boxB)

def get_acc_arry_each_frame(speaker_box_arr):
    
    # get accuracy numpy array from speaker_box_arr, using the expensive config as the ground truth
    # speaker_box_arr[0] as the ground truth
    
    print ("get_acc_arry_each_frame speaker_box_arr: ", speaker_box_arr.shape)
    reso_num, frm_number, boxsize = speaker_box_arr.shape
    print ("get_acc_arry_each_frame reso_num: ", reso_num, frm_number)
    
    acc_arr = np.zeros((reso_num, frm_number))
    acc_arr[0] = np.ones(frm_number, dtype = np.float32)       # first is ground truth, accuracy is 1.0
    for i in range(1, reso_num):
        
        new_combined_array = np.hstack((speaker_box_arr[0], speaker_box_arr[i]))
        #print("get_acc_arry_each_frame new_combined_array shape: ", new_combined_array.shape, new_combined_array[0].shape)
        
        acc_arr[i] = np.apply_along_axis(apply_func_get_acc, 1, new_combined_array)
        
    
    print("get_acc_arry_each_frame acc_arr[i] shape: ", acc_arr.shape, acc_arr[4])
    
    #np.apply_along_axis(, 1)
    
    return acc_arr


def write_numpy_into_file(arr, out_file_path):
    
    with open(out_file_path, 'wb') as f:
        np.save(f, arr)
    
    

def read_numpy(npy_file):
    arr = np.load(npy_file)
    return arr


def transfer_data_numpy():
    
    file_dir_lst =  ["sample_01_out/", "sample_02_out/",  \
                    "sample_03_out/", "sample_04_out/", \
                    "sample_05_out/", "sample_06_out/",  \
                    "sample_07_out/", "sample_08_out/", \
                    "sample_09_out/", "sample_10_out/", \
                    "sample_11_out/", "sample_12_out/",  \
                    "sample_13_out/", "sample_14_out/", \
                    "sample_15_out/", "sample_16_out/", \
                    "sample_17_out/", "sample_18_out/",  \
                    "sample_19_out/", "sample_20_out/", \
                    "sample_21_out/", "sample_22_out/", \
                    "sample_23_out/", "sample_24_out/",  \
                    "sample_25_out/", "sample_26_out/", \
                    "sample_27_out/", "sample_28_out/"]
    
    for file_dir in file_dir_lst[28:28]:
        data_dir = input_dir + file_dir   # "sample_01_out/"
        
        speaker_box_arr, spf_arr = get_accuracy_time_py(data_dir)
        
        acc_arr = get_acc_arry_each_frame(speaker_box_arr)
        
        file_no = "single"   # data_dir.split("/")[-2].split("_")[1]
        out_file_path = data_dir  +  file_no + "_speaker_box.npy"
        write_numpy_into_file(speaker_box_arr, out_file_path)
        
        out_file_path = data_dir  + file_no + "_spf.npy"
        write_numpy_into_file(spf_arr, out_file_path)
        
        
        out_file_path = data_dir  + file_no + "_acc.npy"
        write_numpy_into_file(acc_arr, out_file_path)
    
if __name__== "__main__": 

    
    #transfer_data_numpy()
    input_video_loc = "/var/fubao/videoAnalytics_objectTracking/input_output/speaker_video_dataset/sample_02.mp4"
    #file_name = "sample_01.mp4"
    extract_video_frames(input_video_loc)
    
    
    
    