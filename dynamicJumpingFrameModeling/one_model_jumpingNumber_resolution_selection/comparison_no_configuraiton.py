# -*- coding: utf-8 -*-









#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 19:39:54 2021

@author: fubao
"""

# compare without no configuration

# plot accuracy  in 10 second or even less than 10 second





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 19:44:17 2021

@author: fubao
"""

# object tracking 

# compare our method and show accuracy and latency

# compare without configuration adaptaton, which is similar to offline  once-time adaptation

# MOTrack is our name

import pickle
import numpy as np




def read_pickle_data(pickle_file):
    with open(pickle_file, "rb") as fp:   # Unpickling
        out = pickle.load(fp)
        
    return out


def read_acc_pickle_and_write_file(input_file_path, out_file_path):
    
    

    with open(input_file_path, "rb") as fp:   # Unpickling
        arr_out = pickle.load(fp)
        
    print ("arr_out: ", arr_out.shape)
    np.savetxt(out_file_path, arr_out, delimiter="\t")

    # get accuracy
    
def read_write_operation():
    #dataDir_MoTrack =  "../input_output/vehicle_tracking/sample_video_out/sample_video_json_01/jumping_number_result/video_applied_detection_result/"
    
    #dataDir_MoTrack =  "../input_output/speaker_video_dataset/sample_video_out/sample_01_out/data_instance_xy/minAcc_0.92/video_applied_detection_result/"

    dataDir_MoTrack =  "../../input_output/one_person_diy_video_dataset/output_001_dance/jumping_number_result/jumpingNumber_resolution_selection/intervalFrm-10_speedType-ema_minAcc-0.96/video_applied_detection_result/"

    
    input_file_path = dataDir_MoTrack + "arr_acc_segment_.pkl"
    out_file_path = dataDir_MoTrack + "Segment_arr_traffic.tsv"

    read_acc_pickle_and_write_file(input_file_path, out_file_path)

    # read delay
    input_file_path = dataDir_MoTrack + "arr_delay_up_to_segment_.pkl"
    out_file_path = dataDir_MoTrack + "Segment_delay_traffic.tsv"
    read_acc_pickle_and_write_file(input_file_path, out_file_path)
    
    
    # read acc
    input_file_path = dataDir_MoTrack + "no_adaptation_arr_acc_segment_.pkl"
    out_file_path = dataDir_MoTrack + "no_adaptation_segment_arr_traffic.tsv"
    read_acc_pickle_and_write_file(input_file_path, out_file_path)

    # read delay
    input_file_path = dataDir_MoTrack + "no_adaptation_arr_delay_up_to_segment_.pkl"
    out_file_path = dataDir_MoTrack + "no_adaptation_segment_delay_traffic.tsv"
    read_acc_pickle_and_write_file(input_file_path, out_file_path)
    
    

read_write_operation()