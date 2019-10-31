# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:29:15 2019

@author: yanxi
"""

import numpy as np

'''
Input: <old feature>, <key point matrix>
<kpm> is 3d: frame-kp-xyv
'''
def featureAbsSpeed(oftr, kpm):
    assert oftr.ndim == 1
    assert kpm.ndim == 3
    assert kpm.shape[1] == len(oftr)
    assert kpm.shape[2] == 3
    nfrm, nkp = kpm.shape[:2]
    
    pass


def kp2feature(kpm):
    pass


def kp2featureConf(kpmList, conf):
    pass


def getOnePersonFeatureInputOutputAll001(data_pose_keypoint_dir, data_pickle_dir,  history_frame_num, max_frame_example_used, minAccuracy):
    '''
    get one person's all history keypoint,  plus over a period of resolution feature
    One person’s moving speed of all keypoints V i,k based on the euclidean d distance of current frame with the previous frame {f j−m , m = 1, 2, ..., 24}
    
    current_frame_id is also included in the next frame's id
    
    the input feature here we use the most expensive features first
    
    1120x832_25_cmu_estimation_result
    
    start from history_frame_num;
    the first previous history frames are neglected
    
    based on EMA
    add over a period of resolution a feature
    '''
    
    acc_frame_arr, spf_frame_arr = readProfilingResultNumpy(data_pickle_dir)
    
    confg_est_frm_arr = read_poseEst_conf_frm(data_pickle_dir)
    old_acc_frm_arr = acc_frame_arr
    #print ("getOnePersonFeatureInputOutput01 acc_frame_arr: ", acc_frame_arr.shape)

    # save to file to have a look
    #outDir = data_pose_keypoint_dir + "classifier_result/"
    #np.savetxt(outDir + "accuracy_above_threshold" + str(minAccuracy) + ".tsv", acc_frame_arr[:, :5000], delimiter="\t")
    
    
    #config_ind_pareto = getParetoBoundary(acc_frame_arr[:, 0], spf_frame_arr[:, 0])
    config_ind_pareto = [0, 1, 2, 4, 6, 8, 10, 12, 15, 17, 19, 21, 24, 27, 29, 31, 33, 37, 39, 42, 44, 47, 49, 52, 54, 56, 58, 61, 63, 66, 69]          #     # only testing cmu model not ther pareto boundary, testing only

    print ("getOnePersonFeatureInputOutput01 config_ind_pareto: ", config_ind_pareto)
    acc_frame_arr = acc_frame_arr[config_ind_pareto, :]
    spf_frame_arr =  spf_frame_arr[config_ind_pareto, :]
    
    print ("getOnePersonFeatureInputOutput01 acc_frame_arr: ", acc_frame_arr[:, 0], acc_frame_arr.shape)

    # select one person, i.e. no 0
    
    #max_frame_example_used = 1000   # 8000
    #current_frame_id = 25
    config_id_dict, id_config_dict = read_config_name_from_file(data_pose_keypoint_dir, False)

    
    # get new id map based on pareto boundary/'s result
    new_id_config_dict = defaultdict()
    for i, ind in enumerate(config_ind_pareto):
        new_id_config_dict[i] = id_config_dict[ind]
    id_config_dict = new_id_config_dict
    
    print ("config_id_dict: ", len(config_id_dict), id_config_dict)
    
    
    # only read the most expensive config
    filePathLst = sorted(glob(data_pose_keypoint_dir + "*1120x832_25_cmu_estimation_result*.tsv"))  # must read ground truth file(the most expensive config) first
    
    df_det = pd.read_csv(filePathLst[0], delimiter='\t', index_col=False)         # det-> detection

    print ("filePath: ", filePathLst[0], len(df_det))

    
    history_pose_est_arr = np.zeros((max_frame_example_used, COCO_KP_NUM, 2)) # np.zeros((len(df_det), COCO_KP_NUM, 2))        #  to make not shift when new frames comes, we store all values
    
    previous_frm_indx = 0
    
    
    input_x_arr = np.zeros((max_frame_example_used, 66, 2))       # 17 + 4*4 + 4*4 + 17
    y_out_arr = np.zeros((max_frame_example_used+1), dtype=int)
    
    #current_instance_start_video_path_arr = np.zeros(max_frame_example_used, dtype=int)
    current_instance_start_frm_path_arr = np.zeros(max_frame_example_used, dtype=object)
    
    prev_EMA_speed_arr = np.zeros((COCO_KP_NUM, 2))
    
    prev_EMA_relative_speed_arr2 = np.zeros((4, 2))        # only get 4 keypoints
    prev_EMA_relative_speed_arr3 = np.zeros((4, 2))        # only get 4 keypoints
    prev_EMA_relative_speed_arr4 = np.zeros((4, 2))        # only get 4 keypoints
    prev_EMA_relative_speed_arr5 = np.zeros((4, 2))        # only get 4 keypoints
    
    
    reso_feature_arr = np.zeros(max_frame_example_used, dtype=int)
    history_reso_arr = np.zeros(max_frame_example_used)
    prev_reso_aver = 0.0
    
    config_feature_arr = np.zeros(max_frame_example_used, dtype=int)
    history_config_arr = np.zeros(max_frame_example_used) # + config_ind_pareto[0]
    prev_config_aver = 0.0
    
    frmRt_feature_arr = np.zeros(max_frame_example_used)
    history_frmRt_arr = np.zeros(max_frame_example_used)
    prev_frmRt_aver = 0.0
    
    
    blurriness_feature_arr = np.zeros((max_frame_example_used, 1))
    
    frm_id_debug_only_arr = np.zeros(max_frame_example_used)

    select_frm_cnt = 0
    skipped_frm_cnt = 0

    switching_config_skipped_frm = -1
    switching_config_inter_skip_cnts = 0
    
    selected_configs_acc_lst = blist()      # check the selected config's for all frame's lst, in order to get the accuracy
    
    #video_id = int(data_pose_keypoint_dir.split('/')[-2].split('_')[1])
    
    #arr_feature_frameIndex = np.zeros(max_frame_example_used)        # defaultdict(int) each interval's starting point -- corresponding frame index
    
    for index, row in df_det.iterrows():  
        #print ("index, row: ", index, row)
        #reso = row['Resolution']
        #frm_rate = row['Frame_rate']
        #model = row['Model']
        #num_humans = row['numberOfHumans']        # number of human detected

        if index >= acc_frame_arr.shape[1]:
            break            
        
        imgPath = row['Image_path']
        current_iterative_frm_id = int(imgPath.split('/')[-1].split('.')[0])
        

        est_res = row['Estimation_result']
        reso = row['Resolution']
        width = int(reso.split('x')[0])
        height = int(reso.split('x')[1])
        if str(est_res) == 'nan':  # here select_frm_cnt does not increase
            skipped_frm_cnt += 1
            print ("nan nan est_res: ", est_res, index, select_frm_cnt)
            continue
        # skipping interval by frame_rate
        if ((switching_config_skipped_frm != - 1) and (switching_config_skipped_frm < (switching_config_inter_skip_cnts-skipped_frm_cnt))):   # switching_config_inter_skip_cnts:
            switching_config_skipped_frm += 1
            continue
        #print ("frm_id num_humans, ", reso, model, frm_id)
            
        kp_arr = getPersonEstimation(est_res, width, height)
        #history_pose_est_dict[previous_frm_indx] = kp_arr
         
        history_pose_est_arr[previous_frm_indx] = kp_arr
        #print ("kp_arr, ", kp_arr)
        #break    # debug only
        if previous_frm_indx >= history_frame_num:
            #print ("previous_frm_indx, ", previous_frm_indx, index)
            
            current_cofig = id_config_dict[int(y_out_arr[select_frm_cnt])]
            curr_frm_rate = int(current_cofig.split('-')[1])
            #print ("xxxx current_cofig: ", current_cofig, curr_frm_rate)
            # calculate the human moving speed feature (1)
            feature1_speed_arr = getFeatureOnePersonMovingSpeed(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_speed_arr)
            
            prev_EMA_speed_arr = feature1_speed_arr
            #calculate the relative moving speed feature (2)
            feature2_relative_speed_arr = getFeatureOnePersonRelativeSpeed1(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr2)
            prev_EMA_relative_speed_arr2 = feature2_relative_speed_arr
            #print ("feature1_speed_arr feature2_relative_speed_arr, ", feature1_speed_arr.shape, feature2_relative_speed_arr.shape)
            
            feature3_relative_speed_arr = getFeatureOnePersonRelativeSpeed2(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr3)
            prev_EMA_relative_speed_arr3 = feature3_relative_speed_arr

            feature4_relative_speed_arr = getFeatureOnePersonRelativeSpeed3(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr4)
            prev_EMA_relative_speed_arr4 = feature4_relative_speed_arr
            
            feature5_relative_speed_arr = getFeatureOnePersonRelativeSpeed4(history_pose_est_arr, select_frm_cnt, skipped_frm_cnt, curr_frm_rate, history_frame_num, prev_EMA_relative_speed_arr5)
            prev_EMA_relative_speed_arr5 = feature5_relative_speed_arr
            
            
            current_frame_relative_distance_arr = getFeatureRelativeDistanceClosenessKeyPoint(history_pose_est_arr, select_frm_cnt)
            
            cameraDistance_feature = getFeatureDistanceToCamera(history_pose_est_arr, select_frm_cnt)
                
            total_features_arr = np.vstack((feature1_speed_arr, feature2_relative_speed_arr))
            #print ("total_features_arr total_features_arr, ", frm_id,  total_features_arr.shape)
            #print ("total_features_arr1: ", total_features_arr.shape, feature3_relative_speed_arr.shape)
            
            total_features_arr = np.vstack((total_features_arr, feature3_relative_speed_arr))
            #print ("total_features_arr2: ", total_features_arr.shape, input_x_arr.shape)

            total_features_arr = np.vstack((total_features_arr, feature4_relative_speed_arr))
            
            total_features_arr = np.vstack((total_features_arr, feature5_relative_speed_arr))

            total_features_arr = np.vstack((total_features_arr, current_frame_relative_distance_arr))
            
            total_features_arr = np.vstack((total_features_arr, cameraDistance_feature))
            #print ("total_features_arr2: ", total_features_arr.shape, input_x_arr.shape)
            
            input_x_arr[select_frm_cnt]= total_features_arr  #  input_x_arr[frm_id-1] = total_features_arr
            
            #arr_feature_frameIndex[select_frm_cnt] = index+1          # frame_index because this start from 1 ***.jpg
            #print ("total_features_arr: ", frm_id-1)
            #previous_frm_indx = 1
                
            #y_out_arr[select_frm_cnt+1] = select_config(acc_frame_arr, spf_frame_arr,  select_frm_cnt+1+switching_config_inter_skip_cnts+skipped_frm_cnt, minAccuracy)
            y_out_arr[select_frm_cnt+1] = select_config(acc_frame_arr, spf_frame_arr,  current_iterative_frm_id-1, minAccuracy)
            current_cofig = id_config_dict[int(y_out_arr[select_frm_cnt])]
            
            selected_config_acc = acc_frame_arr[y_out_arr[select_frm_cnt], index]
            
            selected_configs_acc_lst.append(selected_config_acc)
            #print ("current_cofig: ", current_cofig)
            
            current_config_frmRt = int(current_cofig.split('-')[1])
            
            if switching_config_skipped_frm == - 1:
                switching_config_inter_skip_cnts = PLAYOUT_RATE-1 #  math.ceil(PLAYOUT_RATE/current_config_frmRt)-2  #       #math.ceil(PLAYOUT_RATE/frmRt)-1
            else:
                switching_config_inter_skip_cnts = PLAYOUT_RATE  # math.ceil(PLAYOUT_RATE/current_config_frmRt)-1  # PLAYOUT_RATE
                
            reso = int(current_cofig.split('-')[0].split('x')[1])

            history_reso_arr[select_frm_cnt] = reso
                
            curr_reso_aver= getConfigFeature(history_reso_arr, select_frm_cnt, prev_reso_aver)
            
            prev_reso_aver = curr_reso_aver
            
            reso_feature_arr[select_frm_cnt] =  curr_reso_aver # curr_reso_aver y_out_arr[index]    #curr_reso_aver
            #print ("current_cofig reso: ", current_cofig, reso, curr_reso_aver)


            history_config_arr[select_frm_cnt] = y_out_arr[select_frm_cnt]
            curr_config_aver= getConfigFeature(history_config_arr, select_frm_cnt, prev_config_aver)
            
            prev_config_aver = curr_config_aver
            
            config_feature_arr[select_frm_cnt] =  curr_config_aver  # y_out_arr[select_frm_cnt]     # curr_config_aver  #   # y_out_arr[select_frm_cnt]  #   #
            
            
            frmRt = int(current_cofig.split('-')[1])

            history_frmRt_arr[select_frm_cnt] = frmRt
                
            curr_frmRt_aver= getFrmRateFeature(history_frmRt_arr, prev_frmRt_aver)
            prev_frmRt_aver = curr_frmRt_aver
            
            
            #frmRt_feature_arr[select_frm_cnt] = curr_frmRt_aver
            
            #current_iterative_frm_id = previous_frm_indx + skipped_frm_cnt + switching_config_skipped_frm
            #lowest_config_id = [19, 10, 6, 2] 
            #stTime = time.time()
            blurrinessScore_arr =  getBlurrinessFeature(imgPath, current_iterative_frm_id-1)
            #print ("end TimeTimeTime: ", time.time()-stTime)
            blurriness_feature_arr[select_frm_cnt] = blurrinessScore_arr
            
            
            frm_id_debug_only_arr[select_frm_cnt] = current_iterative_frm_id
            
            #current_instance_start_video_path_arr[select_frm_cnt] = video_id
            current_instance_start_frm_path_arr[select_frm_cnt] = row['Image_path']
                
            skipped_frm_cnt = 0       
            select_frm_cnt += 1 
            switching_config_skipped_frm = 1
            

        previous_frm_indx += 1
        
        #how many are used for traing, validation, and test
        if previous_frm_indx > (max_frame_example_used-1):
            break 
        
    
    #arr_feature_frameIndex = arr_feature_frameIndex[history_frame_num:select_frm_cnt]
    input_x_arr = input_x_arr[history_frame_num:select_frm_cnt].reshape(input_x_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    
    current_instance_start_frm_path_arr = current_instance_start_frm_path_arr[history_frame_num:select_frm_cnt].reshape(current_instance_start_frm_path_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((current_instance_start_frm_path_arr, input_x_arr))
    
    # add each instance's starting frame's path
    #current_instance_start_video_path_arr = current_instance_start_video_path_arr[history_frame_num:select_frm_cnt].reshape(current_instance_start_video_path_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    #input_x_arr = np.hstack((current_instance_start_video_path_arr, input_x_arr))
    
    frmRt_feature_arr = frmRt_feature_arr[history_frame_num:select_frm_cnt].reshape(frmRt_feature_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((input_x_arr, frmRt_feature_arr))
    
    reso_feature_arr = reso_feature_arr[history_frame_num:select_frm_cnt].reshape(reso_feature_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((input_x_arr, reso_feature_arr))
   
    config_feature_arr = config_feature_arr[history_frame_num:select_frm_cnt].reshape(config_feature_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((input_x_arr, config_feature_arr))
    
    blurriness_feature_arr = blurriness_feature_arr[history_frame_num:select_frm_cnt].reshape(blurriness_feature_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    input_x_arr = np.hstack((input_x_arr, blurriness_feature_arr))
        
    #frm_id_debug_only_arr = frm_id_debug_only_arr[history_frame_num:select_frm_cnt].reshape(frm_id_debug_only_arr[history_frame_num:select_frm_cnt].shape[0], -1)
    #input_x_arr = np.hstack((input_x_arr, frm_id_debug_only_arr))
            
    y_out_arr = y_out_arr[history_frame_num+1:select_frm_cnt+1]
    #print ("reso_feature_arr, ", reso_feature_arr.shape, reso_feature_arr)
    print ("feature1_speed_arr, ", input_x_arr.shape, y_out_arr.shape, feature1_speed_arr.shape, feature2_relative_speed_arr.shape)

    #checkCorrelationPlot(data_pose_keypoint_dir, input_x_arr, y_out_arr, id_config_dict)
    return input_x_arr, y_out_arr, id_config_dict
