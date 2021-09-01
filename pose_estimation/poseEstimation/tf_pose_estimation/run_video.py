import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=True,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--outputFile', type=str, default='', help='Output file path')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)

    out_video_file = args.outputFile
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  #(*'mp4v')   # ('m','p','4','v')  #(*'mp4v') #('m','p','4','v') #  Be sure to use lower case
    out_video = cv2.VideoWriter(out_video_file, fourcc, 25.0, (w, h))
    
    print("out_video_file :", out_video_file)
    
    out_data_dir = "/media/fubao/TOSHIBAEXT/research_bakup/data_video_analytics/input_output/pose_one_person_diy_video_dataset/05_dance_out_pose_estimation_frames/"
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    frm_index = 0
    while cap.isOpened():
        ret_val, image = cap.read()

        humans = e.inference(image)
        print('human: ', frm_index, humans)
        #if not args.showBG:
        #    image = np.zeros(image.shape)
        npimg = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        #cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv2.imshow('tf-pose-estimation result', image)
        #cv2.imwrite(out_data_dir + str(frm_index) + ".jpg", npimg)
        #fps_time = time.time()
        
        #print("frm_num :", frm_index, image.shape)
        out_video.write(npimg)
        

        if cv2.waitKey(1) == 27:
            break
        frm_index += 1
        
        if frm_index >= 10:
            break
        
    cv2.destroyAllWindows()
logger.debug('finished+')

# python3 run_video.py --resolution "1280x720" --video "/media/fubao/TOSHIBAEXT/research_bakup/data_video_analytics/input_output/pose_one_person_diy_video_dataset/036_dance.mp4" --outputFile "/media/fubao/TOSHIBAEXT/research_bakup/data_video_analytics/input_output/pose_one_person_diy_video_dataset/036_dance_pose_estimation_out.mp4"