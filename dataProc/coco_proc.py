#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:38:46 2019

@author: fubao
"""

# coco dataset demo

# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


dataDir='/media/fubao/TOSHIBAEXT/data_pose_estimation/coco/'


def coco_preprocess():
    '''
    get a video data and keynote result
    '''
    
    #dataDir='/media/fubao/TOSHIBAEXT/data_pose_estimation/coco/'
    dataType='val2017'
    # initialize COCO api for instance annotations
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

    coco=COCO(annFile)
    
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    
    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['traffic'])     # ['person','dog','skateboard'])     # ['person','dog','skateboard']
    imgIds = coco.getImgIds(catIds=catIds )
    
    print("imgIds: ", imgIds)
    imgIds = coco.getImgIds(imgIds = [458755])         # 324158
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    
    # load and display image
    # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
    # use url to load image
    I = io.imread(img['coco_url'])
    plt.axis('off')
    plt.imshow(I)
    plt.show()

    # initialize COCO api for person keypoints annotations
    annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
    coco_kps=COCO(annFile)
    
    # load and display keypoints annotations
    plt.imshow(I); plt.axis('off')
    ax = plt.gca()
    annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco_kps.loadAnns(annIds)
    coco_kps.showAnns(anns)
    
    print ("anns: ", anns)
    plt.show()


def coco_get_imageId(subCategory):
    '''
    get a category's images
    e.g. person
    '''
    
    dataType='val2017'
    # initialize COCO api for instance annotations
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

    coco=COCO(annFile)
    
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    
    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['subCategory'])     # ['person','dog','skateboard']
    imgIds = coco.getImgIds(catIds=catIds )
    
    return imgIds

def getImagesDetection():
    '''
    at at least realtime 30fps 
    use a category of images to make up a video length (but tranfer into frames.)
    unit 1min =  30*60 = 1800 frames
    
    '''
    x = 1
    
    
if __name__== "__main__": 
    coco_preprocess()