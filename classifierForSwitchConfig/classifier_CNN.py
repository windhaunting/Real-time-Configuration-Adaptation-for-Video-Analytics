#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:51:10 2019

@author: fubao
"""

# krea with CNN model
import sys
import os
import csv
import pickle
import time

import numpy as np
import pandas as pd

from glob import glob
from blist import blist
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight

from data_proc_input_NN import get_all_video_x_y_data
from data_proc_input_NN import get_all_augment_data_from_video_segment

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')


from profiling.common_prof import PLAYOUT_RATE




class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        

def train_test_cnn():
    
    
    X_train, X_test, y_train, y_test, class_weights = split_train_testing_data()
    
    print ("input_shape before: ", X_train.shape)
    
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    assert (X_train.shape[-1] == 1)

    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    
    print ("input_shape after: ", X_train.shape, input_shape, num_classes)
    batch_size = 32
    epochs = 200
    learning_rate = 0.001

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(64, (5, 5), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))        # 1000
    model.add(Dense(num_classes, activation='softmax'))


    #model.compile(loss=keras.losses.categorical_crossentropy,
    #          optimizer=keras.optimizers.SGD(lr=learning_rate),
    #          metrics=['accuracy'])
    
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])
    
    history = AccuracyHistory()

    

    model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          class_weight=class_weights,
          validation_data=(X_test, y_test),
          callbacks=[history])
    
    print('history training.acc:', history.acc)
    
    start_time = time.time()
    score = model.evaluate(X_test, y_test, verbose=0)
    print("elapsed test time:" , time.time() - start_time)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    plt.plot(range(0, len(history.acc)), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    
def split_train_testing_data():
    X, y = get_all_video_x_y_data()      #  get_x_y_data()    
    #X, y = get_all_augment_data_from_video_segment()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)
    print ("X_train one hot before , ", X.shape, X_train.shape)
    
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(np.ravel(y_train,  order='C')),
                                                 np.ravel(y_train,  order='C'))
    '''
    num_classes = 30
    enc = OneHotEncoder(n_values = num_classes)
    y_train = enc.fit_transform(y_train.reshape(-1,1)).toarray()
    print ("y_train one hot shape, ",y_train, y_train.shape, y_train[0], y_train[1])
    y_test = enc.fit_transform(y_test.reshape(-1,1)).toarray()
    '''
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    return X_train, X_test, y_train, y_test, class_weights


    
    
if __name__== "__main__": 
        
    train_test_cnn()