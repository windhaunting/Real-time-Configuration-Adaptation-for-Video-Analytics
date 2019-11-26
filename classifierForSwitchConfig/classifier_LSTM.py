#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:38:49 2019

@author: fubao
"""

# https://www.kaggle.com/sanket30/cudnnlstm-lstm-99-accuracy


#  https://github.com/yjchoe/TFCudnnLSTM

# LSTM-based classifier with tensorflow


# use CPU only for LSTM
# reference: https://gist.github.com/siemanko/b18ce332bde37e156034e5d3f60f8a23

# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb

from __future__ import print_function
import sys
import os
import csv
import pickle

import numpy as np
import pandas as pd

from glob import glob
from blist import blist

import tensorflow as tf
from tensorflow.contrib import rnn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from data_proc_input_NN import get_x_y_data
from data_proc_input_NN import get_all_video_x_y_data

current_file_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_cur + '/..')

from profiling.common_prof import PLAYOUT_RATE


    
def RNN(x, weights, biases, timesteps, num_hidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'], states


def generate_batch_data(x_train, y_train, batch_size):
    '''
    get the next batch of traing data
    '''
    lst_batched_train = blist()
    tmp_lst_train_x = blist()
    tmp_lst_train_y = blist()
    batch_step = 1
    i = 0
    while(i < len(x_train)):
        tmp_lst_train_x.append(x_train[i])
        tmp_lst_train_y.append(y_train[i])
        
        if batch_step >= batch_size:
            lst_batched_train.append([tmp_lst_train_x, tmp_lst_train_y])
            tmp_lst_train_x = []
            tmp_lst_train_y = []
            batch_step = 1
        
        batch_step += 1
        i += 1

    # does not add the rest train data without enough batch_size
    return lst_batched_train         


def get_next_batch(lst_batched_data):
    return next(iter(lst_batched_data))


def train_test_lstm():
    
    
    # get the training_testing data
    X_train, X_test, y_train, y_test = split_train_testing_data()
    
        
    val_set_size = len(X_test)//3
    batch_size = 32
    
    val_batch_size = batch_size  # 32
    lst_batched_train = generate_batch_data(X_train, y_train, batch_size)
    
    lst_batched_test = generate_batch_data(X_test, y_test, val_batch_size)
        
    print ("lst_batched_train shape, ", len(lst_batched_train))
    
    
    # Training Parameters
    learning_rate = 0.001
    training_steps = 300000 # 10000     # 10000
    display_step = 200
        
    
    # Network Parameters
    num_input= X_train.shape[1]//PLAYOUT_RATE
    
    print ("num_input: ", num_input)
    timesteps = PLAYOUT_RATE            #  window_size = PLAYOUT_RATE
    num_hidden = 64    # hidden layer num of features
        
    num_classes = 30 
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    
    print (" X shape: ", X .shape)

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    
            
    logits, states = RNN(X, weights, biases, timesteps, num_hidden)
    prediction = tf.nn.softmax(logits)
    
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    
    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    
    y_test_pred = []
    with tf.Session() as sess:
    
        # Run the initializer
        sess.run(init)
        step = 1
        while (step * batch_size < training_steps):
            batch_x, batch_y = get_next_batch(lst_batched_train)
            # Reshape data to get 28 seq of 28 elements
            batch_x = np.asarray(batch_x)
            batch_y = np.asarray(batch_y)
            #print (" batch_x shape: ", batch_x.shape, batch_y.shape)
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
           # batch_y = batch_y.reshape((batch_size, num_classes))
            #print("states.eval(feed_dict={X: batch_x}) :" , states.eval(feed_dict={X: batch_x}))
            #print("states.eval(feed_dict={Y: batch_y})", states.eval(feed_dict={Y: batch_y}))
            
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            

            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
    
            # Once in every 1000 *train* steps we evaluate the validation set (called test here)
            if step % 1000 == 0:
              val_step = 1
              val_acc = 0
              while val_step * val_batch_size < val_set_size:
                valid_x, valid_y = get_next_batch(lst_batched_test)
                valid_x = np.asarray(valid_x)
                valid_y = np.asarray(valid_y)
                valid_x = valid_x.reshape((val_batch_size, timesteps, num_input))

                step_acc = sess.run(accuracy, feed_dict={X: valid_x, Y: valid_y})
                val_acc+=step_acc
                val_step +=1
              
              val_acc = val_acc / val_step
              print("Validation set accuracy: %s" % val_acc)
          
            step += 1
            
        print("Optimization Finished!")
        
        test_data = X_train.reshape((-1, timesteps, num_input))
        test_label = y_train    
        print("Testing Accuracy for train data:", \
            sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
    
        test_data = X_test.reshape((-1, timesteps, num_input))   # X_train.reshape((-1, timesteps, num_input))
        test_label = y_test     # y_train
        print("Testing Accuracy for test data:", \
            sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
       
        y_test_pred = sess.run(train_op, feed_dict={X: test_data}) 
        
        print("Testing predicted:", y_test_pred)
        
def split_train_testing_data():
    #X, y = get_all_video_x_y_data()      #  get_x_y_data()    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)
    print ("X_train one hot before , ",X_train, X_train.shape)
    
    #y_train = tf.one_hot(indices=y_train, depth=30)
    enc = OneHotEncoder(n_values = 30)
    
    y_train = enc.fit_transform(y_train.reshape(-1,1)).toarray()
    print ("y_train one hot shape, ",y_train, y_train.shape, y_train[0], y_train[1])
    print ("x_out_arr shape, ", X.shape, y.shape)
    
    y_test = enc.fit_transform(y_test.reshape(-1,1)).toarray()

    return X_train, X_test, y_train, y_test

if __name__== "__main__": 
        
   
    train_test_lstm()
        
