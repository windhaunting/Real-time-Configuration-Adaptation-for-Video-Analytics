#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:14:18 2019

@author: fubao
"""

import inspect
import pandas as pd

# define a class including each clip's profile result
class cls_profile_video(object):
    # cameraNo is the camera no. for multiple camera streaming. 
    # queryNo is the query type, currently human pose estimation
    # 'frameStartNo', 
    __slots__ = ['cameraNo', 'queryNo', 'resolution', 'frameRate','modelMethod', 'accuracy', 'costFPS']



def retrieve_name(var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]
            
def get_variable_name(strVar):
    '''
    get strVar name as string outpt
    '''
    strVarName = [ k for k,v in locals().items() if v == strVar][0]
    
    return strVarName


class cls_fifo:
    def __init__(self):
        self.data = {}
        self.nextin = 0
        self.nextout = 0
    def append(self, data):
        self.nextin += 1
        self.data[self.nextin] = data
    def pop(self):
        self.nextout += 1
        result = self.data[self.nextout]
        del self.data[self.nextout]
        return result
    def length(self):
        return len(self.data)
    
    
def getBufferedLag(current_buffer, PLAYOUT_RATE):
    '''
    current_buffer: cls_fifo type
    calculate how many lags (s) from how many frames stored in the buffer
    according to STANDARD_FPS
    '''
    buf_len = current_buffer.length()
    lag = buf_len/PLAYOUT_RATE
    
    return lag


def read_profile_data(dataFile):
    '''
    read the synthesized profile data
    '''
    df_config = pd.read_csv(dataFile, delimiter='\t', index_col=False)
    
    #print (df_config.columns)
    
    return df_config