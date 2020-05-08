# -*- coding: utf-8 -*-

import pickle
import csv



def write_pickle_data(data_instance, out_pickle_file):
    
    with open(out_pickle_file, "wb") as fp:   #Pickling
        pickle.dump(data_instance, fp)


def read_pickle_data(pickle_file):
    with open(pickle_file, "rb") as fp:   # Unpickling
        out = pickle.load(fp)
        
    return out

