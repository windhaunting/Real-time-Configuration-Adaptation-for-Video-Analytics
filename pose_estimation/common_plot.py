#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 20:56:51 2020

@author: fubao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:28:05 2019

@author: fubao
"""

# plot common file

import numpy as np

import matplotlib 
matplotlib.use('Agg') 
import matplotlib.ticker as mticker

from matplotlib import pyplot as plt

import matplotlib.pylab as pylab
params = {'legend.fontsize': 24,
          'figure.figsize': (15.69, 12.27),
         'axes.labelsize':24,
         'axes.titlesize':26,
         'xtick.labelsize':26,
         'ytick.labelsize':26}
pylab.rcParams.update(params)

# LABEL_SIZE = 22

def plotScatterLineOneFig(x_lst1, y_lst1, xlabel, ylabel, title_name):
    '''
    plot ScatterLine
    '''
    
    fig, axes=plt.subplots(nrows=1, ncols=1)
    axes.plot(x_lst1, y_lst1, zorder=0) 
    sc11 = axes.scatter(x_lst1, y_lst1, marker="o", color="r", zorder=0)
    axes.set(xlabel=xlabel, ylabel=ylabel)
    axes.set_title(title_name)
    
    return fig


def plotTwoLinesOneFigure(xList, yList1, yList2, xlabel, ylabel, title_name):
    '''
    plot two suplots   2X1 structure
    '''
    
    plt.figure()
    plt.plot(xList, yList1)
    plt.plot(xList, yList2)
    #plt.title('Moving speed of the cat')
    plt.xlabel(xlabel) #, fontsize=LABEL_SIZE)
    plt.ylabel(ylabel) #, fontsize=LABEL_SIZE)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.title(title_name)
    plt.grid(True)
    
    return plt
    
def plotOneScatter(xList, yList, xlabel, ylabel, title_name):
    plt.figure()
    plt.scatter(xList, yList)
    #plt.title('Moving speed of the cat')
    plt.xlabel(xlabel) # , fontsize=LABEL_SIZE)
    plt.ylabel(ylabel) #, fontsize=LABEL_SIZE)
    plt.title(title_name)
    plt.grid(True)
    
    return plt

def plotOneScatterLine(xList, yList, xlabel, ylabel, title_name):
    fig = plt.figure()
    plt.plot(xList, yList)
    plt.scatter(xList, yList)
    #plt.title('Moving speed of the cat')
    plt.xlabel(xlabel) # , fontsize=LABEL_SIZE)
    plt.ylabel(ylabel) # , fontsize=LABEL_SIZE)
    plt.title(title_name)
    plt.grid(True)
    
    return plt


def plotOneBar(xList, yList, xlabel, ylabel, title_name):
    plt.figure()
    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.bar(xList, yList)
    #plt.title('Moving speed of the cat')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(3))
    plt.xlabel(xlabel) # , fontsize=LABEL_SIZE)
    plt.ylabel(ylabel) # , fontsize=LABEL_SIZE)
    plt.title(title_name)
    plt.grid(True)

    return plt


def plot_bar_distribution(y_out, out_file_path, xlabel, ylabel, title):
    plt.figure()
    unique, counts = np.unique(y_out, return_counts=True)
    plt.bar(unique, counts)
    print ("unique, counts: ", unique, counts)

    
    plt.xlabel(xlabel) # , fontsize=LABEL_SIZE)
    plt.ylabel(ylabel) #, fontsize=LABEL_SIZE)
    plt.title(title)
    #plt.show()
    plt.savefig(out_file_path)


def plotTwoSubplots(x_lst, y_lst_1, y_lst_2, x_label, y_label_1, y_label_2, title_name):
    '''
    plot two suplots   2X1 structure
    '''
    fig,axes=plt.subplots(nrows=2, ncols=1)
    axes[0].plot(x_lst, y_lst_1, zorder=1) 

    sc1 = axes[0].scatter(x_lst, y_lst_1, marker="o", color="r", zorder=2)
    axes[1].plot(x_lst, y_lst_2, zorder=1) 
    sc2 = axes[1].scatter(x_lst,y_lst_2, marker="x", color="k", zorder=2)
    axes[0].set(xlabel=x_label, ylabel=y_label_1)
    axes[1].set(xlabel=x_label, ylabel=y_label_2)
    
    #axes[0].legend([sc1], ["Admitted"])
    #axes[1].legend([sc2], ["Not-Admitted"])
    axes[0].set_title(title_name)
    #plt.show()
    fig.tight_layout()
    
    return fig


def plotThreeSubplots(x_lst, y_lst_1, y_lst_2, y_lst_3, x_label, y_label_1, y_label_2, y_label_3, title_name):
    '''
    plot two suplots 3X1 structure
    '''
    
    fig,axes=plt.subplots(nrows=3, ncols=1)
    axes[0].plot(x_lst, y_lst_1, zorder=1) 
    sc1 = axes[0].scatter(x_lst, y_lst_1, marker="o", color="r", zorder=2)
    
    axes[1].plot(x_lst, y_lst_2, zorder=1) 
    sc2 = axes[1].scatter(x_lst,y_lst_2, marker="x", color="k", zorder=2)
    
    axes[2].plot(x_lst, y_lst_3, zorder=1) 
    sc3 = axes[2].scatter(x_lst,y_lst_3, marker="*", color="g", zorder=2)
    
    axes[0].set(xlabel='', ylabel=y_label_1)
    axes[1].set(xlabel='', ylabel=y_label_2)
    axes[2].set(xlabel=x_label, ylabel=y_label_3)
    
    #axes[0].legend([sc1], ["Admitted"])
    #axes[1].legend([sc2], ["Not-Admitted"])
    axes[0].set_title(title_name)
    #axes[1].set_title(title_name_2)
    #plt.show()
    
    fig.tight_layout()

    return fig

