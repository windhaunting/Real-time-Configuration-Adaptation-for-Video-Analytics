#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:28:05 2019

@author: fubao
"""

# plot common file


import matplotlib 
matplotlib.use('Agg') 
import matplotlib.ticker as mticker

from matplotlib import pyplot as plt


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
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.title(title_name)
    plt.grid(True)
    
    return plt
    

def plotOneScatterLine(xList, yList, xlabel, ylabel, title_name):
    plt.figure()
    plt.plot(xList, yList)
    plt.scatter(xList, yList)
    #plt.title('Moving speed of the cat')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title_name)
    plt.grid(True)
    
    return plt


def plotOneBar(xList, yList, xlabel, ylabel, title_name):
    plt.figure()
    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.bar(xList, yList)
    #plt.title('Moving speed of the cat')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(3))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title_name)
    plt.grid(True)

    return plt


