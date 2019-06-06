#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:26:31 2019

@author: fubao
"""

# plot common

import matplotlib 
matplotlib.use('Agg') 

from matplotlib import pyplot as plt


def plotTwoDimensionScatter(xList, yList, xlabel, ylabel, outputPlotPdf):
    
    plt.figure()
    plt.scatter(xList, yList)
    #plt.title('Moving speed of the cat')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True)
    plt.savefig(outputPlotPdf)
    
    

def plotTwoDimensionMultiLines(xList, yLists, xlabel, ylabel, changeLengendLst, outputPlotPdf):
    
    fig=plt.figure()
    plt.rcParams["axes.titlesize"] = 8

    #plt.title('Moving speed of the cat')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #print ('changeLengendLst: ', len(changeLengendLst), len(yLists))
    averageYLst = []
    for i, ylst in enumerate(yLists):
        plt.plot(xList, ylst, label=[changeLengendLst[i]])
        averageYLst.append(sum(ylst) / len(ylst) )
    plt.xticks(xList)
    #xmarks=[i for i in range(1,len(xList)+1, 2)]
    #plt.xticks(xmarks)
    
    #ax = plt.axes()
    #plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')

    plt.title("__".join(str(round(e,3)) for e in averageYLst))
    plt.legend(loc='upper right')

    plt.grid(True)
    plt.savefig(outputPlotPdf)