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



def plotTwoSubplots(x_lst, y_lst_1, y_lst_2, x_label, y_label_1, y_label_2, title_name, outputPlotPdf):
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
    #plt.savefig(outputPlotPdf)
    
    return fig

def plotUpsideDownTwoFigures(x_lst, y_lst_1, y_lst_2, x_label, y_label_1, y_label_2, outputPlotPdf):
    '''
    plot upside down
    '''
    fig, ax = plt.subplots()
    ax.scatter(x_lst, y_lst_1)
    ax.set_ylabel(y_label_1)
    
    y_lst_2 = [-1*v for v in y_lst_2]
    ax.scatter(x_lst, y_lst_2)
    ax.set_ylabel(y_label_2)

    # Formatting x labels
    plt.xticks(rotation=90)
    plt.tight_layout()
    # Use absolute value for y-ticks
    ticks =  ax.get_yticks()
    ax.set_yticklabels([int(abs(tick)) for tick in ticks])
    ax.set_xlabel(x_label)
    plt.show()
    plt.savefig(outputPlotPdf)

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