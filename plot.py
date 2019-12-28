#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:26:31 2019

@author: fubao
"""

# plot common
import numpy as np

import matplotlib 
matplotlib.use('Agg') 

from matplotlib import pyplot as plt


import matplotlib.pylab as pylab
params = {'legend.fontsize': 14,
         'figure.figsize': (8, 8),
         'axes.labelsize': 14,
         'axes.titlesize':14,
         'xtick.labelsize':14,
         'ytick.labelsize':14}   # 'large'}
pylab.rcParams.update(params)




def plotFourSubplots(x_lst, y_lst_1, y_lst_2, y_lst_3, y_lst_4, x_label, y_label_1, y_label_2, y_label_3, y_label_4, title_name_1, title_name_2):
    '''
    plot two suplots 3X1 structure
    '''
    
    fig,axes=plt.subplots(nrows=4, ncols=1)
    axes[0].plot(x_lst, y_lst_1, zorder=1) 
    sc1 = axes[0].scatter(x_lst, y_lst_1, marker="o", color="r", zorder=2)
    
    axes[1].plot(x_lst, y_lst_2, zorder=1) 
    sc2 = axes[1].scatter(x_lst,y_lst_2, marker="x", color="k", zorder=2)
    
    axes[2].plot(x_lst, y_lst_3, zorder=1) 
    sc3 = axes[2].scatter(x_lst,y_lst_3, marker="*", color="g", zorder=2)
    
    axes[3].plot(x_lst, y_lst_4, zorder=1) 
    sc3 = axes[3].scatter(x_lst,y_lst_4, marker="*", color="c", zorder=2)
    
    axes[0].set(xlabel='', ylabel=y_label_1)
    axes[1].set(xlabel='', ylabel=y_label_2)
    axes[2].set(xlabel='', ylabel=y_label_3)
    axes[3].set(xlabel=x_label, ylabel=y_label_4)
    
    #axes[0].legend([sc1], ["Admitted"])
    #axes[1].legend([sc2], ["Not-Admitted"])
    axes[0].set_title(title_name_1)
    axes[1].set_title(title_name_2)
    #plt.show()
    
    fig.tight_layout()

    return fig
def plotThreeSubplots(x_lst, y_lst_1, y_lst_2, y_lst_3, x_label, y_label_1, y_label_2, y_label_3, title_name_1, title_name_2):
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
    axes[0].set_title(title_name_1)
    axes[1].set_title(title_name_2)
    #plt.show()
    
    fig.tight_layout()

    return fig



def plotFiveSubplots(x_lst1, y_lst11, y_lst12, x_lst2, y_lst2, x_lst3, y_lst3, x_lst4, y_lst4, x_lst5, y_lst5,
                     x_label1, x_label2, x_label3, x_label4, x_label5,  y_label1, y_label2, y_label3, y_label4, y_label5, title_name_1, title_name_4):
    '''
    plot five suplots 3X1 structure
    the first plot has two figures
    '''
    
    fig,axes=plt.subplots(nrows=5, ncols=1)
    axes[0].plot(x_lst1, y_lst11, zorder=1) 
    axes[0].plot(x_lst1, y_lst12, zorder=1) 
    sc11 = axes[0].scatter(x_lst1, y_lst11, marker="o", color="r", zorder=2)
    sc12 = axes[0].scatter(x_lst1, y_lst12, marker="x", color="g", zorder=2)
    
    
    axes[1].plot(x_lst2, y_lst2, zorder=1) 
    sc2 = axes[1].scatter(x_lst2,y_lst2, marker="x", color="k", zorder=2)
    
    axes[2].plot(x_lst3, y_lst3, zorder=1) 
    sc3 = axes[2].scatter(x_lst3,y_lst3, marker="*", color="g", zorder=2)
    
    
    axes[3].plot(x_lst4, y_lst4, zorder=1) 
    sc4 = axes[2].scatter(x_lst4,y_lst4, marker="*", color="g", zorder=2)
    
    axes[4].plot(x_lst5, y_lst5, zorder=1) 
    sc5 = axes[2].scatter(x_lst5,y_lst5, marker="*", color="g", zorder=2)
    
    axes[0].set(xlabel=x_label1, ylabel=y_label1)
    axes[1].set(xlabel=x_label2, ylabel=y_label2)
    axes[2].set(xlabel=x_label3, ylabel=y_label3)
    
    axes[1].set(xlabel=x_label4, ylabel=y_label4)
    axes[2].set(xlabel=x_label5, ylabel=y_label5)
    
    #axes[0].legend([sc1], ["Admitted"])
    #axes[1].legend([sc2], ["Not-Admitted"])
    axes[0].set_title(title_name_1)
    axes[3].set_title(title_name_4)
    #plt.show()
    
    fig.tight_layout()

    return fig



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
    axes[0].grid(True)
    axes[1].grid(True)

    #plt.show()
    #plt.savefig(outputPlotPdf)
    fig.tight_layout()
    
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
    
    
def plotTwoLineOneplot(x_lst1, y_lst11, y_lst12, xlabel, ylabel, title_name):
    '''
    plot five suplots 3X1 structure
    the first plot has two figures
    '''
    
    fig,axes=plt.subplots() # (nrows=1, ncols=1)
    axes.plot(x_lst1, y_lst11, zorder=0) 
    axes.plot(x_lst1, y_lst12, zorder=0) 
    sc11 = axes.scatter(x_lst1, y_lst11, marker="o", color="r", zorder=1, label='Predicted_Acc')
    sc12 = axes.scatter(x_lst1, y_lst12, marker="x", color="g", zorder=1,  label='Actual_Acc')
    
    axes.set(xlabel=xlabel, ylabel=ylabel)
    axes.set_title(title_name)
    axes.grid(True)
    #axes.legend(["Predicted_Acc"], ["Actual_Acc"])
    axes.legend(loc='lower right')

    return fig


def plotTwoSubPlotOneFig(x_lst, y_lst1, y_lst2, xlabel, ylabel1, ylabel2, title_name):
    '''
    plot five suplots 3X1 structure
    the first plot has two figures
    '''
    
    fig,axes=plt.subplots(nrows=2, ncols=1)
    axes[0].plot(x_lst, y_lst1, zorder=1) 
    sc1 = axes[0].scatter(x_lst, y_lst1, marker="o", color="r", zorder=2)
    axes[0].set_title(title_name)
    axes[0].grid(True)
    
    axes[1].plot(x_lst, y_lst2, zorder=1) 
    axes[1].grid(True)
    sc2 = axes[1].scatter(x_lst,y_lst2, marker="x", color="k", zorder=2)
    axes[1].set(xlabel=xlabel, ylabel=ylabel2)

    axes[0].grid(True)
    axes[1].grid(True)
    #axes.legend(["Predicted_Acc"], ["Actual_Acc"])
    #axes.legend(loc='lower right')
    
    return fig

def plotLineOneplotWithSticker(x_lst_stickers, y_lst1, xlabel, ylabel, title_name):
    '''
    plot with sticker
    '''
    x_lst1 = range(0, len(x_lst_stickers))
    fig,axes=plt.subplots(nrows=1, ncols=1)
    #print ("x_lst1 :", x_lst1, y_lst1)
    axes.set_xticklabels(x_lst_stickers)  # fontdict={'fontsize': 3, 'horizontalalignment': loc})
    #axes.set_xlim(-1,70)

    plt.setp(axes.get_xticklabels(), fontsize=5, rotation='vertical')


    axes.plot(x_lst1, y_lst1, zorder=0) 
    sc11 = axes.scatter(x_lst1, y_lst1, marker="o", color="r", zorder=0)
    axes.set(xlabel=xlabel, ylabel=ylabel)
    axes.set_title(title_name)
    axes.set_xticks(range(len(x_lst1)))
    axes.grid(True)

    return fig

def plotBarTwoDataWithSticker(x_lst_stickers, y_lst1, y_lst2, xlabel, ylabel, legend_labels, title_name, output_plot_pdf_path):

    pdf = matplotlib.backends.backend_pdf.PdfPages(output_plot_pdf_path)
    #fig = plotTwoLineOneplot(xlst, y_lst1, y_lst2, xlabel, ylabel, title_name)
    x_lst1 = np.arange(0, len(x_lst_stickers))
    fig,axes=plt.subplots(nrows=1, ncols=1)
    print ("x_lst_stickers :", x_lst_stickers)
    #axes.set_xlim(-1,70)
    #plt.setp(axes.get_xticklabels(), fontsize=12, rotation='vertical')

    width = 0.3
    #axes.plot(x_lst1, y_lst1, zorder=0) 
    sc11 = axes.bar(x_lst1-width, y_lst1, width, color='r', align='center', label = legend_labels[0])
    sc12 = axes.bar(x_lst1, y_lst2,  width, color='b', align='center', label = legend_labels[1])
    axes.legend(loc='best', shadow=True)  # , fontsize='small')  
    
    axes.set(xlabel=xlabel, ylabel=ylabel)
    axes.set_title(title_name)
    axes.set_xticks(range(len(x_lst1)))
    axes.set_xticklabels(x_lst_stickers) # , rotation='vertical', fontdict={'fontsize': 8})
    axes.grid(True)

    pdf.savefig(fig)
    
    pdf.close()
    

def plotBarThreeDataWithSticker(x_lst_stickers, y_lst1, y_lst2, y_lst_3, xlabel, ylabel, legend_labels, title_name, output_plot_pdf_path):

    pdf = matplotlib.backends.backend_pdf.PdfPages(output_plot_pdf_path)
    x_lst1 = np.arange(0, len(x_lst_stickers))
    fig,axes=plt.subplots(nrows=1, ncols=1)
    print ("x_lst_stickers :", x_lst_stickers)
    #axes.set_xlim(-1,70)
    #plt.setp(axes.get_xticklabels(), fontsize=12, rotation='vertical')

    width = 0.3
    #axes.plot(x_lst1, y_lst1, zorder=0) 
    sc11 = axes.bar(x_lst1-width, y_lst1, width, color='r', align='center', label = legend_labels[0])
    sc12 = axes.bar(x_lst1, y_lst2,  width, color='b', align='center', label = legend_labels[1])
    sc13 = axes.bar(x_lst1+width, y_lst3,  width, color='g', align='center', label = legend_labels[2])

    axes.legend(loc='best', shadow=True)  # , fontsize='small')  
    
    axes.set(xlabel=xlabel, ylabel=ylabel)
    axes.set_title(title_name)
    axes.set_xticks(range(len(x_lst1)))
    axes.set_xticklabels(x_lst_stickers) # , rotation='vertical', fontdict={'fontsize': 8})
    axes.grid(True)

    pdf.savefig(fig)
    
    pdf.close()
    
def plotStickerMultipleLines(lst_x_lst_stickers_seg_ratio, y_lsts, legend_labels, y_line_value, xlabel, ylabel, title_name):
    '''
    plot with sticker
    '''
    #x_lst1 = range(0, len(x_lst_stickers))
    #print ("y_lsts :", y_lsts)
    fig,axes=plt.subplots(nrows=1, ncols=1)
    #axes.set_xlim(-1,70)

    #plt.setp(axes.get_xticklabels(), fontsize=5, rotation='vertical')


    for i, y_lst1 in enumerate(y_lsts[:-1]):
        
        x_lst1 = range(0, len(lst_x_lst_stickers_seg_ratio[i]))
        print ("plotStickerMultipleLines: ", len(x_lst1), len(y_lst1), lst_x_lst_stickers_seg_ratio[i], x_lst1, y_lst1)
        
        axes.plot(x_lst1, y_lst1, label=legend_labels[i]) 
        axes.scatter(x_lst1, y_lst1, marker="o", color="r", s=5)
        axes.set_xticklabels(lst_x_lst_stickers_seg_ratio[i], rotation = 45, ha="right", fontdict={'fontsize': 13})
        axes.set_xticks(range(len(lst_x_lst_stickers_seg_ratio[i])))

      
        
        if ylabel == 'Accuracy':
            axes.axhline(y=y_line_value, color='black', linestyle='-')
            
    leg = axes.legend(loc='upper right', shadow=True)       # 'best'
    

    #leg.get_frame().set_facecolor('#00FFCC')
    #sc11 = axes.scatter(x_lst1, y_lst1, marker="o", color="r", zorder=0)
    axes.set(xlabel=xlabel, ylabel=ylabel)
    axes.set_title(title_name)
    #axes.set_xticks(range(len(x_lst1)))

    axes.grid(True)

    return fig



def plotLineOneplot(x_lst1, y_lst1, xlabel, ylabel, title_name):
    '''
    plot five suplots 3X1 structure
    the first plot has two figures
    '''
    
    fig,axes=plt.subplots(nrows=1, ncols=1)
    axes.plot(x_lst1, y_lst1, zorder=0) 
    sc11 = axes.scatter(x_lst1, y_lst1, marker="o", color="r", zorder=0)
    axes.set(xlabel=xlabel, ylabel=ylabel)
    axes.set_title(title_name)
    axes.grid(True)

    return fig


def plotOnePlotMultiLine(y_lsts, legend_labels, xlabel, ylabel, title_name):
    '''
    plot multilple line on one plot
    the first plot has two figures
    '''
    
    fig,axes=plt.subplots(nrows=1, ncols=1)
    for i, y_lst in enumerate(y_lsts):
        x_lst1 = range(len(y_lst))
        #print ("plotOnePlotMultiLine xlst: ", len(x_lst1), len(legend_labels), len(y_lst))
        axes.plot(x_lst1, y_lst, zorder=0) 
        sc11 = axes.scatter(x_lst1, y_lst, label = legend_labels[i], zorder=0)
        
    axes.legend(loc='best', shadow=True)  
    axes.set(xlabel=xlabel, ylabel=ylabel, fontsize=13)
    axes.set_title(title_name)
    axes.grid(True)

    return fig


def plotTwoDimensionScatterLine(xList, yList, xlabel, ylabel, outputPlotPdf):
    plt.figure()
    plt.plot(xList, yList)
    plt.scatter(xList, yList)
    #plt.title('Moving speed of the cat')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True)
    plt.savefig(outputPlotPdf)
    
def plotTwoDimensionScatter(xList, yList, xlabel, ylabel, outputPlotPdf):
    
    plt.figure()
    #plt.plot(xList, yList)
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
    
    
    
def plot_fittingLine(x_lst1, y_lst1, xlabel, ylabel, title_name, xy_limScale=False):
     
    fig, axes= plt.subplots(nrows=1, ncols=1)
    #axes.plot(x_lst1, y_lst1, zorder=0) 
    sc11 = axes.scatter(x_lst1, y_lst1, marker="o", color="r", s=2)
    
    #axes = plt.gca()

    #m, b = np.polyfit(x_lst1, y_lst1, 1)    
    #X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
    #axes.plot(X_plot, m*X_plot + b, '-')

    x1, x2 = 0,1
    y1, y2 = 0, 1
    axes.plot([x1,x2],[y1,y2],'k-', color="b")
    
    axes.set(xlabel=xlabel, ylabel=ylabel)
    axes.set_title(title_name)
    
    if xy_limScale:
        axes.set_xlim([0, 1.0])
        axes.set_ylim([0, 1.0])
    axes.grid(True)


