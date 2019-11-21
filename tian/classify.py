# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:20:55 2019

@author: yanxi
"""

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 


import torch
import torch.nn as nn

class ModelThresholdOr():
    def __init__(self):
        self.b=torch.randn(17,requires_grad=True)
        self.c=torch.randn(1,requires_grad=True)
    def forward(self,x):
        o=torch.sigmoid(x-self.b)
        p=torch.sigmoid(o.sum(1)-self.c)
        self.p=p
        return p
    def loss(self,y):
        l=(y-self.p)**2
        self.l=l.sum()
        return self.l    
    def backward(self, lr):
        self.l.backward()
        with torch.no_grad():
            self.b.sub_(lr*self.b.grad)
            self.c.sub_(lr*self.c.grad)
            self.b.grad.zero_()
            self.c.grad.zero_()


