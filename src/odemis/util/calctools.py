# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:19:09 2015

@author: Toon
"""

import numpy as np


def findelement(list,value):
   
   element=np.argmin(np.abs(list-value))
    
   return element


def softwarebin(arr,binmat):
    
    if np.ndim(arr) == 2:
    #binning function, only works for matrices with even numbers
        new_shape = np.shape(arr)/np.array(binmat)
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1],
                 )
        mat = arr.reshape(shape).sum(-1).sum(1)
    
    if np.ndim(arr) == 3:
    #binning function, only works for matrices with even numbers
    
        new_shape = np.shape(arr)/np.array(binmat)
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1],
                 new_shape[2], arr.shape[2] // new_shape[2])
        mat = arr.reshape(shape).sum(-1).sum(1).sum(2)
            
    return mat
