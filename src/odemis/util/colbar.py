# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 12:51:12 2016

@author: Toon
"""

import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1

# hack for getting colorbar with the same height as the figure. For some reason if the aspect ratio is multiple of 5 it does not work


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1/aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)