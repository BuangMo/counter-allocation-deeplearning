# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:12:39 2025

@author: Buang
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython import display

def plotScene():
    '''plots the battlefield scene'''
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    colors = [0, 0, 0, 0]

    markers = (
        ('^', 'None'),          # IR Undetected Marker
        ('D', 'None'),          # IR Guidance Marker
        ('+', 'None'),          # RF Search Marker
        ('x', 'None'),          # RF Acquisition Marker
        ((4, 1, 0), 'None'),          # RF Tracking Marker
        ('s', 'None'),          # RF Guidance Marker
        ('p', 'None'),          # Jammer  Marker
        ('p', 'b'),             # jammer x decoy (filled) Marker
        ('o', 'b'),             # flares marker
        ('o', 'None')           # chaff marker
        )
    #cbar = plt.c
    plt.scatter(x, y, c=colors, cmap='hot', vmin=-1, vmax=1)
    cbar = plt.colorbar(ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label('Jam Factor')
    plt.xlabel('Position (km)')
    plt.ylabel('Position (km)')
    
    ticks = [0, 5, 10, 15, 20]
    margin = 1
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlim(min(ticks) - margin, max(ticks) + margin)
    plt.ylim(min(ticks) - margin, max(ticks) + margin)
    
    plt.tick_params(axis='both', direction='in', length=10, width=1, top=True, right=True)
    plt.grid(True, linestyle=':')
    plt.show()
    
def main():
    plotScene()
    
if __name__ == '__main__':
    main()