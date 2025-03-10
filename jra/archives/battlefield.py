# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:12:39 2025

@author: Buang
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython import display

def battlefield():
    '''plots the battlefield scene'''
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    colors = [0, 0, 0, 0]

    plt.scatter(x, y, c=colors, cmap='hot', vmin=-1, vmax=1)
    plt.xlabel('Position (km)')
    plt.ylabel('Position (km)')
    plt.grid(True, linestyle=':')

    # sets up the colorbar
    cbar = plt.colorbar(ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label('Jam Factor')
    
    # set the x, y ticks
    ticks = [0, 5, 10, 15, 20]
    margin = 1
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.tick_params(axis='both', direction='in', length=10, width=1, top=True, right=True)

    # sets the x, y margins
    plt.xlim(min(ticks) - margin, max(ticks) + margin)
    plt.ylim(min(ticks) - margin, max(ticks) + margin)

    plt.show()
    
def main():
    battlefield()
    
if __name__ == '__main__':
    main()