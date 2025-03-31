# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 11:38:58 2025

@author: Buang
"""
from math import sqrt
import matplotlib.pyplot as plt

def calculateDistance(plat_coord, missile_coord):
    xp = plat_coord[0]
    yp = plat_coord[1]
    zp = plat_coord[2]
    
    xm, ym, zm = missile_coord
    
    return sqrt((xm - xp)**2 + (ym - yp)**2 + (zm - zp)**2)

def plotMissileCoord():
    missile_coord = [
            [6.0000, 6.0000],
            [6.2642, 6.0377],
            [6.5955, 6.1489],
            [6.9881, 6.3294],
            [7.4361, 6.5744],
            [7.9332, 6.8787],
            [8.4731, 7.2366],
            [8.8415, 7.8267]
        ]
    x_vals = [missile_coord[i][0] for i in range(len(missile_coord))]
    y_vals = [missile_coord[i][1] for i in range(len(missile_coord))]
    
    plt.plot(x_vals, y_vals)
    plt.show()

def main():
    debug = True
    platform_coord = [
            [8.00, 6.00, 8.00],
            [8.3333, 6.3333, 8.00],
            [8.6667, 6.6667, 8.00],
            [9.00, 7.00, 8.00],
            [9.33, 7.33, 8.00],
            [9.67, 7.67, 8.00],
            [10.00, 8.00, 8.00],
            [10.00, 8.57, 8.00]
        ]
    
    missile_init_coord = [6., 6., 0.]
    missile_velocity = 1700 / 3600 #km/s
    
    t = 6
    sdistance = calculateDistance(platform_coord[0], missile_init_coord)
    distance = calculateDistance(platform_coord[3], missile_init_coord)
    
    result = [platform_coord[3][i] - missile_init_coord[i] for i in range(len(missile_init_coord))]
    unit_vector = [element / distance  for element in result]
    missile_position = [element * t * missile_velocity  for element in unit_vector]
    missile_position = [round(missile_position[i] + missile_init_coord[i], 4) for i in range(len(missile_init_coord))]
    
    current_distance = calculateDistance(platform_coord[2], missile_position)
    distance1 = missile_velocity * t
    
    if debug: print(f'platform coord at 2 s = {platform_coord[1]}')
    if debug: print(f'slant distance = {distance}')
    if debug: print(f'vector from missile to platform = {result}')
    if debug: print(f'unit vector = {unit_vector}')
    if debug: print(f'missile position = {missile_position}')
    if debug: print(f'missile distance= {sdistance - distance1}')
    if debug: print(f'current distance = {current_distance}')
    print(missile_init_coord[:2])
    
    #plotMissileCoord()
    
if __name__ == '__main__':
    main()