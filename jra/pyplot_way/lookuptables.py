# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 13:38:36 2025

@author: Buang
"""

class LookupTablesAndVariables:
    def __init__(self):
        self.stage_effectiveness = (
            #    rgpo  vgpo   cp    nj   MFT
                (-1.0, -1.0,  1.0,  0.8, 1.0),                                  # search stage
                (-1.0, -1.0,  1.0,  0.9, 1.0),                                  # acquistion stage
                ( 1.0,  1.0, -1.0, -1.0, 0.0),                                  # tracking stage
                ( 1.0,  1.2, -1.0, -1.0, 0.0)                                   # guidanance stage
            )
        self.jamming_interaction_factors_IN = (
            #    rgpo  vgpo   cp    nj   mft
                ( 0.0,  0.2, -0.3, -0.2, 0.0),                                  # rgpo
                ( 0.2,  0.0, -0.3, -0.2, 0.0),                                  # vgpo
                (-0.2, -0.2,  0.0,  0.2, 0.2),                                  # cp
                ( 0.0,  0.0,  0.2,  0.0, 0.0),                                  # nj
                (-0.3, -0.3,  0.3,  0.0, 0.0)                                   # mft
            )
        self.cross_effect = (
            #    t1   t2   t3   t4   t5   t6   t7   t8   t9   t10
                (1.0, 0.4, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),             # t1
                (0.4, 1.0, 0.4, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0),             # t2
                (0.2, 0.4, 1.0, 1.0, 0.4, 0.2, 0.2, 0.0, 0.0, 0.0),             # t3
                (0.2, 0.4, 1.0, 1.0, 0.4, 0.2, 0.2, 0.0, 0.0, 0.0),             # t4
                (0.0, 0.2, 0.4, 0.4, 1.0, 0.4, 0.4, 0.0, 0.0, 0.0),             # t5
                (0.0, 0.0, 0.2, 0.2, 0.4, 1.0, 0.6, 0.2, 0.0, 0.0),             # t6
                (0.0, 0.0, 0.2, 0.2, 0.4, 0.6, 1.0, 0.6, 0.4, 0.2),             # t7
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6, 1.0, 0.4, 0.2),             # t8
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 1.0, 0.4),             # t9
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.4, 1.0)              # t10
            )
        self.nj_cross_effect = (
            #    l7    l6    l5    l4    l3    l2    l1    c     h1    h2    h3    h4    h5    h6    h7  
                (0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.40, 1.00, 0.40, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00),     # narrow
                (0.00, 0.00, 0.00, 0.00, 0.16, 0.32, 0.80, 0.80, 0.80, 0.32, 0.16, 0.00, 0.00, 0.00, 0.00),     # medium-narrow
                (0.00, 0.00, 0.00, 0.12, 0.24, 0.60, 0.60, 0.60, 0.60, 0.60, 0.24, 0.12, 0.00, 0.00, 0.00),     # medium
                (0.00, 0.00, 0.08, 0.16, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.16, 0.08, 0.00, 0.00),     # medium-wide
                (0.00, 0.04, 0.08, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.08, 0.04, 0.00)      # wide
            )
        self.nj_cross_effect_tt7 = (
            #    l7    l6    l5    l4    l3    l2    l1    c     h1    h2    h3    h4    h5    h6    h7  
                (0.00, 0.00, 0.00, 0.00, 0.20, 0.40, 0.60, 1.00, 0.60, 0.40, 0.20, 0.00, 0.00, 0.00, 0.00),     # narrow
                (0.00, 0.00, 0.00, 0.16, 0.32, 0.48, 0.80, 0.80, 0.80, 0.48, 0.32, 0.16, 0.00, 0.00, 0.00),     # medium-narrow
                (0.00, 0.00, 0.12, 0.24, 0.36, 0.60, 0.60, 0.60, 0.60, 0.60, 0.36, 0.24, 0.12, 0.00, 0.00),     # medium
                (0.00, 0.08, 0.16, 0.24, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.24, 0.16, 0.08, 0.00),     # medium-wide
                (0.04, 0.08, 0.12, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.12, 0.08, 0.04)      # wide
            )
        self.chaff_interference_factors = (
            #   distraction dilution
                (   1.0,      1.2  ),       # rgpo
                (   1.0,      1.2  ),       # vgpo
                (   1.2,      0.8  ),       # cp
                (   1.0,      1.0  ),       # nj
                (   1.0,      0.7  )        # mft
            )
        self.chaff_stage_effectiveness = (
            #   distraction dilution
                (   0.8,     -1.0  ),                       # search stage
                (   1.0,     -1.0  ),                       # acquisition stage
                (   0.0,      1.0  ),                       # tracking stage
                (   0.0,      1.0  )                        # guidance stage
            )
        self.chaff_illumination_technique = (
            #   distraction dilution
                (  -0.3,      0.2  ),                       # rgpo
                (  -0.3,      0.2  ),                       # vgpo
                (   0.3,     -0.3  ),                       # cp
                (   0.0,     -0.2  ),                       # nj
                (   0.0,      0.0  )                        # mft
            )
        self.decoy_multiplication_factors = (
            #    search acquisition tracking guidance
                ( 0.8,     0.8,       1.2,     1.2  ),      # rgpo
                ( 0.8,     0.8,       1.2,     1.2  ),      # vgpo
                ( 0.8,     0.8,       0.8,     0.8  ),      # cp
                ( 0.8,     0.8,       0.8,     0.8  ),      # nj
                ( 0.8,     0.8,       0.8,     0.8  )       # mft
            )
        
        # reads the antenna 1 gain data from a file and converts it into floats
        self.agp1 = self.readFile('thesis1/AGP1.txt')
        
        # reads the antenna 2 gain data from a file and converts it into floats
        self.agp2 = self.readFile('thesis1/AGP2.txt')
                
        # converts threats x-y co-ordinates data read from a file line by line into a list and then converts it into floats
        self.scenario_data = self.readFile('thesis1/scenario_data.txt')
                
        # reads the waypoint data from a file and converts it into floats
        self.waypoint_data = self.readFile('thesis1/waypoint_data.txt')
        #print(self.flight_data)
        
    def readFile(self, filename):
        '''reads the data from the file and returns it as a list'''
        file_data = []
        with open(filename, 'r') as fd_file:
            for line in fd_file:
                data = line.split(';')                                          # converts the data into a list using the separator ';'
                data.pop()                                                      # removes the last item from the list
                data = [float(ele) for ele in data]                             # converts the line of data from str to float elements
                file_data.append(data)
                
        return file_data
    
    def readFileStr(self, filename):
        file_data = []
        with open(filename, 'r') as fd_file:
            for line in fd_file:
                data = line.split(';')                                          # converts the data into a list using the separator ';'
                data.pop()                                                      # removes the last item from the list
                data = [ele for ele in data]                             # converts the line of data from str to float elements
                file_data.append(data)
                
        return file_data
        
#if __name__ == '__main__':
    #obj = LookupTablesAndVariables()
        
        
        
        
        
        
        
        
        
        
        