import numpy as np
from math import sqrt
from helper import plotScene

class Mathematics():
    def __init__(self, threat_data_fname, flight_data_fname):
        self.threat_data = []
        self.flight_data = []
        self.time_track = 1
        self.multiplier = 0
        self.waypoint_number = 1
        self.number_of_threats = 0
        self.debug = True

        # converts threats x-y co-ordinates data read from a file line by line into a list and then converts it into floats
        with open(threat_data_fname, 'r') as file:
            for line in file:
                data = line.split(';')                                          # converts the data into a list using the separator ';'
                data.pop()                                                      # removes the last item from the list
                data = [float(ele) for ele in data]                             # converts the line of data from str to float elements
                self.threat_data.append(data)
        
        self.number_of_threats = len(self.threat_data)
        
        if self.debug: print(f'total number of threats: {self.number_of_threats}')
                
        # reads the waypoint data from a file and converts it into floats
        with open(flight_data_fname, 'r') as fd_file:
            for line in fd_file:
                data = line.split(';')
                data.pop()
                data = [float(ele) for ele in data]
                self.flight_data.append(data)
                
        self.genFlightData()
        
    def genFlightData(self):
        '''Calculates the flight data add factors'''
        self.flight_data_comp = []
        
        for idx in range(1, len(self.flight_data)):
            time_diff = ( 2 ) / (self.flight_data[idx][3] - self.flight_data[idx-1][3])
            
            x_vals = (self.flight_data[idx][0] - self.flight_data[idx-1][0]) * time_diff
            y_vals = (self.flight_data[idx][1] - self.flight_data[idx-1][1]) * time_diff
            z_vals = (self.flight_data[idx][2] - self.flight_data[idx-1][2]) * time_diff
            t = self.flight_data[idx][3]
            roll = self.flight_data[idx][4]
            pitch = self.calculatePitch(idx)
            heading = self.calculateHeading(idx)
            rotmat_product = self.calcRotMatProduct(roll, pitch, heading)
            #print(f'[{x_vals}, {y_vals}, {z_vals}, {t}, {roll}, {pitch}, {heading}]')
            self.flight_data_comp.append([x_vals, y_vals, z_vals, t, roll, pitch, heading, rotmat_product])
            
        if self.debug: print(self.flight_data_comp)
        
    def calculateHeading(self, index):
        '''Calculates the heading of the platform'''
        x_change = self.flight_data[index][0] - self.flight_data[index - 1][0]
        y_change = self.flight_data[index][1] - self.flight_data[index - 1][1]
        waypoint_ground_distance = sqrt( x_change**2 + y_change**2 )
        
        # calculates heading of the platform considering the case
        if x_change >= 0:
            self.platform_heading = (180 / np.pi ) * np.arcsin(y_change / waypoint_ground_distance) 
        else:
            self.platform_heading = (180 / np.pi ) * ( np.pi - np.arcsin( y_change / waypoint_ground_distance ))
        
        return self.platform_heading
    
    def calculatePitch(self, index):
        '''Calculates the pitch of the platform at that instance'''
        x_change = self.flight_data[index][0] - self.flight_data[index - 1][0]
        y_change = self.flight_data[index][1] - self.flight_data[index - 1][1]
        z_change = self.flight_data[index][2] - self.flight_data[index - 1][2]
        waypoint_slant = sqrt(x_change**2 + y_change**2 + z_change**2)
        
        # calculate pitch and convert value from radians to degrees
        self.platform_pitch = (180 / np.pi ) * np.arcsin(z_change / waypoint_slant) 
        return self.platform_pitch
    
    def calcRotMatProduct(self, roll, pitch, heading):
        '''calculates the relative position of the threats'''
        # convert from degrees to radians
        alpha = (np.pi * heading) / 180
        beta = (np.pi * pitch) / 180
        gamma = (np.pi * roll) / 180
        
        # generates rotational matrices
        Rz_heading = [
                [np.cos( alpha ), np.sin( alpha ), 0],
                [-np.sin( alpha ), np.cos( alpha ), 0],
                [0, 0, 1]
            ]
        Ry_pitch = [
                [np.cos( -beta ), 0, -np.sin( -beta )],
                [0, 1, 0],
                [ np.sin( -beta ), 0, np.cos( -beta )]
            ]
        Rx_gamma = [
                [1, 0, 0],
                [0, np.cos( gamma ), np.sin( -gamma )],
                [0, -np.sin( -gamma ), np.cos( gamma )]
            ]
        Rz_heading = np.array(Rz_heading)
        Ry_pitch = np.array(Ry_pitch)
        Rx_gamma = np.array(Rx_gamma)
        
        rot_matrix_product = np.matmul(Rz_heading, Ry_pitch, Rx_gamma)
        
        print("\n\n\n")
        print(Rz_heading)
        print()
        print(Ry_pitch)
        print()
        print(Rx_gamma)
        print()
        print(np.matmul(Rz_heading, Ry_pitch, Rx_gamma))
        
        return rot_matrix_product
    
    def getPositionNow(self):
        '''Finds the current position of the platform'''
        # updates the waypoint and multiplier variable based on time
        if (self.time_track * 2) > self.flight_data[self.waypoint_number][3]:
            self.waypoint_number += 1
            self.multiplier = 0
        
        # gets the x, y, z co-ordinates and the time
        x_coord = self.flight_data[self.waypoint_number - 1][0] + ( 1 + self.multiplier ) * self.flight_data_comp[self.waypoint_number - 1][0]
        y_coord = self.flight_data[self.waypoint_number - 1][1] + ( 1 + self.multiplier ) * self.flight_data_comp[self.waypoint_number - 1][1]
        z_coord = self.flight_data[self.waypoint_number - 1][2] + ( 1 + self.multiplier ) * self.flight_data_comp[self.waypoint_number - 1][2]
        t = self.time_track * 2

        #print(x_coord) 
        #print(f'time = {t}')
        if self.time_track == 0:
            platform_pos = (x_coord, y_coord, z_coord, t, 0, None, None)
        else:
            platform_pos = (x_coord, y_coord, z_coord, t, self.flight_data_comp[self.waypoint_number-1][4], self.flight_data_comp[self.waypoint_number-1][5], self.flight_data_comp[self.waypoint_number-1][6])
        
        # update the variables
        self.multiplier += 1
        self.time_track += 1
        
        return platform_pos
    
    def calculateRelativeCoord(self, current_position_of_platform):
        '''Calculates the relative position of all the threats'''
        for idx, value in enumerate(self.threat_data):
            x_change = self.threat_data[idx][0] - current_position_of_platform[0]
            y_change = self.threat_data[idx][1] - current_position_of_platform[1]
            z_change = self.threat_data[idx][2] - current_position_of_platform[2]
            
            # constructs co-ordinate vectors of the threat relative to platform' axis
            omega = np.array([x_change, y_change, z_change]).transpose()
            rel_omega = np.dot(omega, self.flight_data_comp[self.waypoint_number-1][7])
            print(omega)
            print(self.flight_data_comp[self.waypoint_number-1][7])
            print(rel_omega)
            print()
            
           # get the relative co-ordinates
            rel_xchange = rel_omega[0] - current_position_of_platform[0]
            rel_ychange = rel_omega[1] - current_position_of_platform[1]
            rel_zchange = rel_omega[2] - current_position_of_platform[2]
            rel_slant_distance = sqrt( rel_xchange**2 + rel_ychange**2 + rel_zchange**2 )
            rel_ground_distance = sqrt( rel_xchange**2 + rel_ychange**2 )
            
            # calculate the elevation angle of threat
            elevation = ( 180 / np.pi ) * np.arcsin( rel_zchange / rel_slant_distance )
            print(f'elevation angle = {elevation} degrees')
            
            # calculate the azimuth angle of threat
            if rel_xchange >= 0:
                azimuth = ( 180 / np.pi ) * np.arcsin( rel_ychange / rel_ground_distance)
            else:
                azimuth = ( 180 / np.pi ) * (np.pi - np.arcsin( rel_ychange / rel_ground_distance))
                
            print(f'azimuth angle = {azimuth} degrees')            
            
    def __calculateElevation(self, relative_omega):
        elavation_angle = np.arc()
            
def main():
    td_name = 'scenario_data.txt'
    fd_name = 'waypoint_data.txt'
    
    obj = Mathematics(td_name, fd_name)
    #obj.time_track = 3
    print("\n\n\n")
    
    #for i in range(20):
    #platform_current_pos = obj.getPositionNow()
    #print(platform_current_pos)
    #print()
    #obj.calculateRelativeCoord(platform_current_pos)
        
    #plotScene()

if __name__ == '__main__':
    main()