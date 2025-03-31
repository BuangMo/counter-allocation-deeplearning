# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 19:23:52 2025

@author: Buang
"""
from math import sqrt

class RadarStageProgresion:
    def __init__(self):
        self.search_avg_time = 10
        self.acquisition_avg_time = 6
        self.tracking_avg_time = 8
        self.avg_times = (10, 6, 8)
        
        self.je_threshold = 0.1
        self.time_resolution = 2
        self.weight = 1.5
        
        self.radar_range = (7, 6, 8, 9, 7, 9, 9, 7, 7)
        self.radar_stages = ('Search', 'Acquisition', 'Tracking', 'Guidance')
        self.stage_dict = {'search': 0, 'acquisition': 1, 'tracking': 2, 'guidance': 3}
        self.debug = True
        
        # reads the radar cross section and jamming effect files
        self.rcs = self.readFile('thesis1/RCS.txt')
        self.jam_factor_data = self.readFile('thesis1/jam_factor_data.txt')
        
        # reads the way point and scenario data tables
        self.scenario_data = self.readFile('thesis1/scenario_data.txt')
        self.waypoint_data = self.readFile('thesis1/waypoint_data.txt')
        
        self.wp_instance = 0
        self.wp_tracker = 0
        self.updateWaypointVector()
        
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
    
    def updateWaypointVector(self):
        '''Generates the vector for the waypoint co-ordinates using linear interpolation'''
        time_dif_val = (self.waypoint_data[self.wp_instance + 1][3] - self.waypoint_data[self.wp_instance][3])
        
        # using linear interpolation to generate the x-value
        x_val_dif = (self.waypoint_data[self.wp_instance + 1][0] - self.waypoint_data[self.wp_instance][0])
        x_val = x_val_dif * (2 / time_dif_val)
        
        # using linear interpolation to generate the y-value
        y_val_dif = (self.waypoint_data[self.wp_instance + 1][1] - self.waypoint_data[self.wp_instance][1])
        y_val = y_val_dif * (2 / time_dif_val)
        
        # using linear interpolation to generate the z-value
        z_val_dif = (self.waypoint_data[self.wp_instance + 1][2] - self.waypoint_data[self.wp_instance][2])
        z_val = z_val_dif * (2 / time_dif_val)
        
        self.wp_vector = (x_val, y_val, z_val)
        self.wp_instance += 1
        self.wp_tracker = 0

    def updatePlatformCoordinates(self):
        '''Updates the position of the current position of the platform'''
        x_plat = self.waypoint_data[self.wp_instance - 1][0] + self.wp_vector[0] * self.wp_tracker
        y_plat = self.waypoint_data[self.wp_instance - 1][1] + self.wp_vector[1] * self.wp_tracker
        z_plat = self.waypoint_data[self.wp_instance - 1][2] + self.wp_vector[2] * self.wp_tracker

        self.current_waypoint = (x_plat, y_plat, z_plat)
        self.wp_tracker += 1

    def isWithinRange(self, threat_id):
        '''calculates the slant distance between the threat and the current position of the platform'''
        slant = [(self.scenario_data[threat_id][i] - self.current_waypoint[i])**2 for i in range(2)]
        slant = sqrt(sum(slant))
        
        # check to see if olatform is within threat radar range
        if slant <= self.radar_range[threat_id]:
            return True
        else:
            return False
    
    def calculateB(self, rcs_effect, threat_id, time_instance):
        '''Calculates the radar stage-progression modifier'''
        jam_factor = self.jam_factor_data[time_instance][threat_id]
        if abs(jam_factor) >= self.je_threshold:
            rate_modifier = 1 - (self.weight * jam_factor * rcs_effect)
        else:
            rate_modifier = 1
        
        return rate_modifier
    
    def calculateL(self, threat_id, time_instance):
        '''Calculates the effect of the of the RCS on the radar stage progression'''
        if self.jam_factor_data[time_instance][threat_id] >= 0:
            rcs_effect = 1 - self.rcs[time_instance][threat_id]
        else:
            rcs_effect = self.rcs[time_instance][threat_id]
            
        return rcs_effect
    
    def calculateStageProgression(self, prev_progression, rate_modifier):
        '''Calculates the radar stage progression for a radar threat given the previous value'''
        radar_progression = prev_progression - self.time_resolution * rate_modifier
        
        return radar_progression
        
def main():
    obj = RadarStageProgresion()
    prev_progression_stage = 9 * [obj.radar_stages[0]]
    prev_progression = []
    
    for inst in range(len(obj.jam_factor_data)):
        if inst > 18: break
        current_progression = []
        current_radar_stage = []

        # updates the vector of the platform for the current waypoint
        if int(2 * inst) >= int(obj.waypoint_data[obj.wp_instance][3]) and int(2 * inst) < 120:
            obj.updateWaypointVector()
        
        # updates the position of the platform
        obj.updatePlatformCoordinates()
        #if obj.debug: print(f'{inst * 2}: {obj.current_waypoint}')
        
        # iterates through all of the threats
        for threat in range(len(obj.scenario_data) - 4):
            # check to see if the platform is within the threat radar range
            if obj.isWithinRange(threat):
                # calculate the radar stage progression value
                rcs_effect = obj.calculateL(threat, inst)
                rate_modifier = obj.calculateB(rcs_effect, threat, inst)
                stage_progression = obj.calculateStageProgression(prev_progression[threat], rate_modifier)

                # update variables for the break-lock
                
                # updates the variables provided the value of the stage progression
                if stage_progression <= 0.0:
                    new_stage_val = obj.stage_dict[prev_progression_stage[threat].lower()] + 1      # gets the value for the new stage based on previous stage
                    current_radar_stage.append(obj.radar_stages[new_stage_val])             # sets the threat current stage
                    current_progression.append(obj.avg_times[new_stage_val])
                else:
                    new_stage_val = obj.stage_dict[prev_progression_stage[threat].lower()]
                    current_radar_stage.append(obj.radar_stages[new_stage_val])
                    current_progression.append(stage_progression)
            else:
                current_progression.append(obj.search_avg_time)
                current_radar_stage.append(obj.radar_stages[0])  

        if obj.debug: print(f'{2*inst}: {current_radar_stage}')

        # update the prev_progression and prev_progression_stage for tracking
        prev_progression = current_progression
        prev_progression_stage = current_radar_stage

if __name__ == '__main__':
    main()