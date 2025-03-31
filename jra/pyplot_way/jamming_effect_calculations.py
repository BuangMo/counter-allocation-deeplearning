# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 21:10:35 2025

@author: Buang
"""

from math import sqrt

from lookuptables import LookupTablesAndVariables

class Calculations(LookupTablesAndVariables):
    def __init__(self):
        self.channel_techs = ['none', 'none', 'none']                               # channel 1, 2 and 3 jamming techniques
        self.channel_target_threats = [0, 0, 0]                                 # threats the jamming techniques are optimised for
        self.threat_cross_effect = [0.0, 0.0]                                   # the cross effect of the current threat
        self.tech_dict = {'rgpo': 0, 'vgpo': 1, 'cp': 2, 'nj': 3, 'mft': 4}
        self.stage_dict = {'search': 0, 'acquisition': 1, 'tracking': 2, 'guidance': 3}
        self.threat_cr_factors = (.2, .3, .0, .2, .1, .2, .2, .3, .2)
        self.t1_type_id_conversion = [8, 10, 3, 4, 7, 1, 4, 5, 8]
        self.debug = False
        self.in_guidance = {}
        self.missile_velocity = (1400, 1700, 1900, 2200, 1800, 2000, 2200, 1400, 1400)

        super().__init__()
        
        # reads the strategy and stage table data information
        self.strategy_table_data = self.readFileStr('thesis1/strategy_table_data.txt')
        self.stage_table_data = self.readFileStr('thesis1/stage_table_data.txt')
        self.missile_locations = self.readFile('thesis1/missile_locations.txt')
        
    def updateStrategyData(self, time_instance):
        '''Updates the techniques variable and the threats for which the technique are optimised for'''
        # updates the jamming techniques
        self.channel_techs[0] = self.strategy_table_data[time_instance][4].lower()
        self.channel_techs[1] = self.strategy_table_data[time_instance][7].lower()
        self.channel_techs[2] = self.strategy_table_data[time_instance][10].lower()
        
        # updates the threats for which the channel techniques are optimised for
        self.channel_target_threats[0] = int(self.strategy_table_data[time_instance][5]) - 1
        self.channel_target_threats[1] = int(self.strategy_table_data[time_instance][8]) - 1
        self.channel_target_threats[2] = int(self.strategy_table_data[time_instance][11]) - 1
            
    def checkRange(self):
        '''check if platform is within range of threat'''
        pass
    
    def getCrossEffect(self, threat_id):
        '''reads the cross effect table to retrieve the cross effect of the threat'''
        threat_type = self.t1_type_id_conversion[threat_id] - 1
        cross_effect = []
        for i in range(2):
            if self.debug: print(f'channel_techs{i} = {self.channel_techs[i]}')
            if self.channel_techs[i] not in ('none', 'change'):
                if self.channel_techs[i] not in ('nj (n)', 'nj (mn)', 'nj (m)', 'nj (mw)', 'nj (w)'):
                    if self.debug: print(f'channel_target_threats[{i}] = {self.channel_target_threats[i]}')
                    if self.debug: print(f'threat_id = {threat_id}')
                    if self.debug: print(f'threat_type = {threat_type}')
                    ce_val = self.cross_effect[self.channel_target_threats[i]][threat_type]
                elif threat_type == 7 and self.channel_techs[i] == 'nj (n)':
                    # TODO: implement cross effect for noise jamming when examined threat id is 7
                    pass
                else:
                    # TODO: implement cross effect for noise jamming when examined threat id is not 7
                    pass
            
                cross_effect.append(ce_val)
            else:
                cross_effect.append(.0)
            
        return cross_effect
    
    def getChaffInterfernce(self):
        chaff_interfere = [.0, .0]
        for i in range(2):
            if self.channel_techs[2] == 'Dilution Chaff':
                chaff_interfere[i] = self.chaff_interference_factors[self.channel_techs[i]][1]
            elif self.channel_techs[2] == 'Distraction Chaff':
                chaff_interfere[i] = self.chaff_interference_factors[self.channel_techs[i]][0]
            else:
                chaff_interfere[i] = 1.
            
        return chaff_interfere
    
    def calculateTechniqueInteraction(self, threat_id, time_instance):
        '''Calculates the technique interactions for a specific threat'''
        # check if any of the active channels are not being used 
        if self.channel_techs[0] in ('none', 'change'):
            return (1, 1)
        
        if self.channel_techs[1] in ('none', 'change'):
            return (1, 1)
        
        # gets the jamming interaction for the active channel techniques for the time instance
        tech1_num = self.tech_dict[self.channel_techs[0]]
        tech2_num = self.tech_dict[self.channel_techs[1]]
        tech_effect_of_ch2_on_ch1 = 1 + (
            self.jamming_interaction_factors_IN[tech1_num][tech2_num] *
            self.threat_cross_effect[1] *
            self.agp2[time_instance][threat_id]            
            )
        tech_effect_of_ch1_on_ch2 = 1 + (
            self.jamming_interaction_factors_IN[tech2_num][tech1_num] *
            self.threat_cross_effect[0] *
            self.agp1[time_instance][threat_id]            
            )
        
        return (tech_effect_of_ch2_on_ch1, tech_effect_of_ch1_on_ch2)
    
    def calcCRFactors(self, threat_id, se_technique):
        '''Calculates the countermeasure resistance factors for active techniques 1 and 2'''
        if se_technique >= 0:
            crf = 1 - self.threat_cr_factors[threat_id]
        else:
            crf = 1 + self.threat_cr_factors[threat_id]
            
        return crf
    
    def calcRFJammingEffects(self, threat_id, time_instance):
        '''Calculates the jamming effects of active channels 1 and 2'''
        threat_radar_stage = self.stage_table_data[time_instance][threat_id + 1].lower()
        threat_radar_stage_val = self.stage_dict[threat_radar_stage]
        if self.debug: print(f'threat_radar_stage = {threat_radar_stage}')
        
        # get the cross effect values
        self.threat_cross_effect = self.getCrossEffect(threat_id)

        # gets the technique interaction values
        tech_interaction = self.calculateTechniqueInteraction(threat_id, time_instance)
        if self.debug: print(f'tech_interaction = {tech_interaction}')
    
        # gets the chaff interference
        chaff_interference = self.getChaffInterfernce()
        if self.debug: print(f'chaff_interference = {chaff_interference}')
        if self.debug: print(f'threat_cross_effect = {self.threat_cross_effect}')
        
        # calculate the jamming effect of active channel 1
        if self.channel_techs[0] not in ('none', 'change'):
            # gets the stage effectiveness values of the channels for the specified threat
            channel1_tech_val = self.tech_dict[self.channel_techs[0]]
            se_tech1 = self.stage_effectiveness[threat_radar_stage_val][channel1_tech_val]
            if self.debug: print(f'se_tech1 = {se_tech1}')
            
            # gets the countermeasure resistance factors
            cr_factors = self.calcCRFactors(threat_id, se_tech1)
            if self.debug: print(f'cr_factors1 = {cr_factors}')
            if self.debug: print(f'agp1 = {self.agp1[time_instance][threat_id]}')
            
            jamming_effect_ch1 = self.agp1[time_instance][threat_id] * se_tech1 * tech_interaction[0] * self.threat_cross_effect[0] * chaff_interference[0] * cr_factors
            
            # decoy factor
            if self.strategy_table_data[time_instance][-1].lower() not in ('none'):
                if float(self.strategy_table_data[time_instance][-1]) == 1.:
                    jamming_effect_ch1 = jamming_effect_ch1 * self.decoy_multiplication_factors[channel1_tech_val][threat_radar_stage_val]
        else:
            jamming_effect_ch1 = 0
            
        # calculate the jamming effect of active channel 2
        if self.channel_techs[1] not in ('none', 'change'):
            # gets the stage effectiveness values of the channels for the specified threa
            channel2_tech_val = self.tech_dict[self.channel_techs[1]]
            se_tech2 = self.stage_effectiveness[threat_radar_stage_val][channel2_tech_val]
            if self.debug: print(f'se_tech2 = {se_tech2}')
            
            # gets the countermeasure resistance factors
            cr_factors = self.calcCRFactors(threat_id, se_tech2)
            if self.debug: print(f'cr_factors2 = {cr_factors}')
            if self.debug: print(f'agp2 = {self.agp2[time_instance][threat_id]}')
            if self.debug: print(f'threat_cross_effect2 = {self.threat_cross_effect[1]}')
            
            jamming_effect_ch2 = self.agp2[time_instance][threat_id] * se_tech2 * tech_interaction[1] * self.threat_cross_effect[1] * chaff_interference[1] * cr_factors
        
            # decoy factor
            if self.strategy_table_data[time_instance][-1].lower() not in ('none'):
                if float(self.strategy_table_data[time_instance][-1]) == 2.:
                    jamming_effect_ch2 = jamming_effect_ch2 * self.decoy_multiplication_factors[channel2_tech_val][threat_radar_stage_val]
        else:
            jamming_effect_ch2 = 0
            
        return (jamming_effect_ch1, jamming_effect_ch2)
    
    def calcDistance(self, threat_id, time_instance, selector=0):
        '''Calculates either the slant or ground distance between the threat and platform'''
        # gets the x, y, and z co-ordinates of the platform
        xp = float(self.strategy_table_data[time_instance][0])
        yp = float(self.strategy_table_data[time_instance][1])
        zp = float(self.strategy_table_data[time_instance][2])
        
        # gets the x, y, and z co-ordinates of the threat
        xt = self.scenario_data[threat_id][0]
        yt = self.scenario_data[threat_id][1]
        zt = self.scenario_data[threat_id][2]
        
        if selector == 0:
            distance = sqrt((xt - xp)**2 + (yt - yp)**2 + (zt - zp)**2)
        else:
            distance = sqrt((xt - xp)**2 + (yt - yp)**2)
            
        return distance
    
    def getMissileDistance(self, threat_id, time_instance):
        '''Calculates the ground distance between the missile and platform'''
        # gets the x, y, and z co-ordinates of the platform
        xp = float(self.strategy_table_data[time_instance][0])
        yp = float(self.strategy_table_data[time_instance][1])
        zp = float(self.strategy_table_data[time_instance][2])
        
        # gets the x, y, and z co-ordinates of the missile when it was fired
        xm_init, ym_init, zm_init = self.in_guidance[threat_id][:3]
        
        # calculates the slant distance between the current position of the platform and that from which the missile was fired
        distance = sqrt((xm_init - xp)**2 + (ym_init - yp)**2 + (zm_init - zp)**2)
        
        # calculates the current missile position
        projectile_velocity = self.missile_velocity[threat_id] / 3600
        missile2platform_vector = [xp - xm_init, yp - ym_init, zp - zm_init]
        unit_vector = [element / distance  for element in missile2platform_vector]
        missile_position = [element * (2 * time_instance - self.in_guidance[threat_id][3]) * projectile_velocity for element in unit_vector]
        missile_position = [missile_position[i] + self.in_guidance[threat_id][i] for i in range(len(missile_position))]
        
        # gets the coordinates between the current position of the missile and platform
        xm = missile_position[0]
        ym = missile_position[1]
        zm = missile_position[2]
        
        # calculates the ground distance
        distance = sqrt((xm - xp)**2 + (ym - yp)**2 + (zm - zp)**2)
        
        return distance
    
    def checkThreatRadarStage(self, threat_id, time_instance):
        '''check if the threat is in guidance and check the type of weapon-system it uses'''
        in_guidance = False
        if self.stage_table_data[time_instance][threat_id + 1].lower() in ('guidance'):
            in_guidance = True
            
            if threat_id not in tuple(self.in_guidance.keys()):
                missile_init_coord = self.scenario_data[threat_id][:3]
                missile_init_coord.append(2 * time_instance)
                self.in_guidance.update({threat_id: missile_init_coord})
        elif threat_id in tuple(self.in_guidance.keys()) and in_guidance == False:
            self.in_guidance.pop(threat_id)

        return in_guidance
        
    def rangeJammingEffectRelationship(self):
        threat_ground_distance = []
        #print(len(strategy_table_data))
        
        # calculate the slant distance for all threats and time instances and capture the data in a file
        for idx, instance in enumerate(self.strategy_table_data):
            xp = float(instance[0])
            yp = float(instance[1])
            instance_ground_distance = []
            
            for threat in self.threat_data:
                xt = threat[0]
                yt = threat[1]
                
                # calculates the slant distance
                ground_distance = sqrt((xt - xp)**2 + (yt - yp)**2)

                if ground_distance <= threat[5]:
                    instance_ground_distance.append(1)
                else:
                    instance_ground_distance.append(0)
                    
            print(f'{2*idx}: {instance_ground_distance}')
            
        '''
        This shows that the ground based threats are only a threat to the platform once it gets within
        their radar range.        
        '''

def main():
    jamming_effect_values = []
    obj = Calculations()
    #obj.rangeJammingEffectRelationship()

    '''obj.updateStrategyData(21)
    print(obj.channel_techs)
    print(obj.channel_target_threats)
    je_values = obj.calcRFJammingEffects(1, 21)
    print(je_values)
    
    if obj.checkThreatRadarStage(1, 21):
        slant_distance = obj.getMissileDistance(1, 21)
    else:
        slant_distance = obj.calcDistance(1, 21)

    adjusted_jamming_effect = (je_values[0] + je_values[1]) * (1 + (slant_distance / 20)**2)
    print(adjusted_jamming_effect)'''
    
    for idx, data in enumerate(obj.strategy_table_data):
        obj.updateStrategyData(idx)
        #print(obj.channel_techs)
        #print(obj.channel_target_threats)
        je_vals = []
        
        for i in range(len(obj.scenario_data) - 4):
            threat_in_guidance = obj.checkThreatRadarStage(i, idx)
            
            # make an evaluation of whether the platform is within the threat radar range
            ground_distance = obj.calcDistance(i, idx, 1)
            if ground_distance > obj.scenario_data[i][5]:
                je_vals.append("{:.4f}".format(.0))
                continue
            
            je_values = obj.calcRFJammingEffects(i, idx)
            if je_values == (0., 0.):
                je_vals.append("{:.4f}".format(0.0))
                continue
            
            if threat_in_guidance:
                slant_distance = obj.getMissileDistance(i, idx)
            else:
                slant_distance = obj.calcDistance(i, idx)
    
            adjusted_jamming_effect = (je_values[0] + je_values[1]) * (1 + (slant_distance / 20)**2)
            
            if adjusted_jamming_effect > 1.0:
                je_vals.append("{:.4f}".format(1.0))
            elif adjusted_jamming_effect < -1.0:
                je_vals.append("{:.4f}".format(-1.0))
            else:
                je_vals.append("{:.4f}".format(adjusted_jamming_effect))
        
        jamming_effect_values.append(je_vals)
        print(f'{2 * idx}: {je_vals}')
        
    #print(jamming_effect_values)
    #print(f"the jamming effect values are {je} and the distance is {slant}")            
            
if __name__ == '__main__':
    main()
        