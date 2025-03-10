import pygame
import numpy as np
from collections import namedtuple
from symbols import Symbols
from enum import Enum
from time import sleep
from math import sqrt

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
GREY = (211, 211, 211)

SPEED = 10

class Environment():
    def __init__(self, threat_data_fname, flight_data_fname, width, height):
        self.threat_data = []
        self.flight_data = []
        self.iteration = 1
        self.width = width
        self.height = height
        self.x_offset = int((self.height % 100) / 2)
        self.y_offset = int((self.height % 100) / 2)
        
        # converts threats x-y co-ordinates data read from a file line by line into a list and then converts it into floats
        with open(threat_data_fname, 'r') as file:
            for line in file:
                data = line.split(';')                                          # converts the data into a list using the separator ';'
                data.pop()                                                      # removes the last item from the list
                data = [float(ele) for ele in data]                             # converts the line of data from str to float elements
                self.threat_data.append(data)
                
        # reads the waypoint data from a file and converts it into floats
        with open(flight_data_fname, 'r') as fd_file:
            for line in fd_file:
                data = line.split(';')
                data.pop()
                data = [float(ele) for ele in data]
                self.flight_data.append(data)
                
        self.__generateFlightData()
            
        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        self.env_symbol = Symbols(self.display)
        pygame.display.set_caption('Countermeasure Resource Allocation')
        self.clock = pygame.time.Clock()
        self.reset()                                                            # initialises the environment state
        
        _ = self.calc3dDistance()
        
    def reset(self):
        '''Initialises the environment state'''
        self.threats = []
        
        # initialise the threats co-ordinates
        for threat in self.threat_data:
            self.threats.append(Point(self.x_offset + 30*threat[0], self.height - self.y_offset - 30*threat[1]))
        
        # initialise agent coordinates
        self.env_agent = Point(self.x_offset + self.flight_data[0][0], self.height - self.y_offset - 30 * self.flight_data[0][1])
        
    def __generateFlightData(self):
        '''Generates complete flight data given the flight data'''
        self.flight_data_comp = []
        for i in range(1, len(self.flight_data)):
            x1, y1 = self.flight_data[i-1][0], self.flight_data[i-1][1]
            x2, y2 = self.flight_data[i][0], self.flight_data[i][1]
            z1 = self.flight_data[i-1][2]
            num_points = int((self.flight_data[i][3] - self.flight_data[i-1][3]) / 2)
            
            if i != len(self.flight_data) - 1:
                x_vals = np.linspace(x1, x2, num_points, endpoint=False)
                y_vals = np.linspace(y1, y2, num_points, endpoint=False)
                z_vals = np.linspace(z1, z1, num_points, endpoint=False)
            else:
                x_vals = np.linspace(x1, x2, num_points, endpoint=True)
                y_vals = np.linspace(y1, y2, num_points, endpoint=True)
                z_vals = np.linspace(z1, z1, num_points, endpoint=True)
            
            self.flight_data_comp.extend(zip(x_vals, y_vals, z_vals))
        
        print(self.flight_data_comp)
        
    def calcSlantDistance(self):
        '''calculates the 3 distance using the current coordinates of the platform and those of all the threats'''
        distance_from_threat = []
        distance_from_threat = [
            sqrt((self.flight_data_comp[self.iteration][0] - val[0]) ** 2 + 
                 (self.flight_data_comp[self.iteration][1] - val[1]) ** 2 +
                 (self.flight_data_comp[self.iteration][2] - val[2]) ** 2 )
            for val in self.threat_data
            ]
        
        print(distance_from_threat)
        return distance_from_threat
            
    def __renderGrid(self):
        '''draws the grid on the user interface'''    
        grid_space = 150
        # draws the vertical grid lines
        for i in range(int(self.width / grid_space) + 1):
            if i == 0:
                pygame.draw.line(self.display, GREY, (self.y_offset, 0), (self.y_offset, self.height), 2)
            else:
                pygame.draw.line(self.display, GREY, (self.y_offset + i * grid_space, 0), 
                                 (self.y_offset + i * grid_space, self.height), 2)
            
        # draws the horizontal grid lines
        for i in range(int(self.height / grid_space) + 1):
            if i == 0:
                pygame.draw.line(self.display, GREY, (0, self.x_offset), (self.width, self.x_offset), 2)
            else:
                pygame.draw.line(self.display, GREY, (0, self.x_offset + i * grid_space), 
                                 (self.width, self.x_offset + i * grid_space), 2)
    
    def move(self):
        '''Moves the agent through the threats and determines the action to take'''
        self.env_agent = Point(self.x_offset + 30 * self.flight_data_comp[self.iteration][0], 
                               self.height - self.y_offset - 30 * self.flight_data_comp[self.iteration][1])
        self.iteration += 1
    
    def play_step(self):
        # 2. Move agent
        #self.move()
        
        # 5. updates user interface and clock
        self._update_interface()
        self.clock.tick(SPEED)
            
    def _update_interface(self):
        '''Updates the user interface every iteration'''
        self.display.fill(WHITE)
        
        self.__renderGrid()
        self.env_symbol.agentSymbol(self.env_agent.x, self.env_agent.y)
        
        # TODO: set the symbol according to the state of the environment
        for idx, pt in enumerate(self.threats):
            if idx < 8:
                self.env_symbol.rfSearch(pt[0], pt[1])
            else:
                self.env_symbol.irUndetected(pt[0], pt[1])
            
        pygame.display.flip()
            
def main():
    td_name = 'scenario_data.txt'
    fd_name = 'waypoint_data.txt'
    w = 640
    h = 640
    
    obj = Environment(td_name, fd_name, w, h)
    while True:
        obj.play_step()
        #sleep(0.5)
    
if __name__ == '__main__':
    main()
