import pygame

# rgb colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE1 = (0, 0, 255)
GREY = (211, 211, 211)
PURPLE = (147, 112, 219)

BLOCK_SIZE = 8
LINE_THICKNESS = 2

class Symbols:
    def __init__(self, screen):
        self.display = screen
        
    def irUndetected(self, x, y):
        '''Draws the IR undetected symbol'''
        pygame.draw.circle(self.display, PURPLE, [x, y], 1, 0)
        pygame.draw.polygon(self.display, PURPLE, [[x, y - BLOCK_SIZE], [x - BLOCK_SIZE, y + BLOCK_SIZE], [x + BLOCK_SIZE, y + BLOCK_SIZE]], LINE_THICKNESS)
    
    def irGuidance(self, x, y):
        '''Draws the IR guidance symbol'''
        pygame.draw.circle(self.display, PURPLE, [x, y], 1, 0)
        pygame.draw.line(self.display, PURPLE, (x - BLOCK_SIZE, y), (x, y - BLOCK_SIZE), LINE_THICKNESS)
        pygame.draw.line(self.display, PURPLE, (x - BLOCK_SIZE, y), (x, y + BLOCK_SIZE), LINE_THICKNESS)
        pygame.draw.line(self.display, PURPLE, (x, y - BLOCK_SIZE), (x + BLOCK_SIZE, y), LINE_THICKNESS)
        pygame.draw.line(self.display, PURPLE, (x, y + BLOCK_SIZE), (x + BLOCK_SIZE, y), LINE_THICKNESS)
    
    def rfSearch(self, x, y):
        '''Draws the RF search symbol'''
        pygame.draw.line(self.display, PURPLE, (x - BLOCK_SIZE, y), (x + BLOCK_SIZE, y), LINE_THICKNESS)
        pygame.draw.line(self.display, PURPLE, (x, y + BLOCK_SIZE), (x, y - BLOCK_SIZE), LINE_THICKNESS)
    
    def rfAcquisition(self, x, y):
        '''Draws the RF acquisition symbol'''
        pygame.draw.line(self.display, PURPLE, (x - BLOCK_SIZE, y - BLOCK_SIZE), (x + BLOCK_SIZE, y + BLOCK_SIZE), LINE_THICKNESS)
        pygame.draw.line(self.display, PURPLE, (x - BLOCK_SIZE, y + BLOCK_SIZE), (x + BLOCK_SIZE, y - BLOCK_SIZE), LINE_THICKNESS)
        
    def rfTracking(self, x, y):
        '''Draws the RF tracking symbol'''
        pygame.draw.line(self.display, PURPLE, (x - BLOCK_SIZE, y), (x + BLOCK_SIZE, y), LINE_THICKNESS)
        pygame.draw.line(self.display, PURPLE, (x, y + BLOCK_SIZE), (x, y - BLOCK_SIZE), LINE_THICKNESS)
        pygame.draw.line(self.display, PURPLE, (x - BLOCK_SIZE, y - BLOCK_SIZE), (x + BLOCK_SIZE, y + BLOCK_SIZE), LINE_THICKNESS)
        pygame.draw.line(self.display, PURPLE, (x - BLOCK_SIZE, y + BLOCK_SIZE), (x + BLOCK_SIZE, y - BLOCK_SIZE), LINE_THICKNESS)
    
    def rfGuidance(self, x, y):
        '''Draws the RF guidance symbol'''
        pygame.draw.circle(self.display, PURPLE, [x, y], 1, 0)
        pygame.draw.rect(self.display, PURPLE, [x - BLOCK_SIZE, y - BLOCK_SIZE, 2*BLOCK_SIZE, 2*BLOCK_SIZE], LINE_THICKNESS)
    
    def jammer(self, x, y):
        '''Draws the jammer symbol'''
        pygame.draw.circle(self.display, PURPLE, [x, y], 1, 0)
        pygame.draw.polygon(self.display, PURPLE, [[x, y - BLOCK_SIZE], [x + BLOCK_SIZE, y - BLOCK_SIZE/2], 
                                                   [x + BLOCK_SIZE/2, y + BLOCK_SIZE], [x - BLOCK_SIZE/2, y + BLOCK_SIZE], 
                                                   [x - BLOCK_SIZE, y - BLOCK_SIZE/2]], LINE_THICKNESS)
    
    def JammerDecoy(self, x, y):
        '''Draws the jammer + decoy combination symbol'''
        pygame.draw.circle(self.display, PURPLE, [x, y], 1, 0)
        pygame.draw.polygon(self.display, PURPLE, [[x, y - BLOCK_SIZE], [x + BLOCK_SIZE, y - BLOCK_SIZE/2], 
                                                   [x + BLOCK_SIZE/2, y + BLOCK_SIZE], [x - BLOCK_SIZE/2, y + BLOCK_SIZE], 
                                                   [x - BLOCK_SIZE, y - BLOCK_SIZE/2]], 0)
            
    def flares(self, x, y):
        '''Draws the flares symbol'''
        pygame.draw.circle(self.display, PURPLE, [x, y], BLOCK_SIZE, 0)
    
    def chaff(self, x, y):
        '''Draws the chaff symbol'''
        pygame.draw.circle(self.display, PURPLE, [x, y], BLOCK_SIZE, LINE_THICKNESS)
        
    def agentSymbol(self, x, y):
        '''draws the agent symbol'''
        pygame.draw.circle(self.display, BLACK, [x, y], BLOCK_SIZE, 0)