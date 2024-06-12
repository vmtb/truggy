import numpy as np
import math 

class CameraInfo: 
     
    #1024x768  width=1920, height = 1080
    #72ppp = 72 px par pouce
    #1pouce = 25.4 mm
    #du = 0.024
    def __init__(self, fx=50, fy=50, width=1920, height=1080, du=2.83, dv=2.83, theta=0, hauteur=300):
        self.fx=fx/1000 #de mm en m
        self.fy=fy/1000 #de mm en m
        
        self.width = width #en px
        self.height = height #en px
        
        self.du = (du*1000) # px/mm en px/m
        self.dv = (dv*1000)  # px/mm en px/m
         
        
        self.theta = np.radians(theta) #autour de x en degré 
        self.phi = np.radians(0) #autour de y en degré 
        self.gamma =  np.radians(0) #autour de z en degré 
        
        self.hauteur = hauteur/1000 #en m   
    
    
    def Mint(self): 
        alphaU = self.fx*self.du
        alphaV = self.fy*self.dv 
        ### 
        alphaU = 1503.6
        alphaV = 1504.9
        # O = np.array([932.5, 559.1])
        # alphaU = 505.1099
        # alphaV = 505.4937
        O = np.array([310.5698, 247.0657])
        # O = np.array([990, 540])
        self.K = np.array([[alphaU, 0, O[0]], [0, alphaV, O[1]], [0, 0, 1]]) #-0.5174
        return np.hstack([self.K, np.array([[0],[0],[0]])])
    
    def Mdist(self): 
        return np.array([0.1425, -0.2139, 0, 0])
    
    def K(self):
        return self.K
    # Le repère caméra est orienté d'un angle teta par rapport à l'axe X du repère monde
    # L'origine du repère caméra a pour coordonnées O'(0, y, 0) dans le repère monde
    def Mext(self):   
        R1 = np.array([
            [1, 0, 0],
            [0, np.cos(self.theta), -np.sin(self.theta)],
            [0, np.sin(self.theta), np.cos(self.theta)]
        ])
           
        R2 = np.array([
            [np.cos(self.phi), 0, np.sin(self.phi)], 
            [0, 1, 0],
            [-np.sin(self.phi), 0, np.cos(self.phi)]
        ])
           
        R3 = np.array([
            [np.cos(self.gamma), -np.sin(self.gamma), 0],
            [np.sin(self.gamma), np.cos(self.gamma), 0],
            [0, 0, 1],
        ])
 
        self.R = R3@R2@R1
        self.T = np.array([
            [0],
            [self.hauteur],
            [0],
        ])
        
        transformation_matrix = np.vstack([np.hstack([self.R, self.T]), [0, 0, 0, 1] ]) # 
        return transformation_matrix
    
    def R(self):
        return self.R

    def T(self):
        return self.T
    
def cos(n):
    return np.cos(n)

def sin(n):
    return np.cos(n)
