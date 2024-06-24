# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:43:45 2024

@author: Marcos VITOULEY
"""
import numpy as np 
import matplotlib.pyplot as plt

#Obtenir les points 3D à partir des Points 2D(u,v) sachant que y = 0
#Mint: matrice intrinsèque 
#Mext: matrice extrinsèque

class Retroprojection:
    
    def __init__(self, Mint, Mext): 
        self.Mint = Mint
        self.Mext = Mext 
        self.P = np.dot(Mint, Mext)  
        self.P_t = np.transpose(self.P) 
        self.PTP_inv = np.zeros((4,4)) 
        
    
    #Obtenir les points 3D à partir des Points 2D(u,v) sachant que y = 0
    def generatePoint(self, u, v, w=0, y=0): 
        aU = self.Mint[0][0]
        aV = self.Mint[1][1]
        Uo = self.Mint[0][2]
        Vo = self.Mint[1][2]
        h = self.Mext[1, 3]
        a = self.Mext[1, 1]
        b = self.Mext[2, 1]
        z = (-h*aV)/(-b*aV+a*Vo-a*v)
        w = a*z
        x = (w*u-Uo*a*z)/aU 
        return [x,0,z,1]   
        
        
        
