# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:11:30 2024

@author: MON PC
"""


import cv2 
import time 
from utils.camerainfo import CameraInfo
from utils.consignes import ConsigneDeFrame 
import time 
from matplotlib import pyplot as plt
import numpy as np



liste_images=[1] 

for i in liste_images:
    frame = cv2.imread(f"{i}.jpg", )
    
    c = CameraInfo(hauteur=300, theta=25)
    consigne = ConsigneDeFrame(c, debug=True, ligne_par_ligne=False)
    
    # coordinatesOfRedLines = consigne.processLigne(frame) 
    # ffr,_ = consigne.afficherLigneRouge() 
    
    coordinatesOfRedLines = consigne.processLigne(frame, True) 
    closing, _ = consigne.afficherLigneRouge() 

# cv2.imshow("Composante Green Gradient Seuille", ffr)
cv2.waitKey(0)
cv2.destroyAllWindows()