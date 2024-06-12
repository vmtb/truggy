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

liste=[620] #,2,3,4,5,6,7,8,9,10,11]
for i in liste:
    frame = cv2.imread(f"{i}.jpg", )
    
    c = CameraInfo(hauteur=300, theta=25)
    consigne = ConsigneDeFrame(c, debug=True, ligne_par_ligne=False)
    
    # coordinatesOfRedLines = consigne.processLigne(frame) 
    # ffr,_ = consigne.afficherLigneRouge() 
    
    
    coordinatesOfRedLines = consigne.processLigne(frame, True) 
    closing, _ = consigne.afficherLigneRouge()
    
    # _, binary = cv2.threshold(ffr, 127, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((1,5), np.uint8)
    # opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1), plt.title('Sans morphologie mathématique'), plt.imshow(cv2.cvtColor(ffr, cv2.COLOR_BGR2RGB))
    # # plt.subplot(1, 3, 2), plt.title('Binaire'), plt.imshow(binary, cmap='gray')
    # plt.subplot(1, 2, 2), plt.title('Après la réduction du bruit'), plt.imshow(closing, cmap='gray')
    # plt.show()

# cv2.imshow("Composante Green Gradient Seuille", ffr)
cv2.waitKey(0)
cv2.destroyAllWindows()