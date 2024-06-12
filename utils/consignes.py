# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:51:36 2024

@author: MARCOS VITOULEY & HICHEM HAMMA
"""
import cv2 
import matplotlib.pyplot as plt
import numpy as np 
import copy 
import math
from utils.retroprojection import Retroprojection
from utils.pid import PIDController
# Paramètres: largeur
#			: degré de changement de couleur
#           : Pente max = 100 (conformité)
#           : Seuillage
#           : Kernel morphologie mathématique
class ConsigneDeFrame: 
    
    def __init__(self, cameraInfo, debug=False, ligne_par_ligne=False): 
        
        Mint = cameraInfo.Mint()
        Mext = cameraInfo.Mext()
        self.debug = debug
        self.P = np.dot(Mint, Mext)  
        self.rt = Retroprojection(Mint, Mext)
        
        # self.pid_position = PIDController(kP=-0.5, kI=0.1, kD=0, valeurDesiree=0)
        # self.pid_angle = PIDController(kP=2, kI=0.1, kD=0, valeurDesiree=0)
        
        self.pid_position = PIDController(kP=-4, kI=0, kD=0.5, valeurDesiree=0)
        self.pid_angle = PIDController(kP=11, kI=0, kD=0.5, valeurDesiree=0)
         
        self.alpha = 0
        self.x_projete = 0
        
        self.data_x=[]
        self.data_th=[]
        
        # plt.figure(figsize=(20, 6)) 
        self.pos = 1
        self.ligne_par_ligne = ligne_par_ligne
        
        
    
    def debugTrace(self, img, title): 
        if self.debug and (self.ligne_par_ligne==False or (self.ligne_par_ligne and self.pos<8)):
            # cv2.imshow(title, img)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 4, self.pos), plt.title(title), plt.imshow(rgb_image)
            self.pos = self.pos+1
    
    def processLigne(self, img, morph=False): 
        self.img = img 
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.debugTrace(img, "Original")
        
        if(self.ligne_par_ligne==False):
        # self.img = cv2.GaussianBlur(self.img, (7, 5), 0) 
            self.img = cv2.medianBlur(self.img, 11) 
            self.debugTrace(self.img, "Flou gaussien")
            
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.debugTrace(self.img, "Gray Scale")  
            
            ret2,self.img = cv2.threshold(self.img, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            self.debugTrace(self.img, "Binarisation Adaptative") 
            
            if morph:
                kernel = np.ones((7,7), np.uint8)
                opening = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)
                self.img = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
                self.debugTrace(self.img, "Réduction du bruit") 
             
            self.s_only_gray = self.img # cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  
            self.debugTrace(self.s_only_gray, "Remener du binaire (Gray Scale)")
                
        self.pointsDeLimage = []
        
        # Parcourir les lignes 
        start = self.height-10
        
        for numLigne in  range(10, start-100, 80): #start
            position_ligne =   start - numLigne 
            image_filtree = self.img[position_ligne, :] 
            subimg = self.img[position_ligne-8:position_ligne, :]
            # print(position_ligne)
            if(self.ligne_par_ligne):
                self.debugTrace(subimg, "couper")
                subimg = cv2.medianBlur(subimg, 3) 
                self.debugTrace(subimg, "Flou gaussien")
                
                subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
                self.debugTrace(subimg, "Gray Scale")  
                
                ret2, subimg = cv2.threshold(subimg, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                self.debugTrace(subimg, "Binarisation Adaptative") 
                
                if morph:
                    kernel = np.ones((7,7), np.uint8)
                    opening = cv2.morphologyEx(subimg, cv2.MORPH_OPEN, kernel)
                    subimg = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
                    self.debugTrace(subimg, "Réduction du bruit") 
                 
                self.s_only_gray = subimg # cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)    
                image_filtree =self.s_only_gray[1, :] 
             
            
            # appliquer le gradient de Sobel pour la détection de contour
            gradient_x = np.absolute(cv2.Sobel(image_filtree, cv2.CV_64F, 1, 0, ksize=1)) #3
            gradient_y = np.absolute(cv2.Sobel(image_filtree, cv2.CV_64F, 0, 1, ksize=1))  
            gradient_total = np.sqrt(gradient_x**2 + gradient_y**2).astype(np.uint8)  
            
            # seuiller 
            gradient_total = cv2.GaussianBlur(gradient_total, (5, 5), 5);
            _, seuillee = cv2.threshold(gradient_total, 5, 255, cv2.THRESH_BINARY) #35
            gradient = np.gradient(seuillee.reshape(-1)) 
             
            # plt.figure(figsize=(8, 5))
            # plt.title(f'Ligne {position_ligne}')
            # plt.plot(gradient_total, color='black')
            
            coupleIV = [(indice, valeur) for indice, valeur in enumerate(gradient) if valeur != 0]
            couple4 = [] 
            for couple in coupleIV: 
                if not couple4 or (couple4[-1][1] + couple[1] <= 0 and couple4[-1][1] != couple[1]): 
                    couple4.append(couple)
             
            cv2.line(self.img , (1, position_ligne), (1919, position_ligne), (0,255,0), thickness=5)
 
            if(len(couple4)%2!=0):
                couple4.insert(0, (0, -couple4[0][1]))
            taille = 2
            result = [item for item in couple4 if item[1] > 0]
            points = [(result[i], result[i + 1]) for i in range(len(result) - 1)]# ranger deux à deux
            
            self.couple4 = couple4
            if(len(couple4)!=4  and len(couple4)!=6): # || != 4 ou > 8
                continue
            
            for idx, p in enumerate(points):
                if len(p)!=taille and idx != len(points) - 1:
                    print(f"Erreur à la ligne {position_ligne} on a {len(p)} et {taille}\n")
                    continue
                lg = p[0][0]
                ld = p[-1][0]
                
                #Test de largeur  
                lgm = self.rt.generatePoint(lg, position_ligne, 0, 1) #generatePointFromYW
                ldm = self.rt.generatePoint(ld, position_ligne, 0, 1)
                d = np.sqrt((lgm[1] - ldm[1])**2 + (lgm[0] - ldm[0])**2  + (lgm[2] - ldm[2])**2)
                U = ((lg+ld)//2, position_ligne) 
                
                # print(f"\nlg_m={lgm} et \nld_m={ldm} et \nd={d} pour la ligne {U}")
                if self.testLargeur(position_ligne, d):
                    # print(f"d={d} pour la ligne {U}")
                    if not self.pointsDeLimage or self.testConformiteX(abs(self.pointsDeLimage[-1][0][0] - U[0]), position_ligne): 
                        self.pointsDeLimage.append([U, self.rt.generatePoint(U[0], U[1], 0, 1)])
                        break 
                if(len(self.pointsDeLimage)==3): 
                    break
        
        if(len(self.pointsDeLimage)<=1): 
            self.pointsDeLimage = []
        return self.pointsDeLimage
    
    def afficherLigneRouge(self): 
        # if not self.img[0]: 
        #     print(f"Vous devez d'abord process l'image")
        if(len(self.img.shape)>2 and self.img.shape[2]==3):
            self.pos = self.pos-2
            self.seuillee_color = self.img
        else:
            self.seuillee_color = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        
        if(self.debug):
            for i in range(len(self.pointsDeLimage)-1):
                cv2.line(self.seuillee_color, self.pointsDeLimage[i][0], self.pointsDeLimage[i+1][0], (0,255,255), thickness=5)
            self.debugTrace(self.seuillee_color, "Détection")
            
        # Modéliser les points par une droite
        B = []
        A = []
        for pos in self.pointsDeLimage: 
            B.append([pos[1][0], 1]) #[x, 1]
            A.append([pos[1][2]])    #[z]
        B = np.array(B)
        A = np.array(A)
        BTB_inv = np.linalg.inv(np.dot(B.T, B)) 
        AB = np.dot(np.dot(BTB_inv, B.T), A) # print(AB) #z = ax+b
        if(AB[0]==0): 
            return self.seuillee_color, self.x_projete 
        self.x_projete = -AB[1]/AB[0]
        self.z_projete = AB[1]
        angle_rad = math.degrees((math.atan(self.z_projete/self.x_projete)))
        self.alpha =self.sign(angle_rad)* (90 - abs(angle_rad))
        print(f"z = {self.z_projete}")
        print(f"all = {self.alpha}")
        if(self.alpha>90): 
            self.alpha-=2*180 
        elif(self.alpha<-90): 
            self.alpha+=2*180  
        
        # Tracer la ligne modélisée
        p1z = 0.4 
        p2z= 1.5
        p1x = (p1z-AB[1][0])/AB[0][0]
        p2x = (p2z-AB[1][0])/AB[0][0]
        p1 = np.dot(self.P, [p1x, 0,p1z, 1])
        p1 = p1//p1[2] 
        p2 = np.dot(self.P, [p2x, 0, p2z, 1])
        p2 = p2//p2[2] 
        cv2.line(self.seuillee_color, (int(p1[0]), int(p1[1])),  (int(p2[0]), int(p2[1])), (0,0,255), thickness=5)

        self.debugTrace(self.seuillee_color, "Avec la régression")
        # cv2.imshow("Composante Green Gradient", self.s_only_gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imshow('Détection', self.seuillee_color)
        return self.seuillee_color, self.x_projete
    
    
    def recupererTheta(self, dtime):  
        
        corr_position = self.pid_position.update(-self.x_projete*100, dtime)
        corr_angle = self.pid_angle.update(-self.alpha, dtime) 
        
        self.data_x.append(self.x_projete)
        self.data_th.append(self.alpha)
        deltaTheta = corr_position + corr_angle
        
        # deltaTheta = (self.kx * 100) * self.x_projete + self.ka * self.alpha 
        
        return deltaTheta #+I+D
    
    def sign(self, number):
       if (number>0):
          return 1
       elif (number<0):
          return -1
       else:
          return 0 

    def plotCourbe(self): 
        rmse_ = self.calculate_rmse(self.data_th,  np.zeros(len(self.data_th)))
        rmse_x= self.calculate_rmse(self.data_x, np.zeros(len(self.data_x)) )
        print(f"RMSE THETA {rmse_}")
        print(f"RMSE X {rmse_x}")
        plt.figure(figsize=(12, 6))
        plt.plot(self.data_th, color='black')
    
    def calculate_rmse(self, predictions, targets):
        differences = predictions - targets                 
        squared_differences = differences ** 2             
        mean_squared_differences = np.mean(squared_differences)
        rmse = np.sqrt(mean_squared_differences)               
        return rmse
             
 
    def testLargeur(self, position_ligne, d): 
        
        # if position_ligne > self.height/2: 
        #     return d<0.055 and d>0.025
        # else: 
        #     return d<0.085 and d>0.035
        
        if position_ligne > self.height/2: 
            return d<0.075 and d>0.025
        else: 
            return d<0.085 and d>0.035

    def testConformiteX(self, diff, position_ligne):  
        # return True
        # print(f"diff={diff}")
        return diff < 55
    
        if position_ligne > 2*self.height/3: 
            return diff < 70
        elif position_ligne > self.height/2: 
            return diff < 80
        elif position_ligne > self.height*1/3: 
            return diff < 90
        else: 
            return diff < 100
         
        
    def afficher_histogramme(self, img):    
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])  
        plt.figure(figsize=(8, 5))
        plt.title('Histogramme de l\'image')
        plt.xlabel('Niveaux de gris')
        plt.ylabel('Fréquence')
        plt.plot(hist, color='black')
        plt.xlim([0, 256])
        plt.show() 
            
         
        