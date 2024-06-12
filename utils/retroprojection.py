# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:43:45 2024

@author: Marcos VITOULEY
"""
import numpy as np 
import matplotlib.pyplot as plt

class Retroprojection:
    
    def __init__(self, Mint, Mext): 
        self.Mint = Mint
        self.Mext = Mext 
        self.P = np.dot(Mint, Mext)  
        self.P_t = np.transpose(self.P)
#         print(self.Mint)
#         print(self.Mext) 
#         print(self.P)
#         print(np.dot(self.P, self.P_t))
#         print(self.P_t@self.P)
        self.PTP_inv = np.zeros((4,4)) # np.linalg.inv(self.P_t@self.P)
        
    
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

    # z est fonction de U[3]
    # u,v
    # zp = 
    # zp = az+b
    def generatePointFromZY(self, u, v, z, y): 
        T = (self.PTP_inv)@(self.P_t@np.array([u, v, 1])) 
        T[1] = y 
        T[2] = z #On fixe le z à 500, ceci entraine v= 3w+b1  et u =3x+b2 
              
        
        Y = []
        V = [] 
        for i in range(2):  
            T[3]=T[3]+i
            Y.append(T[3])
            prod = np.dot(self.P, np.array(T))  
            UV = prod if prod[2]==0 else prod/prod[2]
            V.append(UV[1]) 
        
        a1 = (V[1]-V[0])/(Y[1]-Y[0])
        b1 = V[1]-a1*Y[1] 
        # print(f" la droite v={a1}w+{b1}")
        w = (v-b1)/a1 
        T[3] = w 
        
        X = []
        U = [] 
        for i in range(2):  
            T[0]=T[0]+i
            X.append(T[0])
            prod = np.dot(self.P, np.array(T))  
            UV = prod if prod[2]==0 else prod/prod[2]
            U.append(UV[0])  
        a2 = (U[1]-U[0])/(X[1]-X[0])
        b2 = U[1]-a2*X[1] 
        print(f" la droite u={a2}x+{b2}")
        x = (u-b2)/a2 
        T[0] = x  
        return T 
    
    def sensIsCroissant(self, T):
        T[2] = 10
        prod = np.dot(self.P, np.array(T))
        UV = prod if prod[2]==0 else prod/prod[2] 
        vDebut = UV[1]
        
        T[2] = 50 
        prod = np.dot(self.P, np.array(T))
        UV = prod if prod[2]==0 else prod/prod[2] 
        vFin = UV[1] 
        return vFin > vDebut
        
     
    def heuristiqueForZ(self, v, T ):
        epsilon = 0.00001
        error = 1
        borneInf = 0
        borneSup = 50
        n = 1
        ERROR = []
        croissant = self.sensIsCroissant(T)
        while(error>epsilon):
            middle =  borneInf + (borneSup-borneInf)/2 
            # print(f"Entre {borneInf} et {borneSup}")
            T[2] = middle 
            prod = np.dot(self.P, np.array(T))
            UV = prod if prod[2]==0 else prod/prod[2] 
            error = np.abs(UV[1] - v) / v if v != 0 else 0
            ERROR.append(error)
            # print(UV[1])
            if(croissant):
                if(v>UV[1]):
                    borneInf =  middle   
                else:  
                    borneSup = middle
            else: 
                if(v<UV[1]):
                    borneInf = middle
                else:  
                    borneSup =  middle   
            
        # plt.figure(figsize=(8, 5)) 
        # plt.title("v en fonction de y, z fixé à la limite de stabilité")
        # plt.plot(ERROR, color='black')  
        # plt.show() 
        
        return T[2]
        
    def generatePointFromYW(self, u, v, y=-0.58, w=1): 
        T = (self.PTP_inv)@(self.P_t@np.array([u, v, 1])) 
        # print(f"Directement de l'inverse {T}") 
        T[1] = y
        T[3] = w #w=1 #On fixe le z à 500, ceci entraine v= 3y+b1  et u =3x+b2, u3=az+b3 
      
        # print(f"Actuel {T}") 
        # algo en complexité log(N)        
        # T[2] =0
        # Z = []
        # U3 = [] 
        # for i in range(15):  
        #     T[2]=T[2]+0.5
        #     Z.append(T[2])
        #     prod = np.dot(self.P, np.array(T))   
        #     UV = prod if prod[2]==0 else prod/prod[2]  
        #     print(T[2])
        #     print(UV[1])
        #     U3.append(UV[1]) 

        # a3 = (U3[1]-U3[0])/(Z[1]-Z[0])
        # b3 = U3[1]-a3*Z[1] 
        # print(f" la droite u3={a3}z+{b3}")
        # # z = (924.5-b3)/a3 
        # # T[2] = z 
        # plt.figure(figsize=(8, 5)) 
        # plt.title(f"v en fonction de z, y étant connu = {y}")
        # plt.axhline(y=v, color='red', linestyle='--', label='Horizontal Line')  # Add a horizontal line at y=100
        # plt.axvline(x=0.69, color='red', linestyle='--', label='Horizontal Line')  # Add a horizontal line at y=100
        
        # plt.plot(Z, U3, color='black')  
        # plt.show() 
         
        T[2] = self.heuristiqueForZ(v, T) 
        
        # Y = []
        # V = [] 
        # for i in range(2):  
        #     T[1]=T[1]+i
        #     Y.append(T[1])
        #     prod = np.dot(self.P, np.array(T))   
        #     UV = prod if prod[2]==0 else prod/prod[2]
        #     V.append(UV[1]) 
        
        # a1 = (V[1]-V[0])/(Y[1]-Y[0])
        # b1 = V[1]-a1*Y[1] 
        # # print(f" la droite v={a1}w+{b1}")
        # y = (v-b1)/a1 
        # T[1] = y 
        
        X = []
        U = [] 
        for i in range(2):  
            T[0]=T[0]+i
            X.append(T[0])
            prod = np.dot(self.P, np.array(T))  
            UV = prod if prod[2]==0 else prod/prod[2] 
            U.append(UV[0])  
        a2 = (U[1]-U[0])/(X[1]-X[0])
        b2 = U[1]-a2*X[1] 
        # print(f" la droite u={a2}x+{b2}")
        # plt.figure(figsize=(8, 5)) 
        # plt.plot(X, U, color='black')  
        # plt.title("u en fonction de x, z fixé à la limite de stabilité")  
        # plt.axhline(y=u, color='red', linestyle='--', label='Horizontal Line')  # Add a horizontal line at y=100

        x = (u-b2)/a2 
        T[0] = x  
        
        return T 
    
    def generatePointFromZW(self, u, v, z, w=1): 
        T = (self.PTP_inv)@(self.P_t@np.array([u, v, 1])) 
        T[2] = z      #On fixe le z à 500, ceci entraine v= 3y+b1  et u =3x+b2, u3=az+b3 
        T[3] = w      #w=1 
         
        
        Y = []
        V = [] 
        for i in range(2):  
            T[1]=T[1]+i
            Y.append(T[1])
            prod = np.dot(self.P, np.array(T))   
            UV = prod if prod[2]==0 else prod/prod[2]
            V.append(UV[1]) 
        
        a1 = (V[1]-V[0])/(Y[1]-Y[0])
        b1 = V[1]-a1*Y[1] 
        # print(f" la droite v={a1}w+{b1}")
        y = (v-b1)/a1 
        T[1] = y 
        
        X = []
        U = [] 
        for i in range(2):  
            T[0]=T[0]+i
            X.append(T[0])
            prod = np.dot(self.P, np.array(T))  
            UV = prod if prod[2]==0 else prod/prod[2]
            U.append(UV[0])  
        a2 = (U[1]-U[0])/(X[1]-X[0])
        b2 = U[1]-a2*X[1] 
        # print(f" la droite u={a2}x+{b2}")
        x = (u-b2)/a2 
        T[0] = x  
        
        return T 
        
        
        
        
        
        
        
        
        
        
        

# T[2] =0
# Z = []
# U3 = [] 
# for i in range(15):  
#     T[2]=T[2]+0.5
#     Z.append(T[2])
#     prod = np.dot(self.P, np.array(T))   
#     UV = prod if prod[2]==0 else prod/prod[2]  
#     print(T[2])
#     print(UV[1])
#     U3.append(UV[1]) 

# a3 = (U3[1]-U3[0])/(Z[1]-Z[0])
# b3 = U3[1]-a3*Z[1] 
# print(f" la droite u3={a3}z+{b3}")
# # z = (924.5-b3)/a3 
# # T[2] = z 
# plt.figure(figsize=(8, 5)) 
# plt.title("v en fonction de y, z fixé à la limite de stabilité")
# plt.plot(Z, U3, color='black')  
# plt.show() 