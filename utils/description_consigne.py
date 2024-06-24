# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:22:37 2024

@author: Marcos VITOULEY

Ce fichier contient l'ensemble des méthodes utilisées dans Consignes.py et une petite description
"""


"""
# Controller
def __init__(self, cameraInfo, debug=False, ligne_par_ligne=False): 

# Fonction utilitaire pour afficher des figures des images traitées pour debugger
def debugTrace(self, img, title): 


# Pour traiter une image
# Si morph est True, l'étape de morphologie mathématique ne sera pas ignoré
def processLigne(self, img, morph=False):  


# Pour modéliser les points par la méthode des MCO 
# et afficher la droite des MCO en rouge sur l'image 
def afficherLigneRouge(self): 
 

# Application du PID pour corriger la position et l'orientation
def recupererTheta(self, dtime):   

# Fonction utilitaire pour avoir le signe d'un nombre
def sign(self, number): 

# Fonction utilitaire pour ploter l'évolution des corrections sur l'angle et calculer
# les ereurs RMSE
def plotCourbe(self):  


# Fonction utilitaire pour calculer les RMSE
def calculate_rmse(self, predictions, targets): 


# Fonction utilitaire pour faire le test de largeur
def testLargeur(self, position_ligne, d):  


# Fonction utilitaire pour faire le test de conformité
def testConformiteX(self, diff, position_ligne): 


# Fonction utilitaire pour sauvegarder les variables
def saveVar(self, var, name):


# Fonction utilitaire pour afficher l'histogramme 
def afficher_histogramme(self, img):  

    """