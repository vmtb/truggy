# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:11:43 2024

@author: Marcos & Hichem
"""

class PIDController:
    def __init__(self, kP=0, kI=0, kD=0, valeurDesiree=0):
        self.kP = kP  # Constante proportionnelle
        self.kI = kI  # Constante intégrale
        self.kD = kD  # Constante dérivée
        self.valeurDesiree = valeurDesiree  # Consigne de l'angle ou position souhaitée
        self.integral = 0.0  # Terme intégral
        self.previousError = 0.0  # Erreur précédente

    def update(self, measuredValue, deltaTime):
        
        error = self.valeurDesiree - measuredValue  # Calcul de l'erreur
        self.integral += error * deltaTime  # Calcul du terme intégral
        derivative = (error - self.previousError) / deltaTime  # Calcul du terme dérivé
        
        
        # Calcul de la sortie du PID
        output = (self.kP * error) + (self.kI * self.integral) + (self.kD * derivative)

        self.previousError = error  # Mise à jour de l'erreur précédente

        return output


