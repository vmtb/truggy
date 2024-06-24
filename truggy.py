# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:12:58 2024

@author: MARCOS VITOULEY & HICHEM HAMMA
"""
import cv2
import time
from utils.camerainfo import CameraInfo
import numpy as np
from utils.consignes import ConsigneDeFrame 
import os

os.makedirs('data/frames', exist_ok=True)  
""" 
# Le servo est sur le PIN 12 (ligne au dessus de la fonction traitementTempsReel())
# Si possible, agir sur le traitement ligne_par_ligne (False ou True)
# ou sur la fonction testDeLargeur dans le fichier consignes.py

________________________________________________________________________________________
"""


# INITIALISATION
c = CameraInfo(hauteur=300, theta=27)
consigne = ConsigneDeFrame(c, ligne_par_ligne=True) 
frames = []
cte = 0.1 #On suppose commander 1fois chaque 0.1s  
waitingTime = 0.001 #Pour ralentir le traitement dans le cas d'une vidéo


from picamera.array import PiRGBArray
from picamera import PiCamera
from piservo import Servo
myservo = Servo(12)
def traitementTempsReel():
    # Initialisation de la PiCamera
    myservo.write(90)
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.start_preview()
    camera.framerate = 24  # Réglage du taux de rafraîchissement
    rawCapture = PiRGBArray(camera, size=(640, 480)) 
    # Laisser le temps à la caméra de s'initialiser
    time.sleep(0.1)
    nbreFrame = 0
     
    
    lastTime = time.time()
    
    # Boucle principale
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True): 
        image = frame.array  
        
        fin = time.time() 
        if (fin - lastTime) >= cte:
            fr = commande(image)
            nbreFrame = nbreFrame + 1
            if len(fr)!=0 and nbreFrame%10==0: 
                lastTime = time.time()
                cv2.imwrite(f'data/frames/{nbreFrame}.jpg', fr)

        
        # Effacer le flux pour la prochaine image
        rawCapture.truncate(0)  
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    camera.stop_preview()
    myservo.write(90)

def traitementVideo(): 
    video_path = "vid/cool3.h264"
    video_capture = cv2.VideoCapture(video_path)
    debut = time.time()
    lastTime = time.time()
    nbreFrame = 0
    
    while True:
        success, frame = video_capture.read()
        fin = time.time() 
        if not success:
            print("Durée d'exécution:", (fin-debut), "secondes")
            break;  
            
        if (fin - lastTime) >= cte:
            fr = commande(frame) 
            if len(fr)!=0: 
                lastTime = time.time() 
                frames.append(fr)
                cv2.imwrite(f'data/frames/{nbreFrame}.jpg', frame)
                cv2.imwrite(f'data/frames/{nbreFrame}r.jpg', fr)
                nbreFrame = nbreFrame+1
        time.sleep(waitingTime)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs as well
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (width, height)) 
    for frame in frames: 
        out.write(frame)
    out.release()
    consigne.plotCourbe()
             


def write_text_on_frame(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 0), thickness=2):
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_origin = position[0], position[1] + text_height 
    cv2.rectangle(frame, (position[0], position[1]), (position[0] + text_width, position[1] + text_height), (255,255,255), -1)
    cv2.putText(frame, text, text_origin, font, font_scale, color, thickness) 
    return frame

def commande(frame): 
    coordinatesOfRedLines = consigne.processLigne(frame, True) 
    if len(coordinatesOfRedLines)>2: 
        try:
            fr, x_projete = consigne.afficherLigneRouge() 
            deltaTheta = consigne.recupererTheta(cte)
            theta = int(90 + deltaTheta)
             
            bornSup = 130
            bornInf = 50
            if theta >bornSup: 
                theta = bornSup
            if theta < bornInf:
                theta = bornInf
                
            myservo.write(theta)
            al = round(float(consigne.alpha), 2) 
            dt = round(float(deltaTheta), 2) 
            x_proj_formatted =round(float(x_projete), 3)*100
            fr = write_text_on_frame(fr,  f"th={int(theta)}, DT={dt}, al={al}, x={x_proj_formatted}", (40, frame.shape[0]-50))  
            return fr 
        except Exception as e:
            print(e)
            pass 
    return []


if __name__=="__main__": 
    # traitementVideo() 
    traitementTempsReel()

# Libérer les ressources 
cv2.destroyAllWindows()


