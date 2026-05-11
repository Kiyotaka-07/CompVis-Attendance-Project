import cv2
import os
import numpy as np
from datetime import datetime

haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_path)

if face_cascade.empty():
  print("Unable to load the face cascade classifier XML file")
else:
  print("Succeed to load the face cascade classifier XML file")

def register_face(name, nim, samples=9):
    capture = cv2.VideoCapture(0)
    
    dataset_path = "dataset"
    user_path = os.path.join(dataset_path, name+"_"+nim)
    
    os.makedirs(user_path, exist_ok=True)
    
    count = 0
    
    print("Press 's' to save, 'q' to exit")
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        mirror_frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(mirror_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        MIN_FACE_SIZE = 120

        for (x, y, w, h) in faces:
            if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
                continue

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            
            cv2.rectangle(mirror_frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
        cv2.imshow("Register", mirror_frame)
        
        key = cv2.waitKey(1)
        
        if key == ord('s') and len(faces) > 0:
            count += 1
            cv2.imwrite(f"{user_path}/{count}.jpg", face)
            print(f"Foto {count} tersimpan")
        
        if key == ord('q') or count >= samples:
            break
    
    capture.release()
    cv2.destroyAllWindows()
    print("Registrasi selesai")