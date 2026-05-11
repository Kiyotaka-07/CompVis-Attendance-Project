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

def train_model():
    dataset_path = "dataset"
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces = []
    labels = []
    label_dict = {}
    
    label_id = 0
    
    for user in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user)
        
        if not os.path.isdir(user_path):
            continue
        
        label_dict[label_id] = user
        
        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            faces.append(img)
            labels.append(label_id)
        
        label_id += 1
    
    recognizer.train(faces, np.array(labels))
    recognizer.save("face_model.yml")
    
    print("Training selesai")
    return label_dict