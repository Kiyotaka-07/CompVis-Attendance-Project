import cv2
import os
import numpy as np
from datetime import datetime

# print(cv2.data.haarcascades)
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
        
        for (x, y, w, h) in faces:
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

def attendance(label_dict, threshold=50):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_model.yml")
    
    cap = cv2.VideoCapture(0)
    
    print("Tekan 'q' untuk keluar")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        mirror_frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(mirror_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            
            label, confidence = recognizer.predict(face)
            
            if confidence < threshold:
                name = label_dict[label]
                # nim = label_dict[label]
                text = f"{name} - Hadir"
                color = (0,255,0)
                
                # Simpan log absensi
                with open("attendance_log.txt", "a") as f:
                    f.write(f"{name}, {datetime.now()}\n")
                    
            else:
                text = "Unknown"
                color = (0,0,255)
            
            cv2.rectangle(mirror_frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(mirror_frame, text, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow("Attendance", mirror_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while(True):
        if __name__ == "__main__":
            print("1. Register")
            print("2. Attendance")
            print("3. Exit")
            pilihan = input("Pilih (1/2/3): ")

            match pilihan:
                case "1":
                    nama = input("Nama anda: ")
                    nimA = input("NIM anda: ")
                    register_face(nama, nimA, samples=15)
                case "2":
                    label_dict = train_model()
                    attendance(label_dict, threshold=70)
                case "3":
                    print("Exiting the program...")
                    break
                case _:
                    print("Invalid")