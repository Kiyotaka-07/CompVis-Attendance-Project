import cv2
import pickle
from datetime import datetime

haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

# load label dictionary
with open("labels.pkl", "rb") as f:
    label_dict = pickle.load(f)


def attendance(threshold=70):

    cap = cv2.VideoCapture(0)

    print("Tekan 'q' untuk keluar")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        mirror_frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(mirror_frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        MIN_FACE_SIZE = 120

        for (x, y, w, h) in faces:

            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (200, 200))

            label, confidence = recognizer.predict(face)

            if confidence < threshold:

                name = label_dict[label]

                text = f"{name} - Hadir"

                color = (0, 255, 0)

                with open("attendance_log.txt", "a") as f:
                    f.write(f"{name}, {datetime.now()}\n")

            else:

                text = "Unknown"

                color = (0, 0, 255)

            cv2.rectangle(mirror_frame,
                          (x, y),
                          (x+w, y+h),
                          color, 2)

            cv2.putText(mirror_frame,
                        text,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)

        cv2.imshow("Attendance", mirror_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    attendance()