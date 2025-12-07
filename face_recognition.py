import cv2
import os

cascade_path = "models/haarcascade_frontalface_default.xml"
face_classifier = cv2.CascadeClassifier(cascade_path)

# load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# optional mapping id->name (edit as needed)
names = {1: "Swastika", 2: "Rohan"}

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        id_, confidence = recognizer.predict(face)  # lower is better
        conf_text = f"{round(confidence,2)}"
        label = names.get(id_, f"ID {id_}")
        text = f"{label} {conf_text}"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == 13:
        break

cam.release()
cv2.destroyAllWindows()
