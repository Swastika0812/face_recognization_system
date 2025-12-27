import cv2
import os

# ------------------ Paths ------------------
CASCADE_PATH = "models/haarcascade_frontalface_default.xml"
TRAINER_PATH = "trainer/trainer.yml"

# ------------------ Load Face Cascade ------------------
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise IOError("Haarcascade XML not found or failed to load")

# ------------------ Load Trained Model ------------------
if not os.path.exists(TRAINER_PATH):
    raise IOError("trainer.yml not found. Run train_model.py first")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_PATH)

# ------------------ ID → Name Mapping ------------------
names = {
    1: "Swastika",
    2: "Rohan"
}

# ------------------ Start Camera ------------------
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise IOError("Cannot access webcam")

print("Face recognition started (Press Q or ESC to exit)")

# ------------------ Recognition Loop ------------------
while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        if confidence < 70:
            name = names.get(label, "Unknown")
            text = f"{name} ({round(confidence, 2)})"
            color = (0, 255, 0)
        else:
            text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    #  Exit with Q or ESC
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# ------------------ Cleanup ------------------
cam.release()
cv2.destroyAllWindows()
print("Camera stopped")
