import cv2
import os
import time

CASCADE_PATH = "models/haarcascade_frontalface_default.xml"
DATASET_PATH = "dataset"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise IOError("Cannot open webcam")

user_id = 1   # change manually for new person
count = 0
last_capture_time = 0

print("Camera started — 1 image per second")
print("Move your face slowly (left, right, up, down)")
print("Press Q / ESC to stop early")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_time = time.time()

    for (x, y, w, h) in faces:
        if current_time - last_capture_time >= 1:  # 1 second gap
            count += 1
            last_capture_time = current_time

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            cv2.imwrite(f"dataset/user.{user_id}.{count}.jpg", face)

            print(f"Saved image {count}")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Images: {count}/50", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27 or count >= 50:
        break

cam.release()
cv2.destroyAllWindows()
print(f"Collected {count} images")
