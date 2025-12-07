import cv2
import os

# 1: Path to cascade
cascade_path = "models/haarcascade_frontalface_default.xml"
face_classifier = cv2.CascadeClassifier(cascade_path)

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          # convert to gray
    faces = face_classifier.detectMultiScale(gray, 1.3, 5) # detect
    if len(faces) == 0:
        return None
    # take last detected face (or first) and crop
    x, y, w, h = faces[0]
    return gray[y:y+h, x:x+w]   # return grayscale crop

# --------- Main ----------
if __name__ == "__main__":
    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    cam = cv2.VideoCapture(0)                 # open webcam (0)
    user_id = input("Enter numeric user ID: ")   # e.g. 1
    user_name = input("Enter user name (optional): ")

    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image from camera")
            break

        face = face_extractor(frame)
        if face is not None:
            count += 1
            face = cv2.resize(face, (200, 200))
            file_path = f"dataset/user.{user_id}.{count}.jpg"
            cv2.imwrite(file_path, face)
            cv2.putText(face, str(count), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 2)
            cv2.imshow("Face Capture", face)

        if cv2.waitKey(1) == 13 or count >= 50:  # Enter key or 50 images
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Collected", count, "images for user", user_id)
