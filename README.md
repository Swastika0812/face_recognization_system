# Introduction
  Step into the world of Computer Vision and Artificial Intelligence!
This project focuses on building a real-time Face Recognition System that can detect, capture, train, and recognize human faces using a live webcam feed.

By leveraging OpenCV and the LBPH algorithm, the system demonstrates how machines can visually perceive and identify individuals — a core concept behind modern security, surveillance, and authentication systems.

🔍 All Python scripts can be found inside the [trainer folder](/trainer/).
# Background
With the growing use of **biometric authentication** in security systems, attendance management, and smart devices, face recognition has become one of the most practical applications of AI.

This project was built to:

Understand how real-time face detection works

Learn how face recognition models are trained

Implement a complete end-to-end pipeline from data collection to recognition

The project uses **Haar Cascade classifiers** for face detection and **LBPH (Local Binary Pattern Histogram)** for face recognition — both well-suited for real-time applications.

# Objectives of This Project
#### The key goals I wanted to achieve were:

- How to detect faces from a live camera feed

- How to capture and store face images automatically

- How to train a recognition model using collected images

- How to recognize faces in real time with confidence scores

- How to build a reusable and scalable face recognition system



# Tools & Technologies Used

#### To build this face recognition system, I used the following tools:

- Python – Core programming language

- OpenCV – For image processing, face detection, and recognition

- NumPy – Numerical operations on image data

- Haar Cascade Classifier – Face detection

- LBPH Algorithm – Face recognition

- isual Studio Code – Development environment

- Git & GitHub – Version control and project hosting

# Project Workflow

The project is divided into three major stages, forming a complete face recognition pipeline.

### 1️⃣ Face Detection & Dataset Creation

In this stage:

- The webcam is activated

- Faces are detected using Haar Cascade

- A green rectangle highlights detected faces

- 50 grayscale face images are captured per user

- Images are saved automatically in the dataset folder

This step ensures the model receives diverse face angles and expressions.
```python
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
```

📌 Key Features

- Live camera window

- Real-time face detection

- Automatic image counter

- Dataset generation without manual effort

![Auto Face Capturing](assets\face_capturing.png)

### 2️⃣ Model Training

   ##### Once the dataset is created:

- Images are read and labeled

- Facial features are extracted

- The LBPH Face Recognizer is trained

- The trained model is saved for reuse

This step transforms raw images into a trained recognition model.
```python
import cv2
import numpy as np
import os

if not hasattr(cv2, "face"):
    raise ImportError("OpenCV-contrib not installed. Run: pip install opencv-contrib-python")


recognizer = cv2.face.LBPHFaceRecognizer_create()

DATASET_PATH = "dataset"
TRAINER_PATH = "trainer"


def get_images_and_labels(path):
    faces = []
    ids = []

    for file in os.listdir(path):
        if not file.lower().endswith(".jpg"):
            continue

        image_path = os.path.join(path, file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        try:
            # filename format: user.<id>.<count>.jpg
            id_num = int(file.split(".")[1])
        except:
            print(f"Skipping invalid file: {file}")
            continue

        img = cv2.resize(img, (200, 200))  # Ensure same size
        faces.append(img)
        ids.append(id_num)

    return faces, np.array(ids)


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print("dataset folder not found")
        exit()

    faces, ids = get_images_and_labels(DATASET_PATH)

    if len(faces) == 0:
        print("No images found. Run face_capture.py first.")
        exit()

    recognizer.train(faces, ids)

    if not os.path.exists(TRAINER_PATH):
        os.makedirs(TRAINER_PATH)

    recognizer.save(os.path.join(TRAINER_PATH, "trainer.yml"))

    print(f"Training complete")
    print(f"Total images trained: {len(faces)}")
    print(f"Unique users: {len(set(ids))}")
```

📌 Why LBPH?

- Works well in real-time

- Handles lighting variations

- Effective for small datasets

- Computationally efficient

### 3️⃣ Real-Time Face Recognition

##### In the final stage:

- The webcam starts again

- Faces are detected and recognized

- The system displays:

1. Person’s name

2. Confidence score

- Unknown faces are labeled accordingly

```python
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
```

📌 Live Output Includes

- Full face visible in video

- Green bounding box around face

- Name & confidence displayed

- Smooth real-time performance

![Live Output: Face-recognise](assets\Output.png)




The images shown in this repository (screenshots and demo outputs) use **AI-generated faces**.

🔒 **Privacy Notice**  
- No real person’s facial data is stored or shared  
- All demo faces are synthetically generated for demonstration purposes only  
- The system works the same way for real-time webcam input

This approach ensures privacy safety while demonstrating the complete face detection and recognition pipeline.

# What I Learned

This project significantly strengthened my understanding of Computer Vision fundamentals:

-  Face Detection Logic : <br>
Learned how Haar Cascade classifiers identify facial features in images.

- Image Processing : <br>
Worked with grayscale conversion, resizing, and frame-by-frame video analysis.

-  Model Training & Recognition : <br>
Understood how LBPH extracts facial patterns and compares them during recognition.

-  Real-Time Systems : <br>
Built a system that performs detection and recognition live without lag.

- Results & Insights :

1. Face detection works accurately in real time

2. Recognition performs well for known faces

3. LBPH provides fast and reliable results for small datasets

4. Dataset quality directly impacts recognition accuracy

**This project demonstrates how AI-powered vision systems are built from scratch using Python.**

# Conclusion

This Face Recognition System project provided hands-on experience in AI, Computer Vision, and real-time image processing. It showcases how theoretical concepts can be transformed into practical, real-world applications.

- The project reflects my ability to:

- Work with Python and OpenCV

- Build end-to-end AI systems

- pply logical problem-solving skills

- Structure professional GitHub projects