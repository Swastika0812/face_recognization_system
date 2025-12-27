import cv2
import numpy as np
import os

# ------------------ Check Face Module ------------------
if not hasattr(cv2, "face"):
    raise ImportError("OpenCV-contrib not installed. Run: pip install opencv-contrib-python")

# ------------------ Initialize Recognizer ------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()

DATASET_PATH = "dataset"
TRAINER_PATH = "trainer"

# ------------------ Load Images & Labels ------------------
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

# ------------------ Main ------------------
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
