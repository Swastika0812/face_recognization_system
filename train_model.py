import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()  # LBPH recognizer
dataset_path = "dataset"

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    faces = []
    ids = []
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # filename: user.<id>.<count>.jpg
        id_num = int(os.path.basename(image_path).split(".")[1])
        faces.append(img)
        ids.append(id_num)
    return faces, np.array(ids)

if __name__ == "__main__":
    faces, ids = get_images_and_labels(dataset_path)
    if len(faces) == 0:
        print("No images found in dataset/. Run face_capture.py first.")
        exit()

    recognizer.train(faces, ids)
    if not os.path.exists("trainer"):
        os.makedirs("trainer")
    recognizer.save("trainer/trainer.yml")
    print("Training complete, saved trainer/trainer.yml")
