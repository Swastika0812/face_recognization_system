

# ...existing code...
import sys
try:
    import cv2
except Exception as e:
    print("Error: could not import OpenCV (cv2). Install it with 'pip install opencv-python'.")
    print("Exception:", e)
    sys.exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open camera. Check the device index or permissions.")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        # If a frame wasn't read, break to avoid a busy infinite loop.
        print("Warning: failed to read frame from camera.")
        break

    cv2.imshow("video frame", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# ...existing code...