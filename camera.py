import cv2

def open_camera():
    return cv2.VideoCapture(0)

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture frame")
    return frame

def show_frame(frame):
    cv2.imshow("Facial Expression Analyzer", frame)

def release_camera(cap):
    cap.release()
    cv2.destroyAllWindows()
