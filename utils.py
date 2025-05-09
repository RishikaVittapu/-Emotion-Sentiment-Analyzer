import cv2

def draw_emotion(frame, face_coords, emotion):
    (x, y, w, h) = face_coords
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, emotion, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
