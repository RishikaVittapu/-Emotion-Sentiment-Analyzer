from deepface import DeepFace

def analyze_emotion(face_img):
    try:
        analysis = DeepFace.analyze(
            face_img,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='mtcnn' 
        )
        return analysis[0]['dominant_emotion']
    except Exception as e:
        return "Unknown"
