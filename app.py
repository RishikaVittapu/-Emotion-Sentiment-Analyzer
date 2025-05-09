import cv2
from camera import open_camera, read_frame, show_frame, release_camera
from detector import load_face_detector, detect_faces
from analyzer import analyze_emotion
from utils import draw_emotion
from text_analyzer import analyze_text_sentiment

def run_facial_emotion_analyzer():
    cap = open_camera()
    face_cascade = load_face_detector()

    frame_count = 0
    last_emotion = "Detecting..."

    while True:
        frame = read_frame(cap)
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(face_cascade, gray)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if frame_count % 10 == 0:
                last_emotion = analyze_emotion(face_img)
            draw_emotion(frame, (x, y, w, h), last_emotion)

        show_frame(frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_camera(cap)

def run_text_sentiment_analyzer():
    print("\n📝 TEXT SENTIMENT ANALYZER")
    while True:
        text = input("Enter a sentence (or type 'exit' to go back):\n> ")
        if text.lower() == 'exit':
            break
        sentiment = analyze_text_sentiment(text)
        print(f"Sentiment: {sentiment}\n")

def main_menu():
    while True:
        print("\n=== SENTIMENT ANALYZER MENU ===")
        print("1. Facial Expression Detection (Webcam)")
        print("2. Text Sentiment Analysis")
        print("3. Exit")
        choice = input("Select an option (1/2/3): ")

        if choice == '1':
            run_facial_emotion_analyzer()
        elif choice == '2':
            run_text_sentiment_analyzer()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
