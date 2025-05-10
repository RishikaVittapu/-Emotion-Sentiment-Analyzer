import streamlit as st
from deepface import DeepFace
from textblob import TextBlob
from PIL import Image
import numpy as np
import cv2
import base64
import os
import time

# ---------- Background Image Setup ----------
def set_background(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            data_url = base64.b64encode(f.read()).decode()
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{data_url}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

# Add slight delay to avoid JS fetch race condition
time.sleep(1)
set_background("back.png")

# ---------- Load Emotion Model Once ----------
@st.cache_resource
def load_emotion_model():
    return DeepFace.build_model("Emotion")

emotion_model = load_emotion_model()

# ---------- Text Sentiment Analysis ----------
def analyze_text_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive 😊"
    elif polarity < 0:
        return "Negative 😞"
    else:
        return "Neutral 😐"

# ---------- Facial Emotion Detection ----------
def analyze_face_emotion(uploaded_image):
    try:
        if uploaded_image is None:
            return "No image provided."

        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        analysis = DeepFace.analyze(
            img_path=image_bgr,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',  # faster than mtcnn
            models={'emotion': emotion_model}
        )

        emotions = analysis[0]['emotion']
        dominant_emotion = analysis[0]['dominant_emotion']
        confidence = emotions[dominant_emotion]
        return f"{dominant_emotion.capitalize()} ({confidence:.2f}%)"

    except Exception as e:
        return f"Error: {str(e)}"

# ---------- Streamlit App Layout ----------
st.title("😊 Emotion and Sentiment Analyzer")
tabs = st.tabs(["Text Sentiment Analysis", "Facial Emotion Detection"])

with tabs[0]:
    st.subheader("Text Sentiment Analysis")
    user_text = st.text_area("Enter your text")
    if st.button("Analyze Text"):
        result = analyze_text_sentiment(user_text)
        st.success(f"Sentiment: {result}")

with tabs[1]:
    st.subheader("Facial Emotion Detection")
    option = st.radio("Choose input method:", ("Upload Image", "Use Camera"))

    if option == "Upload Image":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    else:
        uploaded_image = st.camera_input("Take a picture")

    if st.button("Analyze Emotion") and uploaded_image is not None:
        emotion_result = analyze_face_emotion(uploaded_image)
        st.success(f"Detected Emotion: {emotion_result}")
