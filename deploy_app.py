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
set_background("back.png")  # Ensure back.png is in the correct location

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

        # Convert uploaded image to a PIL image and save it temporarily
        image = Image.open(uploaded_image).convert("RGB")
        image_path = "temp_image.jpg"  # Temporary image path
        image.save(image_path)

        # Analyze the image using DeepFace
        analysis = DeepFace.analyze(
            img_path=image_path,  # Use the saved image path
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
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
