# Emotion & Sentiment Analyzer 🎭🧠

A Streamlit-based web app that detects **facial emotions** using DeepFace and performs **text sentiment analysis** using TextBlob. It combines visual and linguistic cues to provide a simple yet insightful emotional intelligence tool.

## 🔍 Features

- 📷 **Facial Emotion Detection** (via image upload)
- 📝 **Text Sentiment Analysis** (positive, negative, neutral)
- 📊 Emotion confidence scores with optional bar chart (for enhancements)
- 🎨 Custom background UI with clean two-tab layout
- 💻 Simple and interactive Streamlit interface

## 🚀 Tech Stack

- [Streamlit](https://streamlit.io/) – Web UI
- [DeepFace](https://github.com/serengil/deepface) – Emotion recognition
- [TextBlob](https://textblob.readthedocs.io/en/dev/) – Sentiment analysis
- [OpenCV](https://opencv.org/) & [PIL](https://pillow.readthedocs.io/en/stable/) – Image processing

## 🛠 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/RishikaVittapu/emotion-sentiment-analyzer.git
cd emotion-sentiment-analyzer

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
