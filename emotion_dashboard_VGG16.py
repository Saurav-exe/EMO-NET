import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import plotly.express as px


# Load VGG16-based Emotion Detection Model
@st.cache_resource
def load_model():
    model_path = "emotion_model_vgg16.h5"
    if not os.path.exists(model_path):
        st.error("Model file not found.")
        return None
    return tf.keras.models.load_model(model_path)


# Load Haar Cascade for Face Detection
@st.cache_resource
def load_face_detector():
    cascade_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        st.error("Haar Cascade file not found.")
        return None
    return cv2.CascadeClassifier(cascade_path)


# Process Video and Detect Emotions
def process_video(video_path, skip_frames=5):
    cap = cv2.VideoCapture(video_path)
    face_detector = load_face_detector()
    emotion_model = load_model()
    emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    frame_number = 0
    emotions_over_time = []
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % skip_frames != 0:
            frame_number += 1
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_color = frame[y:y + h, x:x + w]
            roi_color = cv2.resize(roi_color, (48, 48))
            roi = img_to_array(roi_color) / 255.0
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_model.predict(roi)
            label = emotion_labels[np.argmax(preds)]
            emotions_over_time.append(label)
            timestamps.append(frame_number)

        frame_number += 1

    cap.release()
    return timestamps, emotions_over_time


# Streamlit App
st.title("🎭 Emotion Detection and Analysis Dashboard")
uploaded_file = st.file_uploader("📤 Upload a Video for Emotion Detection", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    timestamps, emotions = process_video(video_path)
    data = {"Timestamp": timestamps, "Emotion": emotions}
    st.write("✅ Video processing complete! View your results below:")

    # Visualizations
    fig = px.scatter(data, x="Timestamp", y="Emotion", title="Emotion Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Cleanup
    os.remove(video_path)
