import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import time
from keras.models import model_from_json
from keras_preprocessing.image import img_to_array
import plotly.express as px


# Load Emotion Detection Model
@st.cache_resource
def load_model():
    model_path = "emotion_model.json"
    weights_path = "emotion_model.h5"
    if not os.path.exists(model_path) or not os.path.exists(weights_path):
        st.error("Model files not found.")
        return None
    with open(model_path, "r") as json_file:
        loaded_model_json = json_file.read()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights(weights_path)
    return emotion_model


# Load Haar Cascade for Face Detection
@st.cache_resource
def load_face_detector():
    cascade_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        st.error("Haar Cascade file not found.")
        return None
    return cv2.CascadeClassifier(cascade_path)


# Emotion Scores for Gamification
emotion_scores = {
    "Happy": 2,
    "Neutral": 1,
    "Surprise": 1,
    "Fear": -1,
    "Sad": -2,
    "Angry": -3,
    "Disgust": -3
}


# Function to calculate Emotion Score
def calculate_emotion_score(data):
    return sum(emotion_scores.get(emotion, 0) for emotion in data["Emotion"])


# Function to provide positivity tips based on emotion score
# Function to provide multiple positivity tips based on emotion score
def get_tips_for_emotion(score):
    if score > 10:
        tips = [
            "🌟 Fantastic! Keep spreading positivity.",
            "🎨 Try engaging in creative activities like art or music.",
            "🌿 Spend time in nature to boost your happiness.",
            "📖 Read an inspiring book to keep the positivity flowing."
        ]
    elif score > 0:
        tips = [
            "😊 You're doing well! Stay around happy people.",
            "🙌 Practice gratitude—write down three things you're thankful for.",
            "🎶 Listen to your favorite music to uplift your mood.",
            "🏋️ Exercise for 10-15 minutes to release endorphins."
        ]
    elif score > -10:
        tips = [
            "😐 Feeling neutral? Try deep breathing exercises.",
            "🚶 Go for a short walk outside for fresh air and relaxation.",
            "🍫 Treat yourself to something small you enjoy.",
            "🤝 Talk to a friend or family member—it helps more than you think!"
        ]
    else:
        tips = [
            "😢 Need a mood boost? Talk to someone who supports you.",
            "🏃 Engage in physical activity—it can lift your spirits.",
            "🎭 Watch a comedy or funny videos for instant laughter.",
            "📅 Plan something exciting to look forward to!"
        ]

    return "\n\n".join(tips)


# Process Video and Collect Emotion Data with Progress Bar
def process_video(video_path, output_path, skip_frames=1):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 for better support
    temp_output_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

    emotion_model = load_model()
    face_detector = load_face_detector()

    emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear",
                      3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

    frame_number = 0
    emotions_over_time = []
    progress_bar = st.progress(0)  # Initialize progress bar

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
            roi_gray = gray_frame[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_model.predict(roi, verbose=0)[0]
            label = emotion_labels[np.argmax(preds)]

            emotions_over_time.append(label)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36, 255, 12), 2)

        out.write(frame)
        frame_number += 1

        # Update Progress Bar
        progress_bar.progress(frame_number / total_frames)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    progress_bar.empty()  # Remove progress bar after completion

    return pd.DataFrame({"Emotion": emotions_over_time})


# Generate Emotion Timeline Scatter Plot
def generate_emotion_scatterplot(data):
    data["Frame"] = data.index  # Add frame numbers
    fig = px.scatter(data, x="Frame", y="Emotion", title="🎯 Emotion Timeline Game",
                     labels={"Frame": "Frame Number", "Emotion": "Detected Emotion"},
                     color="Emotion", opacity=0.7)

    # Calculate Emotion Score
    total_score = calculate_emotion_score(data)

    # Display Emotion Score
    st.subheader(f"🏆 Emotion Score: {total_score}")

    # Show positivity tip
    st.info(get_tips_for_emotion(total_score))

    return fig


# Generate Pie Chart
def generate_emotion_distribution(data):
    emotion_counts = data["Emotion"].value_counts().reset_index()
    emotion_counts.columns = ["Emotion", "Count"]
    fig = px.pie(emotion_counts, names="Emotion", values="Count", title="Emotion Distribution")
    return fig


# Streamlit App
st.title("🎭 EmoNet: We Empower Machines With Emotion ")

uploaded_file = st.file_uploader("📤 Upload a Video for Emotion Detection", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    output_video_path = os.path.join(tempfile.gettempdir(), "analyzed_video.mp4")

    st.info("⏳ Processing your video... Please wait.")

    emotion_data = process_video(video_path, output_video_path, skip_frames=1)

    if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
        st.success("✅ Video processing complete! View your results below:")

        tab1, tab2, tab3 = st.tabs(["🎬 Processed Video", "📊 Emotion Distribution", "🎯 Emotion Timeline Game"])

        with tab1:
            st.video(output_video_path)

        with tab2:
            st.plotly_chart(generate_emotion_distribution(emotion_data), use_container_width=True)

        with tab3:
            st.plotly_chart(generate_emotion_scatterplot(emotion_data), use_container_width=True)

    os.remove(video_path)
