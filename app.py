import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import json
from gtts import gTTS
from playsound import playsound
from keras.models import load_model
import logging
import threading

# استيراد إعدادات ومستخرج الخصائص الحديث من features.py
from features import ACTIONS, SEQ_LEN, TOTAL_FEATURES, extract_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page Configuration 
st.set_page_config(
    page_title="SignSpeak AI: Intelligent Sign Language Communication",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling 
st.markdown("""
    <style>
        .main {
            background-color: #121212; 
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
        }

        .stMetric {
            background-color: #1e1e1e;
            padding: 15px;
            border-radius: 12px;
            border: 1px solid #333;
            box-shadow: 0 2px 6px rgba(0,0,0,0.4);
        }

        .stAlert {
            border-radius: 10px;
            padding: 12px;
            font-weight: 500;
        }
        
        h1, h2, h3 {
            color: #ffffff;
            font-family: 'Segoe UI Semibold', sans-serif;
            letter-spacing: 0.5px;
        }

        button {
            background-color: #0078d7 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            font-weight: bold !important;
        }
    </style>
""", unsafe_allow_html=True)

class SignLanguageInferenceEngine:
    """Encapsulates AI Sequence Model loading and MediaPipe hand tracking pipeline."""
    
    @staticmethod
    @st.cache_resource
    def load_ai_model(model_path: str):
        try:
            logger.info("Loading Keras model from disk...")
            return load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    @staticmethod
    @st.cache_resource
    def initialize_mediapipe():
        mp_hands = mp.solutions.hands
        return mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.75, 
            min_tracking_confidence=0.75
        )

# استخدام الموديل v2 والـ STM
MODEL_PATH = 'action_model_v2.keras'
model = SignLanguageInferenceEngine.load_ai_model(MODEL_PATH)
hands_detector = SignLanguageInferenceEngine.initialize_mediapipe()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# gTTS Asynchronous Text-to-Speech Subsystem (Optimized & Fast)
def speak_text_async(word: str) -> None:
    """Safely executes gTTS speech synthesis asynchronously without blocking Streamlit UI."""
    def run_speech():
        try:
            filename = "temp_speech.mp3"
            if os.path.exists(filename):
                os.remove(filename)
            
            tts = gTTS(text=word, lang='en', slow=False)
            tts.save(filename)
            playsound(filename)
            
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as ex:
            logger.warning(f"TTS Engine warning: {ex}")

    t = threading.Thread(target=run_speech, daemon=True)
    t.start()

# Sidebar 
st.sidebar.title("🎛️ Control Panel")
st.sidebar.info("STM / Sequence-Based Real-Time Sign Language Translation System")

run_system = st.sidebar.checkbox('Activate Video Stream', value=False)
threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.95, 0.85, 0.05)
clear_button = st.sidebar.button("Clear Sentence Buffer")

# Main Dashboard Layout 
st.title("Enterprise Real-Time Sign Language-to-Speech Platform (STM / LSTM)")
st.markdown("Advanced Deep Learning Sequence Pipeline (30 Frames Window & 132 Features).")
st.markdown("---")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("🎥 Live Feed & Landmark Extraction")
    frame_placeholder = st.empty()

with col2:
    st.subheader("💬 Translated Sequence")
    sentence_placeholder = st.empty()
    confidence_placeholder = st.empty()
    st.markdown("---")
    st.subheader("📊 System Telemetry")
    status_placeholder = st.empty()

# Session State Management for STM Sequence Buffer
if 'sentence_history' not in st.session_state:
    st.session_state.sentence_history = []

if 'prediction_buffer' not in st.session_state:
    st.session_state.prediction_buffer = []

if 'sequence_buffer' not in st.session_state:
    st.session_state.sequence_buffer = []

if clear_button:
    st.session_state.sentence_history = []
    st.session_state.prediction_buffer = []
    st.session_state.sequence_buffer = []
    st.rerun()

HISTORY_WINDOW_SIZE = 10

# Main Video Processing 
if run_system:
    if model is None:
        status_placeholder.error(f"Critical Error: Model file ({MODEL_PATH}) missing or unreadable.")
    else:
        cap = cv2.VideoCapture(0)
        status_placeholder.success("STM Sequence Engine Active 🟢")
        
        while run_system:
            success, frame = cap.read()
            if not success:
                st.warning("Hardware Error: Unable to access webcam stream.")
                break

            # Preprocessing Pipeline with cv2.flip BEFORE MediaPipe
            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(image_rgb)

            # Draw Hand Landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

            # Extract 132 features from features.py
            res = extract_features(results)
            st.session_state.sequence_buffer.append(res)
            st.session_state.sequence_buffer = st.session_state.sequence_buffer[-SEQ_LEN:]

            current_confidence = 0.0
            predicted_word = ""

            # STM Model Inference (Executed when 30 frames are collected)
            if len(st.session_state.sequence_buffer) == SEQ_LEN:
                input_data = np.expand_dims(st.session_state.sequence_buffer, axis=0) # Shape: (1, 30, 132)
                predictions = model.predict(input_data, verbose=0)[0]
                best_action_index = np.argmax(predictions)
                current_confidence = float(predictions[best_action_index])
                predicted_word = ACTIONS[best_action_index]

                if current_confidence > threshold:
                    st.session_state.prediction_buffer.append(predicted_word)
                    if len(st.session_state.prediction_buffer) > HISTORY_WINDOW_SIZE: 
                        st.session_state.prediction_buffer.pop(0)

                    # Majority voting for temporal smoothing
                    if st.session_state.prediction_buffer.count(predicted_word) > int(HISTORY_WINDOW_SIZE * 0.6):
                        current_action = predicted_word
                        if len(st.session_state.sentence_history) == 0 or current_action != st.session_state.sentence_history[-1]:
                            st.session_state.sentence_history.append(current_action)
                            speak_text_async(current_action)
                else:
                    if len(st.session_state.prediction_buffer) > 0:
                        st.session_state.prediction_buffer.pop(0)
            else:
                if not results.multi_hand_landmarks:
                    st.session_state.prediction_buffer.clear()

            # Dynamic UI Updates
            formatted_sentence = ' '.join(st.session_state.sentence_history).upper()
            sentence_placeholder.markdown(f"### `{formatted_sentence if formatted_sentence else 'Listening for sign gestures...'}`")
            
            # Confidence Card
            confidence_placeholder.markdown(
                f"""
                <div style="
                    background-color: #d4edda; 
                    border: 2px solid #c3e6cb; 
                    padding: 12px; 
                    border-radius: 12px; 
                    text-align: center;
                    color: #155724;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                ">
                    <p style="margin: 0; font-size: 13px; font-weight: 600; text-transform: uppercase;">Inference Confidence</p>
                    <h3 style="margin: 0; font-size: 26px; font-weight: bold; color: #155724;">{current_confidence * 100:.1f}%</h3>
                    <small style="color: #155724;">Sequence Buffer: {len(st.session_state.sequence_buffer)}/{SEQ_LEN}</small>
                </div>
                """, 
                unsafe_allow_html=True
            )

            # Streamlit Frame Render
            frame_rgb_final = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb_final, channels="RGB", use_container_width=True)

        cap.release()
else:
    status_placeholder.warning("System Standby 🔴 (Toggle stream from sidebar)")
    frame_placeholder.info("Activate the camera feed via the control panel to begin real-time translation.")

    #streamlit run app.py