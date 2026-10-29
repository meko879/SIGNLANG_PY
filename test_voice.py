# live_check.py
# Advanced STM (Sequence/LSTM) Sign Language Recognition + Async gTTS Voice

import cv2
import numpy as np
import mediapipe as mp
import time
import os
from gtts import gTTS
from playsound import playsound
from keras.models import load_model
from features import ACTIONS, SEQ_LEN, TOTAL_FEATURES, extract_features

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def speak_gtts_async(word):
    def run_speech():
        try:
            tts = gTTS(text=word, lang='en', slow=False)
            filename = f"temp_{time.time()}.mp3"
            tts.save(filename)
            playsound(filename)
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            print(f"Audio Error: {e}")

    import threading
    t = threading.Thread(target=run_speech, daemon=True)
    t.start()

if __name__ == '__main__':
    # تحميل الموديل الحديث الخاص بالمتسلسلات (STM)
    MODEL_PATH = 'action_model_v2.keras'
    print(f"[INFO] Loading sequence model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)

    # MediaPipe Hands
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2, 
        min_detection_confidence=0.75, 
        min_tracking_confidence=0.75
    )

    # control variables 
    sentence = []
    threshold = 0.85              
    prediction_history = []   
    history_length = 10       
    sequence = []                 # قائمة لحفظ الـ 30 إطار الأخيرة (STM Sequence Buffer)

    last_spoken_word = ""
    last_spoken_time = 0
    cooldown_duration = 1.0   

    cap = cv2.VideoCapture(0)
    window_name = 'STM Sign Language to Speech - Graduation Project'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("[INFO] System active with STM model and gTTS voice...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image = cv2.flip(frame, 1) 
        image = cv2.resize(image, (1280, 720))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = hands.process(image_rgb)
        
        # draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                )
                
        # استخراج الـ 132 خاصية باستخدام ملف features.py الحديث
        res = extract_features(results)
        sequence.append(res)
        sequence = sequence[-SEQ_LEN:]  # الاحتفاظ بآخر 30 إطار فقط

        current_action = ""
        current_confidence = 0.0

        # التنبؤ يتم حصراً عند اكتمال الـ 30 إطار (STM Logic)
        if len(sequence) == SEQ_LEN:
            input_data = np.expand_dims(sequence, axis=0)  # Shape: (1, 30, 132)
            predictions = model.predict(input_data, verbose=0)[0]
            best_action_index = np.argmax(predictions)
            current_confidence = predictions[best_action_index]
            predicted_word = ACTIONS[best_action_index]

            if current_confidence > threshold:
                prediction_history.append(predicted_word)
                if len(prediction_history) > history_length: 
                    prediction_history.pop(0)

                if prediction_history.count(predicted_word) > int(history_length * 0.7):
                    current_action = predicted_word
                    
                    current_time = time.time()
                    if len(sentence) == 0 or current_action != sentence[-1]:
                        if (current_action != last_spoken_word) or (current_time - last_spoken_time > cooldown_duration):
                            sentence.append(current_action)
                            # نطق الكلمة صوتياً بشكل متوازي وبدون توقف الشاشة
                            speak_gtts_async(current_action)
                            last_spoken_word = current_action
                            last_spoken_time = current_time
            else:
                if len(prediction_history) > 0: 
                    prediction_history.pop(0)
        else:
            if not results.multi_hand_landmarks:
                prediction_history.clear()

        if len(sentence) > 6: 
            sentence = sentence[-6:]

        h, w, _ = image.shape
        cv2.rectangle(image, (0, 0), (w, 75), (15, 15, 15), -1)
        
        sentence_text = ' '.join(sentence).upper()
        display_text = sentence_text if sentence_text else "SHOW YOUR HAND TO START..."
        cv2.putText(image, display_text, (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if current_action:
            info_text = f"Action: {current_action.upper()} ({current_confidence * 100:.0f}%)"
            cv2.putText(image, info_text, (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2, cv2.LINE_AA)
        else:
            loading_text = f"Collecting Frames: {len(sequence)}/{SEQ_LEN}" if results.multi_hand_landmarks else "WAITING FOR HANDS..."
            cv2.putText(image, loading_text, (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, image)

        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()