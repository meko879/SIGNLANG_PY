# live_check.py
# Real-time Sequence / STM Live Testing using LSTM and 132 features

import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from features import ACTIONS, SEQ_LEN, extract_features

# 1. Load the Sequence Model
MODEL_PATH = 'action_model_v2.keras'
print(f"[INFO] Loading sequence model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)

# 2. Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# 3. Variables for Sequence Control
sentence = []
threshold = 0.85
prediction_history = []
history_length = 10
sequence = []  # قائمة لتخزين الـ 30 إطار الأخيرة (Sequence Buffer)

cap = cv2.VideoCapture(0)
print("[INFO] Starting STM / Sequence Sign Language Recognition...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image_rgb)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
            )
            
    # استخراج الـ 132 خاصية وإضافتها للمتسلسلة
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
                
                if len(sentence) == 0 or current_action != sentence[-1]:
                    sentence.append(current_action)
        else:
            if len(prediction_history) > 0:
                prediction_history.pop(0)
    else:
        if not results.multi_hand_landmarks:
            prediction_history.clear()

    if len(sentence) > 4:
        sentence = sentence[-4:]

    h, w, _ = image.shape
    
    # رسم شريط العرض بالأعلى
    cv2.rectangle(image, (0, 0), (w, 70), (15, 15, 15), -1)
    
    sentence_text = ' '.join(sentence).upper()
    display_text = sentence_text if sentence_text else "SHOW YOUR HAND..."
    cv2.putText(image, display_text, (25, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # عرض حالة تجميع الإطارات أو الكلمة المكتشفة بالأسفل
    if current_action:
        info_text = f"Action: {current_action.upper()} ({current_confidence * 100:.0f}%)"
        cv2.putText(image, info_text, (25, h - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2, cv2.LINE_AA)
    else:
        loading_text = f"Collecting Frames: {len(sequence)}/{SEQ_LEN}" if results.multi_hand_landmarks else "WAITING FOR HANDS..."
        cv2.putText(image, loading_text, (25, h - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('STM Sign Language Recognition', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()