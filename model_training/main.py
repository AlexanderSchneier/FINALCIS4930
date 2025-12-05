import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

model = load_model("rock_paper_scissors_model.h5")
labels = ["rock", "paper", "scissors", "none"] 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            h, w, c = frame.shape
            x_min = w
            y_min = h
            x_max = y_max = 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if y < y_min: y_min = y
                if x > x_max: x_max = x
                if y > y_max: y_max = y

            pad = 20
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size > 0:
                hand_img_resized = cv2.resize(hand_img, (224, 224))
                hand_img_resized = hand_img_resized.astype('float32') / 255.0
                hand_img_resized = np.expand_dims(hand_img_resized, axis=0)

                preds = model.predict(hand_img_resized)
                class_idx = np.argmax(preds)
                confidence = np.max(preds)
                label = f"{labels[class_idx]} ({confidence*100:.1f}%)"

                # Draw bounding box and label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow('Hand Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

