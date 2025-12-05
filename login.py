import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import os

# ====== LOAD MODEL ======
model = tf.keras.models.load_model("model_training/rock_paper_scissors_model.h5", compile=False)
LABELS = ["rock", "paper", "scissors", "none"]

# ====== PASSWORD CONFIG ======
PASSWORD = ["rock", "paper", "scissors", "rock"]
SAVE_FILE = "captured_gestures.json"

# ====== MEDIAPIPE SETUP ======
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# ====== HELPER: HAND CROP & PREPROCESS ======
def extract_hand(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None, frame

    h, w, _ = frame.shape
    x_min, y_min, x_max, y_max = w, h, 0, 0

    for lm in results.multi_hand_landmarks[0].landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x), max(y_max, y)

    pad = 20
    x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
    x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

    hand_img = frame[y_min:y_max, x_min:x_max]
    if hand_img.size == 0:
        return None, frame

    return hand_img, frame

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_gesture(frame):
    hand_img, _ = extract_hand(frame)
    if hand_img is None:
        return "none", 0.0
    processed = preprocess(hand_img)
    preds = model.predict(processed, verbose=0)[0]
    idx = np.argmax(preds)
    return LABELS[idx], float(np.max(preds))

# ====== CAPTURE ONE GESTURE ======
def capture_gesture(cap, gesture_number):
    print(f"\nShow gesture #{gesture_number} and press SPACE to capture.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            continue
        frame = cv2.flip(frame, 1)

        gesture, conf = classify_gesture(frame)
        cv2.putText(frame, f"Detecting: {gesture} ({conf:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to capture, ESC to cancel",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Gesture Capture", frame)

        key = cv2.waitKey(1)
        if key == 32:  # SPACE
            print(f"Captured: {gesture} (confidence: {conf:.2f})")
            return gesture
        elif key == 27:  # ESC
            print("Cancelled by user.")
            return None

# ====== MAIN LOGIN LOOP ======
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        return

    user_gestures = []
    try:
        for i in range(1, len(PASSWORD) + 1):
            gesture = capture_gesture(cap, i)
            if gesture is None:
                break
            user_gestures.append(gesture)
        print("\nYour gestures:", user_gestures)

        # Compare to password
        if user_gestures == PASSWORD:
            print("Login Successful!")
        else:
            print("❌ Login Failed: Gestures do not match.")

        with open(SAVE_FILE, "w") as f:
            json.dump(user_gestures, f)
        print(f"Saved captured gestures to {os.path.abspath(SAVE_FILE)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
