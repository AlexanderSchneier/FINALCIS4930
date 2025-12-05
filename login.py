import cv2
import numpy as np
import random
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model_training/rock_paper_scissors_model.h5",compile = False)


GESTURES = ["rock", "paper", "scissors"]

#sample password
PASSWORD = ["rock", "paper", "scissors", "rock"]

#issue with my own camera
'''
def open_camera():
    for i in [0,1,2]:
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            print(f"using camera {i}")
            return cam
    raise RuntimeError("No camera available")
'''
#resizing to match model expectations
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


#run model to get prediction
def classify_gesture(frame):
    processed = preprocess(frame)
    preds = model.predict(processed)[0]
    idx = np.argmax(preds)
    return GESTURES[idx]


def capture_gesture(gesture_number):
    #allow user to see themselves and choose when to capture the gesture
    cam = cv2.VideoCapture(0)
    print(f"\nShow gesture #{gesture_number} and press SPACE to capture.")

    x = 0;
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Camera error.")
            x = x +1
            if x == 20:
                print("Error, too many camera errors")
                break;
            continue

        cv2.putText(frame, "Press SPACE to capture gesture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Gesture Capture", frame)

        key = cv2.waitKey(1)

        if key == 32: #space key
            gesture = classify_gesture(frame)
            print(f"Captured gesture: {gesture}")
            cam.release()
            cv2.destroyAllWindows()
            return gesture

        if key == 27: #esc key
            print("Cancelled by user.")
            cam.release()
            cv2.destroyAllWindows()
            exit()

    cam.release()
    cv2.destroyAllWindows()


def main():
    user_gestures = []

    # Capture 4 gestures from webcam
    for i in range(1, 5):
        gesture = capture_gesture(i)
        user_gestures.append(gesture)

    print("\nYour gesture sequence:", user_gestures)

    if user_gestures == PASSWORD:
        print("\nSUCCESS: Gesture password matched!")
    else:
        print("\nERROR: One or more gestures did not match the password.")


if __name__ == "__main__":
    main()

