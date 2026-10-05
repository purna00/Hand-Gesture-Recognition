import cv2
import mediapipe as mp
import pyautogui
import time
import os

# Create screenshot folder
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

last_screenshot_time = 0
last_volume_time = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    total_fingers = 0
    gesture = "No Hand"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark

            fingers = []

            # Thumb
            if landmarks[4].x < landmarks[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other Fingers
            tips = [8, 12, 16, 20]

            for tip in tips:

                if landmarks[tip].y < landmarks[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = sum(fingers)

            # Gesture Recognition

            if total_fingers == 0:
                gesture = "FIST"

            elif total_fingers == 5:
                gesture = "OPEN PALM"

            elif total_fingers == 1 and fingers[0] == 1:
                gesture = "THUMBS UP"

            elif total_fingers == 2:
                gesture = "TWO FINGERS"

            else:
                gesture = "UNKNOWN"

            current_time = time.time()

            # Screenshot
            if gesture == "OPEN PALM":

                if current_time - last_screenshot_time > 3:

                    filename = f"screenshots/screenshot_{int(current_time)}.png"

                    pyautogui.screenshot(filename)

                    print("Screenshot Captured")

                    last_screenshot_time = current_time

            # Volume Up
            if gesture == "THUMBS UP":

                if current_time - last_volume_time > 1:

                    pyautogui.press("volumeup")

                    print("Volume Up")

                    last_volume_time = current_time

            # Volume Down
            if gesture == "TWO FINGERS":

                if current_time - last_volume_time > 1:

                    pyautogui.press("volumedown")

                    print("Volume Down")

                    last_volume_time = current_time

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"Fingers: {total_fingers}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        3
    )

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()