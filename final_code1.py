import cv2
import mediapipe as mp
import time
import serial

# ----------------- Serial Setup -----------------
# Change 'COM3' to the port your ESP32 is connected to
# Baud rate must match ESP32
esp32 = serial.Serial('COM4', 115200, timeout=1)
time.sleep(2)  # Wait for ESP32 to initialize

# ----------------- Mediapipe Setup -----------------
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ----------------- Video Capture -----------------
cap = cv2.VideoCapture(0)

# Device states
fan_on = False
light_on = False
fall_detected = False

# Count fingers function
def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
    thumb_tip = 4
    fingers = []

    # Thumb
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process both hand and pose
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)

    # ---- HAND GESTURE SECTION ----
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count = count_fingers(hand_landmarks)

            # Fan control
            if finger_count == 5 and not fan_on:
                fan_on = True
                esp32.write(b'FAN_ON\n')  # Send command to ESP32
                print("🌀 Fan Turned ON")
            elif finger_count == 0 and fan_on:
                fan_on = False
                esp32.write(b'FAN_OFF\n')
                print("🌀 Fan Turned OFF")

            # Light control
            elif finger_count == 3 and not light_on:
                light_on = True
                esp32.write(b'LIGHT_ON\n')
                print("💡 Light Turned ON")
            elif finger_count == 2 and light_on:
                light_on = False
                esp32.write(b'LIGHT_OFF\n')
                print("💡 Light Turned OFF")

    # ---- FALL DETECTION SECTION ----
    if pose_results.pose_landmarks:
        mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = pose_results.pose_landmarks.landmark

        shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
        hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y +
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2

        shoulder_y *= h
        hip_y *= h

        if (hip_y - shoulder_y) < 50:
            if not fall_detected:
                fall_detected = True
                print("⚠ Fall Detected!")
        else:
            fall_detected = False

    # ---- DISPLAY STATUS ----
    status_text = f"Fan: {'ON' if fan_on else 'OFF'} | Light: {'ON' if light_on else 'OFF'}"
    fall_text = "⚠ Fall Detected!" if fall_detected else "Normal"

    cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, fall_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255) if fall_detected else (0, 255, 0), 3)

    cv2.imshow("Elderly Care System - Fall + Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
esp32.close()
