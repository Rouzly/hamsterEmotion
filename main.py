import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode, HandLandmarker, HandLandmarkerOptions
import time
import os
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def load_and_resize(path, size=(150, 150)):
    img = cv2.imread(resource_path(path))
    if img is None:
        raise FileNotFoundError(f"Не найден файл {path}")
    return cv2.resize(img, size)

img_normal = load_and_resize("Images/normal.jpg")
img_sad = load_and_resize("Images/sad.jpg")
img_squint = load_and_resize("Images/squint.jpg")
img_happy = load_and_resize("Images/happy.jpg")
img_angry = load_and_resize("Images/angry.webp")
img_surprised = load_and_resize("Images/surprised.jpg")
img_like = load_and_resize("Images/like.webp")
img_dislike = load_and_resize("Images/dislike.webp")

# Настройка моделей
options = FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=resource_path('face_landmarker.task')),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True
)
options_hands = HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=resource_path('hand_landmarker.task')),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_landmarker = FaceLandmarker.create_from_options(options)
hand_landmarker = HandLandmarker.create_from_options(options_hands)

cap = cv2.VideoCapture(0)
start_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        continue
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp_ms = int((time.time() - start_time) * 1000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    detection_result_hands = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    gesture = "none"

    if detection_result.face_landmarks and detection_result.face_blendshapes:
        blendshapes = detection_result.face_blendshapes[0]

        # Инициализация переменных эмоций
        mouthSmileLeft = mouthSmileRight = mouthFrownLeft = mouthFrownRight = 0
        mouthDimpleLeft = mouthDimpleRight = jawOpen = 0
        browDownLeft = browDownRight = browInnerUp = 0
        eyeWideLeft = eyeWideRight = 0

        for item in blendshapes:
            if item.category_name == "mouthSmileLeft": mouthSmileLeft = item.score
            if item.category_name == "mouthSmileRight": mouthSmileRight = item.score
            if item.category_name == "mouthFrownLeft": mouthFrownLeft = item.score
            if item.category_name == "mouthFrownRight": mouthFrownRight = item.score
            if item.category_name == "mouthDimpleLeft": mouthDimpleLeft = item.score
            if item.category_name == "mouthDimpleRight": mouthDimpleRight = item.score
            if item.category_name == "jawOpen": jawOpen = item.score
            if item.category_name == "browDownLeft": browDownLeft = item.score
            if item.category_name == "browDownRight": browDownRight = item.score
            if item.category_name == "browInnerUp": browInnerUp = item.score
            if item.category_name == "eyeWideLeft": eyeWideLeft = item.score
            if item.category_name == "eyeWideRight": eyeWideRight = item.score

        # Проверка рук
        if detection_result_hands.hand_landmarks:
            hand_landmarks = detection_result_hands.hand_landmarks[0]
            thumb_tip_y = hand_landmarks[4].y
            thumb_mcp_y = hand_landmarks[2].y
            fingers_folded = (
                hand_landmarks[8].y > hand_landmarks[6].y and
                hand_landmarks[12].y > hand_landmarks[10].y and
                hand_landmarks[16].y > hand_landmarks[14].y and
                hand_landmarks[20].y > hand_landmarks[18].y
            )
            if fingers_folded:
                if thumb_tip_y < thumb_mcp_y:
                    gesture = "like"
                elif thumb_tip_y > thumb_mcp_y:
                    gesture = "dislike"

        # Подсчет эмоций
        smile = (mouthSmileLeft + mouthSmileRight) / 2
        frown = (mouthFrownLeft + mouthFrownRight) / 2
        dimple = (mouthDimpleLeft + mouthDimpleRight) / 2
        browDown = (browDownLeft + browDownRight) / 2
        eyeWide = (eyeWideLeft + eyeWideRight) / 2

        happy_score = smile + dimple
        sad_score = (frown + browInnerUp) * 2
        angry_score = browDown / 1.5
        surprised_score = jawOpen + eyeWide + browInnerUp

        max_score = max(happy_score, sad_score, angry_score, surprised_score)

        display_img = img_normal
        if max_score < 0.15:
            display_img = img_normal
        elif max_score == happy_score:
            display_img = img_happy
        elif max_score == sad_score:
            display_img = img_sad
        elif max_score == angry_score:
            display_img = img_angry
        elif max_score == surprised_score:
            display_img = img_surprised

        if gesture == "like":
            display_img = img_like
        elif gesture == "dislike":
            display_img = img_dislike

        display_img_resized = cv2.resize(display_img, (150, frame.shape[0]))
        combined = np.hstack((frame, display_img_resized))
        cv2.imshow("Window", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

face_landmarker.close()
hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()
