import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

img_normal = cv2.imread("Images/normal.jpg")
img_sad = cv2.imread("Images/sad.webp")
img_squint = cv2.imread("Images/squint.jpg")

#Resize

img_normal = cv2.resize(img_normal, (150,150))
img_sad = cv2.resize(img_sad, (150,150))
img_squint = cv2.resize(img_squint, (150,150))

options = FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True
)

face_landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)


while(True):
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    cv2.imshow("Window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            for i, landmark in enumerate(face_landmarks):
                x = landmark.x
                y = landmark.y
                z = landmark.z

face_landmarker.close()
cap.release()
cv2.destroyAllWindows()