import cv2
import mediapipe as mp
import numpy as np

img_normal = cv2.imread("Images/normal.jpg")
img_sad = cv2.imread("Images/sad.webp")
img_squint = cv2.imread("Images/squint.jpg")

#Resize

img_normal = cv2.resize(img_normal, (150,150))
img_sad = cv2.resize(img_sad, (150,150))
img_squint = cv2.resize(img_squint, (150,150))

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
