# EmotionDetector

Real-time emotion recognition from webcam that displays a corresponding image based on your facial expression.

## Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11 | Programming language |
| OpenCV | 4.x | Camera capture, image processing, displaying video |
| MediaPipe | 0.10.x | Face detection, 468 facial landmarks, emotion blendshapes |
| NumPy | 1.x | Image array manipulation |
| Git | - | Version control |

## Key Components

- **FaceLandmarker** – MediaPipe model for face tracking and blendshape extraction
- **Blendshapes** – Facial expression coefficients (smile, brow_furrow, mouth_frown, brow_raise)
- **OpenCV** – Webcam access (`VideoCapture`), image resizing, overlay rendering

## Features

- Real-time emotion detection (happy, sad, angry, surprised, neutral)
- Reaction image overlay on video feed
- Works with any standard webcam

## Requirements

- Python 3.8 – 3.11
- Webcam

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your_username/EmotionDetector.git
cd EmotionDetector
