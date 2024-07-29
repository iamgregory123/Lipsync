import cv2
import numpy as np

def detect_lips(frame, target_shape=(75, 46, 140, 1)):
    # Assume frame is already in grayscale

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Initialize an empty region of interest
    lips_roi = np.zeros((46, 140), dtype=np.uint8)  # Change to your target ROI size

    # Process the first detected face (assuming the most relevant face is detected first)
    if len(faces) > 0:
        x, y, w, h = faces[0]

        # Calculate the region for the mouth based on the detected face
        mouth_y_start = y + int(0.75 * h)
        mouth_y_end = y + h
        mouth_x_start = x
        mouth_x_end = x + w

        # Crop the mouth region from the frame
        mouth_roi = frame[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]

        # Resize the mouth ROI to the fixed dimensions (46, 140)
        if mouth_roi.size > 0:
            lips_roi = cv2.resize(mouth_roi, (140, 46))

    return lips_roi