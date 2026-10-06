import cv2
import numpy as np
import mediapipe as mp

def extract_hand_landmarks(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    
    if result.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))
        return np.array(landmarks)
    return None

def preprocess_image(image):
    image = cv2.resize(image, (640, 480))
    return image

def extract_features_from_images(image_paths):
    features = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = preprocess_image(image)
        landmarks = extract_hand_landmarks(image)
        if landmarks is not None:
            features.append(landmarks.flatten())
    return np.array(features)