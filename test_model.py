import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from train_model import extract_and_resize_face

MODEL_FILE = 'face_classifier.keras'
TRAINING_CLASSES_DIR = 'train_test'
IMG_RESOLUTION = (200, 200)

def evaluate_image(img_path):
    if not os.path.exists(img_path):
        print(f"Error: Target image {img_path} not found.")
        return
        
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Trained model weights at {MODEL_FILE} not found.")
        return

    print(f"Inferencing {img_path} using model...")
    loaded_net = load_model(MODEL_FILE)
    
    if not os.path.exists(TRAINING_CLASSES_DIR):
        print(f"Error: Source classes directory {TRAINING_CLASSES_DIR} not found.")
        return
    
    label_folders = sorted([item for item in os.listdir(TRAINING_CLASSES_DIR) if os.path.isdir(os.path.join(TRAINING_CLASSES_DIR, item))])

    loaded_image = load_img(img_path, target_size=IMG_RESOLUTION)
    array_image = np.asarray(loaded_image)
    cropped_face = extract_and_resize_face(array_image)
    normalized_input = np.expand_dims(cropped_face / 255.0, axis=0)
    
    logits = loaded_net.predict(normalized_input)[0]
    best_match_idx = np.argmax(logits)
    match_label = label_folders[best_match_idx]
    confidence = logits[best_match_idx]

    print(f"\nResult classification: {match_label}")

    plt.imshow(cropped_face.astype(np.uint8))
    plt.title(f"Label: {match_label}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    evaluate_image('hang1.jpg')
