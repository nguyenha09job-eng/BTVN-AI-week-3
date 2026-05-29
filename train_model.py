import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

PATH_TRAIN = 'face/train'
PATH_VAL = 'face/valid'
IMG_SIZE = (200, 200)
BATCH = 32

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_and_resize_face(img):
    if np.max(img) <= 1.0:
        img = img * 255.0
    
    src_img = img.astype(np.uint8)
    height, width, _ = src_img.shape
    
    try:
        gray_scaled = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
        detected_rects = detector.detectMultiScale(gray_scaled, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(detected_rects) > 0:
            largest_face = max(detected_rects, key=lambda r: r[2] * r[3])
            fx, fy, fw, fh = largest_face
            
            offset_y = int(fh * 0.1)
            offset_x = int(fw * 0.1)
            
            y_start = max(0, fy - offset_y)
            y_end = min(height, fy + fh + offset_y)
            x_start = max(0, fx - offset_x)
            x_end = min(width, fx + fw + offset_x)
            
            face_region = src_img[y_start:y_end, x_start:x_end]
            final_face = cv2.resize(face_region, (width, height), interpolation=cv2.INTER_LINEAR)
            return final_face.astype(np.float32)
    except Exception:
        pass
        
    y_crop_start, y_crop_end = int(height * 0.175), int(height * 0.825)
    x_crop_start, x_crop_end = int(width * 0.175), int(width * 0.825)
    center_region = src_img[y_crop_start:y_crop_end, x_crop_start:x_crop_end]
    fallback_face = cv2.resize(center_region, (width, height), interpolation=cv2.INTER_LINEAR)
    return fallback_face.astype(np.float32)

if __name__ == '__main__':
    generator_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=extract_and_resize_face
    )

    generator_val = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=extract_and_resize_face
    )

    flow_train = generator_train.flow_from_directory(
        PATH_TRAIN,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical"
    )

    flow_val = generator_val.flow_from_directory(
        PATH_VAL,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical"
    )

    net = Sequential()
    
    net.add(Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4), input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
    net.add(Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)))
    net.add(BatchNormalization(momentum=0.9))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Dropout(0.2))

    net.add(Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)))
    net.add(Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)))
    net.add(BatchNormalization(momentum=0.9))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Dropout(0.2))

    net.add(Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)))
    net.add(Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)))
    net.add(BatchNormalization(momentum=0.9))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Dropout(0.25))

    net.add(Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)))
    net.add(Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)))
    net.add(BatchNormalization(momentum=0.9))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Dropout(0.3))

    net.add(GlobalAveragePooling2D())
    net.add(Dense(256, activation="relu", kernel_regularizer=l2(1e-4)))
    net.add(BatchNormalization(momentum=0.9))
    net.add(Dropout(0.5))
    net.add(Dense(31, activation="softmax"))

    net.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )
    
    net.summary()

    cb_early = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    cb_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=8,
        min_lr=1e-6,
        verbose=1
    )

    cb_checkpoint = ModelCheckpoint(
        filepath='face_classifier.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    train_history = net.fit(
        flow_train,
        epochs=30,
        validation_data=flow_val,
        callbacks=[cb_early, cb_plateau, cb_checkpoint]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(train_history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(train_history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    
    axes[1].plot(train_history.history['loss'], label='Train Loss')
    axes[1].plot(train_history.history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
