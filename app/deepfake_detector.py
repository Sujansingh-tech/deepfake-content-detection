import cv2
import numpy as np
import tensorflow as tf
import os

model_path = os.path.join('model', 'deepfake_model.h5')
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_stream):
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))  # adjust size if needed
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def detect_deepfake(image_stream):
    img = preprocess_image(image_stream)
    prediction = model.predict(img)[0][0]
    return "Deepfake" if prediction > 0.5 else "Real"
