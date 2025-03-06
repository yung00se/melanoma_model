import cv2
import numpy as np
import tensorflow as tf
import os

model = None # Global for model

def load_model():
    """Load the pre trained model"""
    global model
    if model is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'model.keras')
        model = tf.keras.models.load_model(model_path)
    return model

def normalize_image(image, target_min=0.0, target_max=1.0):
    """idk how to normalize this shit"""


def predict_lesion(image):
    model = load_model()
    
    resize_image = cv2.resize(image, (256,256))
    # normalized image if we need to do that
    
    input_image = np.expand_dims(resize_image, axis=0)
    
    prediction = model.predict(input_image)[0][0]
    
    return prediction