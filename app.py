from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2
import numpy as np
import json

app = Flask(__name__)

# Define the static and upload directories
STATIC_FOLDER = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the deep learning model
model = tf.keras.models.load_model(r'models\Best_DenseNet121.h5', compile=False)

# Emotion labels
with open(r'models\class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Assuming the model was trained with images rescaled to [0, 1]
    return img_array

def predict_image(model, img_array):
    prediction = model.predict(img_array)
    return prediction

def decode_prediction(prediction, class_names):
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

def predict_breed(img_path, model, class_names, target_size=(224, 224)):
    img_array = load_and_preprocess_image(img_path, target_size)
    prediction = predict_image(model, img_array)
    predicted_breed = decode_prediction(prediction, class_names)
    return predicted_breed

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction page route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Read and preprocess the image
            predicted_breed = predict_breed(filepath, model, class_names)
            print(f'The predicted breed is: {predicted_breed}')

            # Pass prediction and image file path to result template
            return render_template('result.html', prediction=predicted_breed, image_file=filename)

if __name__ == '__main__':
    # Ensure the upload directory exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
