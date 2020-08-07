from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/fer2013_mini_XCEPTION.119-0.65.hdf5'
#haarcascade_path = 'models/haarcascade_frontalface_default.xml'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
#face_cascade = cv2.CascadeClassifier(haarcascade_path)

# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    #faces = sorted(faces, reverse=True,
     #       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    #(fX, fY, fW, fH) = faces
                # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
    #roi = gray[fY:fY + fH, fX:fX + fW]'''
    roi = cv2.resize(gray, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    #img = image.load_img(img_path, target_size=(64, 64))
    # Preprocessing the image
    #x = image.img_to_array(img)
    
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x)

    preds = model.predict(roi)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
        label = EMOTIONS[preds.argmax()]

        # Process your result for human
          # Convert to string
        return label
    return None


if __name__ == '__main__':
    app.run(debug=True)
