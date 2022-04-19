import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow_hub as hub
from os.path import dirname, abspath
import cv2
import json

app = Flask(__name__)
def get_model():
    global model
    """
    json_file = open('mobilenet_model.yaml', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={'KerasLayer': hub.KerasLayer})
    """
    model_file_path = os.path.join(dirname(__file__), 'models','train_model.h')
    model  = load_model(model_file_path)
    # load weights into new model
    print("Model loaded!")
    
def load_image(img_path):
    
    img = cv2.imread(img_path)
    #img = np.squeeze(img, axis=1)
    img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
    img = img /255
    
    return img


IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32

with open('classes.json') as json_file:
    classes = json.load(json_file)
    
def predict_class(image):
    print(image.shape)
    probabilities = model.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}




def prediction(img_path):
    new_image = load_image(img_path)
    
    pred = predict_class(new_image)
    
    ans = "PREDICTED: class: %s, confidence: %f" % (list(pred.keys())[0], list(pred.values())[0])
    return ans
    
get_model()

@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    
    if request.method == 'POST':
        
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join('static', filename)                       #slashes should be handeled properly
        file.save(file_path)
        print(filename)
        product = prediction(file_path)
        print(product)
        
    return render_template('predict.html', product = product, user_image = file_path)            #file_path can or may used at the place of filename

if __name__ == "__main__":
    app.run()