import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import decode_predictions
import cv2
import pickle
import os

model_save_file = 'mobilenet_model.sav'




model = pickle.load(open(model_save_file, 'rb'))
IMAGE_SHAPE = (224, 224)

def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    img = Image.open(image_data)
    img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
    img = img /255
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()

BATCH_SIZE = 32
data_dir = 'dataset'
train_dir = os.path.join(data_dir, 'train')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
  rescale = 1./255,
  rotation_range=40,
  horizontal_flip=True,
  width_shift_range=0.2, 
  height_shift_range=0.2,
  shear_range=0.2, 
  zoom_range=0.2,
  fill_mode='nearest' )

train_generator = train_datagen.flow_from_directory(
    train_dir, 
    subset="training", 
    shuffle=True, 
    seed=42,
    color_mode="rgb", 
    class_mode="categorical",
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE)

classes = {j: i for i, j in train_generator.class_indices.items()}

def predict(image):
    probabilities = model.predict(np.asarray([img]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}

def classify():
    original = Image.open(image_data)
    prediction = predict(img)
    table = tk.Label(frame, text=("PREDICTED: class: %s, confidence: %f" % (list(prediction.keys())[0], list(prediction.values())[0]))).pack()









root = tk.Tk()
root.title('Plant Disease Classifier')
root.iconbitmap('class.ico')
root.resizable(False, False)
tit = tk.Label(root, text="Plant Disease Classifier", padx=25, pady=6, font=("", 12)).pack()
canvas = tk.Canvas(root, height=500, width=500, bg='grey')
canvas.pack()
frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
chose_image = tk.Button(root, text='Choose Image',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=load_img)
chose_image.pack(side=tk.LEFT)
class_image = tk.Button(root, text='Classify Image',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=classify)
class_image.pack(side=tk.RIGHT)

root.mainloop()

