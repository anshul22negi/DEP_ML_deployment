import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.models import model_from_yaml
import cv2
import os
import tensorflow_hub as hub

yaml_file = open('mobilenet_model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml, custom_objects={'KerasLayer': hub.KerasLayer})
# load weights into new model
model.load_weights("mobilenet_model.h5")
print("Loaded model from disk")

#model_save_file = 'mobilenet_model.sav'
#model = tf.keras.models.load_model(model_save_file)

IMAGE_SHAPE = (224, 224)
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
    print(image.shape)
    probabilities = model.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}

top=tk.Tk()
top.geometry('800x600')
top.title('Plant Disease Classfier')
top.configure(background='#CDCDCD')
label=tk.Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = tk.Label(top)

def load_image(filename):
    img = cv2.imread(filename)
    #img = np.squeeze(img, axis=1)
    img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
    img = img /255
    
    return img

def classify(file_path):
    global label_packed
    image = load_image(file_path)
    #image = image.resize((224,224))
    #image = np.expand_dims(image, axis=0)
    #image = np.array(image)
    prediction = predict(image)
    sign = "PREDICTED: class: %s, confidence: %f" % (list(prediction.keys())[0], list(prediction.values())[0])
    print("PREDICTED: class: %s, confidence: %f" % (list(prediction.keys())[0], list(prediction.values())[0]))
    label.configure(foreground='#011638', text=sign)

def show_classify_button(file_path):
    classify_b=tk.Button(top,text="Classify Image", command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=tk.Button(top,text="Upload an image",command=upload_image, padx=10,pady=5)
upload.configure(background='#364156', foreground='white', font=('arial',10,'bold'))
upload.pack(side=tk.BOTTOM,pady=50)
sign_image.pack(side=tk.BOTTOM,expand=True)
label.pack(side=tk.BOTTOM,expand=True)
heading = tk.Label(top, text="Plant Disease Classifier",pady=20, font=('arial',20,'bold'))

heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()