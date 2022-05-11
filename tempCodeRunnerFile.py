
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

def predict_class(image):
    print(image.shape)
    probabilities = model.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}
