import os
import numpy as np
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt


# Load the trained model
model = load_model('animal_vgg16.hdf5')  # If you have saved the model earlier

# Class names
class_names = ['birds','cats','dogs','panda']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to match the training condition
    return img_array

# Function to predict the class of the image
def predict_image(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class, np.max(prediction)
# Specify the directory containing test images

 
test_directory = '4animals'

# Get all image file paths from the directory
all_test_images = [os.path.join(test_directory, img) for img in os.listdir(test_directory) if img.lower().endswith(('png', 'jpg', 'jpeg'))]

# Randomly select 10 images from the directory
test_images = random.sample(all_test_images, 10)


for img_path in test_images:
    predicted_class, confidence = predict_image(img_path)
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')
    plt.show()