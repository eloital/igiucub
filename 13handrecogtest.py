from keras.models import model_from_json
import numpy as np
import keras
# from keras.preprocessing import image# preprocessing images from test dataset
from keras.utils import img_to_array, load_img


#loading the weights of model file in  directory
model=keras.models.load_model('model.h6')
print("Loaded model from disk")

def classify(img_file):
    # img_name = img_file
    img_name = img_file
    test_image = load_img(img_name, target_size = (256, 256),color_mode="grayscale")
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    arr = np.array(result[0])
    print(arr)
    maxx = np.amax(arr)
    max_prob = arr.argmax(axis=0)
    max_prob = max_prob + 1
    classes=["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
    result = classes[max_prob - 1]
    print(img_name,result)

import os
path = 'C:/Users/volet/OneDrive/Desktop/Artificial intelligence/Dataset2/test'
files = []#empty array
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
     if '.png' in file:
       files.append(os.path.join(r, file))

for f in files:
   classify(f)
   print('\n')