from tensorflow.keras.models import load_model 
import os
new_model = load_model(os.path.join('models','final_model_2.h5'))
import numpy as np
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
img = cv2.imread('bio.jpg')
resize = tf.image.resize(img, (256,256))
yhat = new_model.predict(np.expand_dims(resize/255, 0))
if yhat > 0.5: 
    print(f'Predicted class is Non-Biodegradable')
else:
    print(f'Predicted class is Biodegradable')