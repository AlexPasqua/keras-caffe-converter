import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions

import numpy as np
import matplotlib.pyplot as plt
import os.path
import argparse
import time


# Images
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# load an image from file
# image = load_img('/storage/mug.jpg', target_size=(224, 224))

imagegen = ImageDataGenerator()
train = imagegen.flow_from_directory("/storage/imagenette2/train/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))
print('train loaded')
val = imagegen.flow_from_directory("/storage/imagenette2/val/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))
print('val loaded')

print(np.shape(train))

# convert the image pixels to a numpy array
# image = img_to_array(image)
# image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) # we make it 4-dimensionale

# prepare the image for the VGG model
# image = preprocess_input(image)

# Model
"""model = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
model.summary()

print('Ttraining:')
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(train, epochs=30, validation_data=val)"""

"""print('Predicting:')
predictions = model.predict(val)
labels = decode_predictions(predictions)
print(np.shape(labels))
sum = 0
for i in range(1000):
    sum = sum + labels[i][0][2]
avg_conf = sum / 1000.0"""
# print(f"Avg confidence: {avg_conf * 100.0}%")
