import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os.path
import argparse
import time

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

model = tf.keras.applications.VGG19(
    include_top=False,
    weights='imagenet',
    input_shape=(32,32,3),
    classifier_activation='relu'
)

start_time = time.time()
predictions = model.predict(test_images)
end_time = time.time()
print(f"\nTime for prediction of {len(test_labels)} images: {end_time - start_time} seconds")

model.save('vgg19_cifar10_classifier.h5')
