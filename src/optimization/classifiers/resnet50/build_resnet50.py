import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import decode_predictions
import tensorflow_model_optimization as tfmot
import numpy as np


resnet = tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)
model = Sequential([resnet, Dense(10, 'softmax')])
model.summary()

# Load dataset
print('\nLoading dataset:')
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='/storage/imagenette2/train/',
    labels='inferred',
    image_size=(224,224),
    label_mode='int',
    batch_size=17,
    #validation_split=0.2,
    #subset='training',
    shuffle=False
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='/storage/imagenette2/val/',
    labels='inferred',
    image_size=(224,224),
    label_mode='int',
    batch_size=25,
    #validation_split=0.9,
    #subset="validation",
    shuffle=False
 )

"""print(np.shape(list(train_ds)))
print(np.shape(list(train_ds)[0]))
print(np.shape(list(train_ds)[0][1]))"""

"""for image in val_ds:
    #print(np.shape(image))
    #print(np.shape(image[0]))
    #print(np.shape(image[1]))
    print(image[0][0][0][0][0])
    break"""

'''for image_batch, labels_batch in dataset:
  print(image_batch.shape)
  print(labels_batch.shape)
  print()'''

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
  #batch_size=32
)

model.save('resnet50.h5')
