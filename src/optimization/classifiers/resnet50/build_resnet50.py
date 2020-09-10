import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import decode_predictions
import tensorflow_model_optimization as tfmot
import numpy as np


model = tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)
model.summary()

# Load dataset
imagegen = ImageDataGenerator()
train = imagegen.flow_from_directory("/storage/imagenette2/train/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))
print('train loaded')
val = imagegen.flow_from_directory("/storage/imagenette2/val/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))
print('val loaded')

print('Predicting:')
predictions = model.predict(val)
labels = decode_predictions(predictions)
sum = 0
for i in range(1000):
    sum = sum + labels[i][0][2]
avg_conf = sum / 1000.0
print(avg_conf)
