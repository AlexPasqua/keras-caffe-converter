import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers

import numpy as np


img_width = img_height = 200
num_classes = 10

(train_imgs, train_lbls), (test_imgs, test_lbls) = tf.keras.datasets.cifar10.load_data()
train_imgs = train_imgs.astype('float32') / 255.0
test_imgs = test_imgs.astype('float32') / 255.0

# Convert class vectors to binary class matrices.
train_lbls = keras.utils.to_categorical(train_lbls, num_classes)
test_lbls = keras.utils.to_categorical(test_lbls, num_classes)

#train_imgs_mean = np.mean(train_imgs, axis=0)
#train_imgs -= train_imgs_mean
#test_imgs -= train_imgs_mean

print('train_imgs shape:', train_imgs.shape)
print(train_imgs.shape[0], 'train samples')
print(test_imgs.shape[0], 'test samples')
print('train_lbls shape:', train_lbls.shape)


resnet = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))

model = Sequential()
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(resnet)
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

history = model.fit(train_imgs, train_lbls, epochs=5, batch_size=20, validation_data=(test_imgs, test_lbls))



# Freeze the layers which you don't want to train. Here I am freezing the all layers.
#for layer in resnet.layers[:]:
    #layer.trainable = False


# complete the model to be used with cifar10
"""out = resnet.output
pred = Dense(1000)(out)
pred = Dense(100)(pred)
pred = Dense(10, activation='softmax', name='pred')(pred)
model = Model(inputs=resnet.input, outputs=pred)
model.summary()

# compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizers.SGD(), metrics=["accuracy"])

# train the model
model.fit(train_imgs, train_lbls, epochs=2, verbose=1, validation_split=0.1)"""
