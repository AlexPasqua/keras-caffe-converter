"""
This script allows you to create ResNet50 (with imagenet pretrained weights) modified to perform
classifications on CIFAR10.
Furthermore it's possible to prune that model and evaluate the differences compared to the base one
(in terms of accuracy and evaluation time).

Arguments:
    -m --mode: {create | prune | evaluate}
        - create: it creates, train and save the model
        - prune: it loads the base model, prunes it and evaluates the differences with the base one
        - evaluate: load, compile and evaluate the model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import optimizers
import tensorflow_model_optimization as tfmot

import numpy as np
import argparse
import time


model_path = '../../models/resnet50_retrained_cifar10.h5'

img_width = img_height = 32
num_classes = 10

# Load cifar10
(train_imgs, train_lbls), (test_imgs, test_lbls) = tf.keras.datasets.cifar10.load_data()
train_imgs = train_imgs.astype('float32') / 255.0
test_imgs = test_imgs.astype('float32') / 255.0

# Convert class vectors to binary class matrices.
train_lbls = keras.utils.to_categorical(train_lbls, num_classes)
test_lbls = keras.utils.to_categorical(test_lbls, num_classes)

# Subtract the mean from the images
#train_imgs_mean = np.mean(train_imgs, axis=0)
#train_imgs -= train_imgs_mean
#test_imgs -= train_imgs_mean

print('train_imgs shape:', train_imgs.shape)
print(train_imgs.shape[0], 'train samples')
print(test_imgs.shape[0], 'test samples')
print('train_lbls shape:', train_lbls.shape)


def create():
    """
    This function imports ResNet50 from Keras.applications and load it with pretrained imagenet weights.
    Then it adds some layer to adapt the model for performing classifications on cifar10, specifically it adds:
    - Flatten
    - BatchNormalization
    - Dense (fully connected) with 128 neurons, activation ReLU
    - Dropout of 0.5
    - BatchNormalization
    - Dense (fully connected) with 64 neurons, activation ReLU
    - Dropout of 0.5
    - BatchNormalization
    - Dense (fully connected) with 10 neurons (= number of classes), activation Softmax

    Then the function compiles, fits and saves the model.
    """

    # ALTERNATIVE 1: work with small images, it takes more epochs to get a good accuracy and pruning will be possible
    # Comment this block and uncomment ALTERNATIVE 2 if you wish to use the other method
    resnet = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
    inp = resnet.input
    out = layers.Flatten()(resnet.output)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(128, activation='relu')(out)
    out = layers.Dropout(0.5)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(64, activation='relu')(out)
    out = layers.Dropout(0.5)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(10, activation='softmax')(out)
    model = Model(inp, out)
    model.summary()

    """
    # ALTERNATIVE 2: work with upsampled images, it takes few epochs to get a good accuracy but it won't be possible
    # to prune ResNet50's layers, this because ResNet50 itself is seen as a single layer in a Sequential model.
    # Comment this block and uncomment ALTERNATIVE 1 if you wish to use the other method
    resnet = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
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
    """

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

    history = model.fit(train_imgs, train_lbls, epochs=20, batch_size=20, validation_data=(test_imgs, test_lbls), use_multiprocessing=True)

    model.save(model_path)


def evaluate():
    """ Load model, compile it and evaluate it with the test dataset """
    model = load_model(model_path)
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])
    model.evaluate(test_imgs, test_lbls, batch_size=20)
    # Single prediction
    # print(f'Predicted class: {np.argmax(model.predict(test_imgs[0][np.newaxis, ...]))}')
    # print(f'Actual class: {np.argmax(test_lbls[0])}')


def apply_pruning(layer):
    """
    This functions is called for every layer during a model cloning.
    If the current layer is a Conv2D (convolutional) or Dense (fully connected), the functions
    returns a 'Prunable layer' i.e. a layer with pruning functionalities,
    otherwise it returns the layer as it was passed to the function.
    """
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
        return prune_low_magnitude(layer, pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(0.5, 0))
    return layer


def prune():
    """
    Pruning function.
    Keras pruning is based on the training of a prunable model.
    We can get a prunable model from a base one using Keras' tfmot.sparsity.keras.prune_low_magnitude
    """
    model = load_model(model_path)
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

    # Set up some parameters for pruning and fitting
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    num_images = train_imgs.shape[0]
    epochs = 1
    batch_size = 20
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                 final_sparsity=0.80,
                                                                 begin_step=0,
                                                                 end_step=end_step)
    }

    # Clone the model. Each layer is passed to 'apply_pruning' before going to be part of pruned_model
    pruned_model = tf.keras.models.clone_model(model, clone_function=apply_pruning)
    pruned_model.summary()

    # Compile and fit the pruned_model to effectively apply pruning
    pruned_model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])
    pruned_model.fit(train_imgs, train_lbls,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(test_imgs, test_lbls),
                      callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
    )

    # test on evaluation time
    t1 = time.time()
    _, base_acc = model.evaluate(test_imgs, test_lbls, batch_size=batch_size)
    t2 = time.time()
    base_model_eval_time = t2 - t1
    t1 = time.time()
    _, pruned_acc = pruned_model.evaluate(test_imgs, test_lbls, batch_size=batch_size)
    t2 = time.time()
    pruned_model_eval_time = t2 - t1
    print(f'Base model accuracy: {base_acc}\nPruned model accuracy: {pruned_acc}\n')
    print(f'Base model evaluation time: {base_model_eval_time}\nPruned model evaluation time: {pruned_model_eval_time}')

    # Strip from the model the pruning wrapper and save it
    pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    pruned_model.save('../../models/resnet50_retrained_cifar10_pruned.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script either creates and train a ResNet50 for CIFAR10 or it evaluates that model (if already existing)")
    parser.add_argument('-m', '--mode', action='store', type=str, choices={'create', 'evaluate', 'prune'}, default='create', help='Mode')
    args = parser.parse_args()
    if args.mode == 'create': create()
    elif args.mode == 'evaluate': evaluate()
    elif args.mode == 'prune': prune()
