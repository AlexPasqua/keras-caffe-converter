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


model_path = '../../../models/resnet50_retrained_cifar10.h5'

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


def create():
    resnet = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))

    # Freeze the layers we don't want to train
    #for layer in resnet.layers[:]:
    #    layer.trainable = False

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

    """model = Sequential()
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

    history = model.fit(train_imgs, train_lbls, epochs=5, batch_size=20, validation_data=(test_imgs, test_lbls), use_multiprocessing=True)

    model.save(model_path)"""


def evaluate():
    model = load_model(model_path)
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])
    model.evaluate(test_imgs, test_lbls, batch_size=20)
    # Single prediction
    # print(f'Predicted class: {np.argmax(model.predict(test_imgs[0][np.newaxis, ...]))}')
    # print(f'Actual class: {np.argmax(test_lbls[0])}')


def apply_pruning(layer):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
        return prune_low_magnitude(layer)
    return layer


def prune():
    model = load_model(model_path)
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

    pruned_model = tf.keras.models.clone_model(model, clone_function=apply_pruning)
    pruned_model.summary()

    #pruned_model = prune_low_magnitude(model, **pruning_params)
    pruned_model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])
    pruned_model.fit(train_imgs, train_lbls,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(test_imgs, test_lbls),
                      callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
    )
    #pruned_model.evaluate(test_imgs, test_lbls, batch_size=batch_size)
    pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    pruned_model.save('../../../models/resnet50_retrained_cifar10_pruned.h5')

    # test on evaluation time
    t1 = time.time()
    _, base_acc = model.evaluate(test_imgs, test_lbls, batch_size=batch_size)
    t2 = time.time()
    base_model_eval_time = t2 - t1
    t1 = time.time()
    pruned_model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])
    _, pruned_acc = pruned_model.evaluate(test_imgs, test_lbls, batch_size=batch_size)
    t2 = time.time()
    pruned_model_eval_time = t2 - t1
    print(f'Base model accuracy: {base_acc}\nBase model evaluation time: {base_model_eval_time}\n')
    print(f'Pruned model accuracy: {pruned_acc}\nPruned model evaluation time: {pruned_model_eval_time}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script either creates and train a ResNet50 for CIFAR10 or it evaluates that model (if already existing)")
    parser.add_argument('-m', '--mode', action='store', type=str, choices={'create', 'evaluate', 'prune'}, default='create', help='Mode')
    args = parser.parse_args()
    if args.mode == 'create': create()
    elif args.mode == 'evaluate': evaluate()
    elif args.mode == 'prune': prune()
