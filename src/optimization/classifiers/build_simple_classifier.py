"""
Demo script to create a simple classifier for FashionMNIST and prune it
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import matplotlib.pyplot as plt
import os.path
import argparse
import time


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_images(dtype, train_images, test_images):
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if dtype == 'float16':
        train_images = np.float16(train_images)
        test_images = np.float16(test_images)
    elif dtype == 'float32':
        train_images = np.float32(train_images)
        test_images = np.float32(test_images)
    elif dtype == 'float64':
        train_images = np.float64(train_images)
        test_images = np.float64(test_images)

    return train_images, test_images


def load_images(dtype):
    # Import Fashion MNIST dataset
    '''
    Classes:
        0	T-shirt/top
        1	Trouser
        2	Pullover
        3	Dress
        4	Coat
        5	Sandal
        6	Shirt
        7	Sneaker
        8	Bag
        9	Ankle boot
    '''
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    """
    # Show example image from dataset
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    """

    # Preprocess the images
    train_images, test_images = preprocess_images(dtype, train_images, test_images)

    return (train_images, train_labels), (test_images, test_labels)


def build_and_save_model(model_filename, dtype, train_images, train_labels, test_images, test_labels):
    """
    Build, compile, fit, evaluate and save model
    Arguments:
        model_filename: the path to the model
        dtype: a string indicatind the data type (float16 | float32 | float64)
        train_images, train_labels, test_images, test_labels: the dataset as return by 'load_images'
    """
    # Build model
    tf.keras.backend.set_floatx(dtype)
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10),
        keras.layers.Softmax()
    ])

    # Compile and train the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print('\nTraining:')
    model.fit(train_images, train_labels, epochs=10)

    # Evaluation
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    model.save(model_filename)
    print(f'Saved model to {model_filename}')


def predict(model_filename, train_images, train_labels, test_images, test_labels):
    """
    Load a model and perform a single prediction
    """
    model = tf.keras.models.load_model(model_filename)
    start_time = time.time()
    predictions = model.predict(train_images)
    end_time = time.time()
    print(f"\nTime for prediction of {len(train_labels)} images: {end_time - start_time} seconds")


def prune(model, train_images, train_labels):
    """
    Prune the model
    Arguments:
        model: the base model to prune (Keras' Model object)
    """
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    batch_size = 128
    epochs = 2
    validation_split = 0.1
    num_images = train_images.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define the pruning chedule (and eventaully other pruning parameters)
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                 final_sparsity=0.80,
                                                                 begin_step=0,
                                                                 end_step=end_step)
    }
    # Alternative pruning schedule: ConstantSparsity
    # pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.8, 0)}

    # Get a 'Pruneble Model' from the base model
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    # Callbacks for training. UpdatePruningStep() is necessary for pruning
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    # Training with pruning callbacks
    model_for_pruning.fit(train_images, train_labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_split=validation_split,
                          callbacks=callbacks)

    return model_for_pruning


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="If it doesn't already exists, this scripts creates and train a small classifier for Fashion MNIST")
    parser.add_argument(
        '-dt', '--data_type', action='store', type=str, choices={'float16', 'float32', 'float64'}, default='float32',
        help="A data type [float16 | float32 | float64]"
    )
    args = parser.parse_args()

    # Load images
    train, test = load_images(args.data_type)

    # Create the model if it doesn't exist
    model_filename = 'classifier_fashionMNIST_' + args.data_type + '.h5'
    if not os.path.exists(model_filename):
        build_and_save_model(model_filename, args.data_type, train[0], train[1], test[0], test[1])

    # Load and compile model
    model = tf.keras.models.load_model(model_filename)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Pruning
    model_for_pruning = prune(model, train[0], train[1])

    # comparison between base model and pruned model
    t = time.time()
    _, base_model_accuracy = model.evaluate(test[0], test[1], verbose=2)
    time_base_model = time.time() - t

    t = time.time()
    _, model_for_pruning_accuracy = model_for_pruning.evaluate(test[0], test[1], verbose=0)
    time_pruned_model = time.time() - t

    print('\n\nBase model accuracy: ', base_model_accuracy)
    print('Pruned test accuracy: ', model_for_pruning_accuracy)
    print(f'Base model evaluation time: {time_base_model}\nPruned model evaluation time: {time_pruned_model}')

    # Export the model
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    model_for_export.save('/storage/keras-caffe_converter_optimizer/models/simple_classifier_PRUNED.h5', include_optimizer=False)
