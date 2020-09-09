import tensorflow as tf
from tensorflow import keras
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
    # Build model
    tf.keras.backend.set_floatx(dtype)
    odel = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
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

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    model.save(model_filename)
    print(f'Saved model to {model_filename}')


def predict(model_filename, train_images, train_labels, test_images, test_labels):
    model = tf.keras.models.load_model(model_filename)
    start_time = time.time()
    predictions = model.predict(train_images)
    end_time = time.time()
    print(f"\nTime for prediction of {len(train_labels)} images: {end_time - start_time} seconds")
    # index = 0
    # print('\nResults:')
    # for i in range(len(test_labels)):
        # if predicted != actual...
        # if np.argmax(predictions[i]) != test_labels[i]:
        #    print("\nFound wrong prediction!")
        #    print("Predicted: {:20s}\tActual: {:20s}".format(
        #        class_names[ np.argmax(predictions[i]) ], class_names[test_labels[i]]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="If it doesn't already exists, this scripts creates and train a small classifier for Fashion MNIST"
    )
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

    predict(model_filename, train[0], train[1], test[0], test[1])
