import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_images():
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

    # Process the images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)


def build_and_save_model(train_images, train_labels, test_images, test_labels):
    # Build model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    # Compile and train the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print('\nTraining:')
    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    # Attach a softmax layer to the model to make probability predictions
    model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    model.save('classifier_fashionMNIST.h5')
    print('Saved model to classifier_fashionMNIST.h5')


def predict(train_images, train_labels, test_images, test_labels):
    model = tf.keras.models.load_model('classifier_fashionMNIST.h5')
    predictions = model.predict(test_images)
    index = 0
    print(f'\nPredict image {index}: {class_names[ np.argmax(predictions[index]) ]}\n')


if __name__ == '__main__':
    train, test = load_images()
    import os.path
    if not os.path.exists('classifier_fashionMNIST.h5'):
        build_and_save_model(train[0], train[1], test[0], test[1])
    predict(train[0], train[1], test[0], test[1])
