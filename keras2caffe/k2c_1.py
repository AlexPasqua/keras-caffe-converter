"""
Reads the net's weights and store them into a pickle
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, ReLU, MaxPooling2D, PReLU, Concatenate

import caffe
from caffe import layers as cl, params as cp

import numpy as np
import pickle
import argparse


def save_keras_weights(model_path, weights_path):
    """
    Reads the net's weights and store them into a pickle.

    Arguments:
        model_path: the path to the Keras model file
        weights_path: the path to the pickle file for storing the net's weights
    """

    keras_model = tf.keras.models.load_model(model_path)

    # Since both Keras and Caffe models don't fit in the memory, I create a dictionary with the weights to be copied
    weights = {}
    for i in range(len(keras_model.layers)):
        layer = keras_model.layers[i]

        # Skip pad layers and layers with no weights
        if layer.name[-3 :] == 'pad' or layer.get_weights() == []:
            continue

        key = (layer.name, layer.__class__.__name__)
        weights[key] = layer.get_weights()

    # Save the weights dictionary
    weights_path = weights_path + ('' if weights_path[-4 : ] == '.pkl' else '.pkl')
    with open(weights_path, 'wb') as f:
        pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reads the Keras network's parameters and create a picke to export"
    )
    parser.add_argument('keras_model', action='store', help="The filename (full path including extension) of the file that contains the Keras model.")
    parser.add_argument('weights_path', action='store', help="The filename (full path WITHOUT extension) of the file that will contain the Keras model's weights.")
    args = parser.parse_args()
    save_keras_weights(args.keras_model, args.weights_path)
