import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

import argparse

import os
from os import system


def keras2caffe(keras_model_path, prototxt_path, caffemodel_path):
    if prototxt_path == None:
        prototxt_path = 'generated_prototxt'
        cmd = 'python3 keras2caffe/create_prototxt.py ' + keras_model_path + ' ' + prototxt_path
        os.system(cmd)
    cmd = 'python3 keras2caffe/k2c_1.py ' + keras_model_path + ' km_weights.pkl'
    os.system(cmd)
    cmd = 'python3 keras2caffe/k2c_2.py ' + prototxt_path + ' ' + ' km_weights.pkl ' + caffemodel_path
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reads the Keras model and create an equivalent Caffe one")
    parser.add_argument('keras_model', action='store', help="The filename (full path including extension) of the file that contains the Keras model.")
    parser.add_argument('caffemodel', action='store', help="The filename (full path WITHOUT extension) of the file where to save the Caffe model")
    parser.add_argument('--prototxt', action='store', help="The filename (full path including file extension) of the '.prototxt' file that defines the Caffe model.")
    args = parser.parse_args()
    keras2caffe(args.keras_model, args.prototxt, args.caffemodel)
