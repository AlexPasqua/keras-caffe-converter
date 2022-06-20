""" From a Keras model, creates and equivalent Caffe one """

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

import os
from os import system

import argparse


def keras2caffe(keras_model_path, output_dir, prototxt_path, caffemodel_name):
    """
    From a Keras model, creates and equivalent Caffe one

    Arguments:
        keras_model_path: the path to the keras model
        output_dir: the directory where to save the caffemodel and the prototxt if necessary
        prototxt_path: the path to the prototxt (optional: if missing, a prototxt will be created)
        caffemodel_name: The name (without extension) of the file where to save the Caffe model
    """

    output_dir = output_dir + ('/' if output_dir[-1] != '/' else '')

    if prototxt_path == None:
        prototxt_path = output_dir + 'generated_prototxt.prototxt'
        cmd = 'python3 keras2caffe/create_prototxt.py ' + keras_model_path + ' ' + prototxt_path
        os.system(cmd)
    cmd = 'python3 keras2caffe/k2c_1.py ' + keras_model_path + ' ' + output_dir + 'km_weights.pkl'
    os.system(cmd)

    if caffemodel_name == None:
        caffemodel_path = output_dir + 'generated_caffemodel'
    else:
        caffemodel_path = output_dir + caffemodel_name
    cmd = 'python3 keras2caffe/k2c_2.py ' + prototxt_path + ' ' + output_dir + 'km_weights.pkl ' + caffemodel_path
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reads the Keras model and create an equivalent Caffe one")
    parser.add_argument('keras_model', action='store', help="The filename (full path including extension) of the file that contains the Keras model.")
    parser.add_argument('output_dir', action='store', help="The path to the output directory where to save the caffemodel (and prototxt if necessary)")
    parser.add_argument('--caffemodel_name', action='store', help="The name (without extension) of the file where to save the Caffe model")
    parser.add_argument('--prototxt', action='store', help="The filename (full path including file extension) of the '.prototxt' file that defines the Caffe model.")
    args = parser.parse_args()
    keras2caffe(args.keras_model, args.output_dir, args.prototxt, args.caffemodel_name)
