import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

import argparse


def caffe2keras(argv):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reads the Caffe network's defnition from its prototxt, the parameters from its caffemodel \
                    and generates a Python file containing the network (architecture + parameters) in Keras"
    )
    parser.add_argument('prototxt', action='store', help="The filename (full path including file extension) of the '.prototxt' file that defines the Caffe model.")
    parser.add_argument('caffemodel', action='store', help="The filename (full path including file extension) of the '.caffemodel' file that contains the network's parameters")
    parser.add_argument('outfile', action='store', help="The filename (full path WITHOUT extension) of the file where you want the code to be written in.")
    args = parser.parse_args()
    caffe2keras(args)
