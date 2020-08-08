import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

import argparse

import sys
sys.path.insert(1, 'caffe2keras')

import create_nn_struct


def write_beginning(outfile):
    import_modules = {
        'tensorflow': ['keras'],
        'tensorflow.keras': ['models'],
        'tensorflow.keras.layers': ['*']
    }
    outfile.write('import tensorflow as tf\n')
    for k in import_modules:
        for module in import_modules[k]:
            outfile.write('from {} import {}\n'.format(k, module))
    outfile.write("\n\ndef keras_model():\n")


def caffe2keras(prototxt_path, outfile_path):
    outfile_path = outfile_path + ('' if outfile_path[-3 : ] == '.py' else '.py')
    with open(outfile_path, 'w') as outfile:
        write_beginning(outfile)
    create_nn_struct.write_nn_struct_code_keras(prototxt_path, outfile_path)
    with open(outfile_path, 'a') as outfile:
        outfile.write("\n\n\treturn keras_model\n\n\n")
        outfile.write("if __name__ == '__main__':\n\tkeras_model()")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reads the Caffe network's defnition from its prototxt, the parameters from its caffemodel \
                    and generates a Python file containing the network (architecture + parameters) in Keras"
    )
    parser.add_argument('prototxt', action='store', help="The filename (full path including file extension) of the '.prototxt' file that defines the Caffe model.")
    parser.add_argument('caffemodel', action='store', help="The filename (full path including file extension) of the '.caffemodel' file that contains the network's parameters")
    parser.add_argument('outfile', action='store', help="The filename (full path WITHOUT extension) of the file where you want the code to be written in.")
    args = parser.parse_args()
    caffe2keras(args.prototxt, args.outfile)
