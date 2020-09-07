""" Converts a caffe model into a Keras equivalent one """

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

import os.path
from pathlib import Path
import argparse

import sys
sys.path.insert(1, 'caffe2keras')
sys.path.insert(1, 'caffe2keras/caffe_weight_converter')

import create_nn_struct


def write_beginning(outfile):
    """
    Writes the initial part of the source file to generate the Keras model (imports etc)
    Arguments:
        outfile: a file object to write the code in
    """
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


def write_src(prototxt_path, output_dir):
    """
    Writes the Python source file defining the Keras model
    Arguments:
        prototxt_path: the path to the model's prototxt file
        outsrc_path: the path to the output Python source file defining the Keras model
    """

    # Create the output directory (if it doens't exist)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    outsrc_path = output_dir + 'net_caffe2keras.py'
    with open(outsrc_path, 'w') as outfile:
        write_beginning(outfile)
    create_nn_struct.write_nn_struct_code_keras(prototxt_path, outsrc_path)
    with open(outsrc_path, 'a') as outfile:
        outfile.write("\n\n\treturn keras_model\n\n\n")
        outfile.write(f"if __name__ == '__main__':\n\tkeras_model()")


def caffe2keras(prototxt_path, caffemodel_path, output_dir, verbose):
    """
    Converts a caffe model into a Keras qìequivalent one
    Arguments:
        prototxt_path: the path to the model's prototxt file
        outsrc_path: the path to the output Python source file defining the Keras model
    """

    output_dir = output_dir + ('/' if output_dir[-1] != '/' else '')

    # Write the Python source file containing the keras model
    write_src(prototxt_path, output_dir)

    # Import the source file we just created
    import sys
    sys.path.insert(1, output_dir)
    import net_caffe2keras

    # Take the model from the source file
    keras_model = net_caffe2keras.keras_model()
    if verbose: keras_model.summary()

    # Export weights from the Caffe net, load them into the Keras model and save the model
    args = argparse.Namespace()
    args.out_file = output_dir + 'weights'
    args.prototxt = prototxt_path
    args.caffemodel = caffemodel_path
    args.format = 'hdf5'
    args.include_non_weight = True
    args.skip_unknown = True
    args.backend = 'tf'
    if verbose: args.verbose = True
    else: args.verbose = False
    import caffe_weight_converter
    caffe_weight_converter.convert_caffe_weights(args)
    keras_model.load_weights(output_dir + 'weights.h5')
    keras_model.save(output_dir + 'keras_model.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reads the Caffe network's defnition from its prototxt, the parameters from its caffemodel \
                    and generates a Python file containing the network (architecture + parameters) in Keras"
    )
    parser.add_argument('prototxt', action='store', help="The filename (full path including file extension) of the '.prototxt' file that defines the Caffe model.")
    parser.add_argument('caffemodel', action='store', help="The filename (full path including file extension) of the '.caffemodel' file that contains the network's parameters")
    parser.add_argument('output_dir', action='store', help="The path to the directory where to save the Keras model and the file where you want the code to be written in.")
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    caffe2keras(args.prototxt, args.caffemodel,args.output_dir, args.verbose)
