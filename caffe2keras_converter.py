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


def caffe2keras(prototxt_path, km_src_path, km_path):
    km_src_path = km_src_path + ('' if km_src_path[-3 : ] == '.py' else '.py')
    km_path = km_path + ('' if km_path[-3 : ] == '.h5' else '.h5')

    # Write beginning of the source file containing the Keras model
    with open(km_src_path, 'w') as outfile:
        write_beginning(outfile)

    # Write in km_src_path th ìe Keras code to generate the model
    create_nn_struct.write_nn_struct_code_keras(prototxt_path, km_src_path)

    # Write the end of the source file
    with open(km_src_path, 'a') as outfile:
        outfile.write("\n\n\treturn keras_model\n\n\n")
        outfile.write(f"if __name__ == '__main__':\n\tkeras_model().save('{km_path}')")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reads the Caffe network's defnition from its prototxt, the parameters from its caffemodel \
                    and generates a Python file containing the network (architecture + parameters) in Keras"
    )
    parser.add_argument('prototxt', action='store', help="The filename (full path including file extension) of the '.prototxt' file that defines the Caffe model.")
    parser.add_argument('caffemodel', action='store', help="The filename (full path including file extension) of the '.caffemodel' file that contains the network's parameters")
    parser.add_argument('keras_model_source_file', action='store', help="The filename (full path WITHOUT extension) of the Python file where you want the code to be written in.")
    parser.add_argument('keras_model', action='store', help="The filename (full path WITHOUT extension) of the .h5 file containing the Keras model")
    args = parser.parse_args()
    caffe2keras(args.prototxt, args.keras_model_source_file, args.keras_model)
