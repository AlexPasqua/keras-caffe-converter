"""
This script goes through the Keras model and creates the prototxt of an equivalent Caffe one
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

import caffe
from caffe import layers as L

import numpy as np
import argparse


def fix_prototxt(prototxt_path):
    """
    Delete from prototxt a redundant input layer
    Arguments:
        prototxt_path: the path to the prototxt file to fix
    """
    with open(prototxt_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i] == 'layer {\n':
                begin_index = i

            if 'type: "Input"' in lines[i]:
                for j in range(begin_index, len(lines)):
                    if lines[j] == '}\n':
                        end_index = j
                        break

                for k in range(begin_index, end_index + 1):
                    lines[k] = ''

    with open(prototxt_path, 'w') as f:
        for line in lines:
            f.write(line)


def create_caffe_net_struct(keras_model_path, prototxt_path):
    """
    It goes through the Keras model and creates the prototxt of an equivalent Caffe one.

    Arguments:
        keras_model_path: the path to a file containing the Keras model
        prototxt_path: the path to the prototxt file to create
    """

    # load the keras model
    keras_model = tf.keras.models.load_model(keras_model_path)

    # create a Caffe NetSpec
    caffe_net = caffe.NetSpec()

    # read layer by layer of the keras model
    types = []
    for i in range(len(keras_model.layers)):
        layer = keras_model.layers[i]
        type = layer.__class__.__name__
        name = layer.name

        if type not in types:
            types.append(type)

        if type == 'InputLayer':
            # position 0 because it returns a list with one tuple containing the shape dimensions,
            # then I make that tuple a list (to potentially modify its items)
            shape = list(layer.output_shape[0])
            with open(prototxt_path, 'w') as prototxt:
                prototxt.write(f'input: "{name}"\n')
                shape[0] = shape[0] if shape[0] != None else 1
                prototxt.write(f'input_dim: {shape[0]}\ninput_dim: {shape[3]}\n')
                prototxt.write(f'input_dim: {shape[1]}\ninput_dim: {shape[2]}\n')
            caffe_net.tops[name] = L.Input()

        elif type in ('Conv2D', 'ReLU', 'MaxPooling2D'):
            # To get the bottom, we first access to the node which connects
            # those two layers and then takes the layer which is on its input
            bottom_name = layer._inbound_nodes[0].inbound_layers.name
            found_bottom = False
            for k in caffe_net.tops.keys():
                if k == bottom_name:
                    found_bottom = True
                    bottom = caffe_net.tops[k]
                    break

            if not found_bottom:
                #print(f"Bottom NOT found for layer {name}")
                pass

            else:
                if type == 'Conv2D':
                    config = layer.get_config()
                    filters = layer.get_weights()[0]
                    biases = layer.get_weights()[1]
                    num_output = config['filters']  # equivalent to: num_output = np.shape(biases)[0]
                    kernel_size = config['kernel_size'][0]  # equivalent to: kernel_size = np.shape(filters)[0] (assuming only spatially square kernels)
                    # TODO: calculate pad to write it in the prototxt
                    if config['padding'] == 'same':
                        pass
                    elif config['padding'] == 'valid':
                        pass
                    caffe_net.tops[name] = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size)

                elif type == 'ReLU':
                    caffe_net.tops[name] = L.ReLU(bottom)

                elif type == 'MaxPooling2D':
                    # TODO: fill this MaxPooling2D section
                    pass

    print('All types present: ', types)
    with open(prototxt_path, 'a') as prototxt:
        prototxt.write(str(caffe_net.to_proto()))

    fix_prototxt(prototxt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reads the Keras model and creates the structure of an equivalent Caffe one"
    )
    parser.add_argument('keras_model', action='store', help="The filename (full path including extension) of the file that contains the Keras model.")
    parser.add_argument('prototxt', action='store', help="The filename (full path WITHOUT extension) of the output prototxt for the Caffe model.")
    args = parser.parse_args()

    args.prototxt = args.prototxt + ('' if args.prototxt[-9 : ] == '.prototxt' else '.prototxt')
    create_caffe_net_struct(args.keras_model, args.prototxt)
