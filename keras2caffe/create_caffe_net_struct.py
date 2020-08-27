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

        elif type in ('Conv2D', 'ReLU', 'MaxPooling2D', 'PReLU'):
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
                # For padding, kernel_size and pool_size, I only take the 1st number of the tuple I get with layer.get_config()
                # because Caffe only accepts spatially square kernels
                if type == 'Conv2D':
                    config = layer.get_config()
                    filters = layer.get_weights()[0]
                    biases = layer.get_weights()[1]
                    num_output = config['filters']  # equivalent to: num_output = np.shape(biases)[0]
                    kernel_size = config['kernel_size'][0]  # equivalent to: kernel_size = np.shape(filters)[0]
                    if config['padding'] == 'same':     # maintain the same spatial size
                        stride = config['strides'][0]
                        layer_input_size = np.shape(layer._inbound_nodes[0].inbound_layers.output)[1]
                        pad = (stride * (np.shape(layer.output)[1] - 1) - layer_input_size + kernel_size) // 2
                    elif config['padding'] == 'valid': pad = 0
                    caffe_net.tops[name] = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, pad=pad)

                elif type == 'ReLU':
                    caffe_net.tops[name] = L.ReLU(bottom)

                elif type == 'MaxPooling2D':
                    config = layer.get_config()
                    pool_size = config['pool_size'][0]
                    stride = config['strides'][0]
                    caffe_net.tops[name] = L.Pooling(bottom, pool=0, stride=stride, kernel_size=pool_size)

                elif type == 'PReLU':
                    caffe_net.tops[name] = L.PReLU(bottom)

        elif type == 'Concatenate':
            # To get the bottom, we first access to the node which connects
            # the two layers and then we take each inbound layer (which is one of the many bottoms)
            bottoms_list = []
            for i in range(np.shape(layer._inbound_nodes[0].inbound_layers)[0]):
                current = layer._inbound_nodes[0].inbound_layers[i]
                # In case a layer is followed by an activation layer, even if the top
                # does not take the name of the activation layer, the inbound layer will be that.
                # So in this case we take the "bottom of the bottom", because on the prototxt the concat layer
                # wants the name of the "top" field of the layers to connect.
                # e.g.
                # layer {name: "conv1", top: "conv1" ...} layer {name: "relu1", type: "ReLU" top: "conv1" ...}
                # layer {name: concat, type: "Concat", bottom: "conv1" ...}
                if current.__class__.__name__ in ('ReLU', 'PReLU'):
                    bottom_name = current._inbound_nodes[0].inbound_layers.name
                else:
                    bottom_name = current.name

                for k in caffe_net.tops.keys():
                    if k == bottom_name:
                        bottom = caffe_net.tops[k]
                bottoms_list.append(bottom)

            # unfortunately the following works only with concatenation of 2 or 3 layers
            if len(bottoms_list) == 2:
                caffe_net.tops[name] = L.Concat(bottoms_list[0], bottoms_list[1])
            elif len(bottoms_list) == 3:
                caffe_net.tops[name] = L.Concat(bottoms_list[0], bottoms_list[1], bottoms_list[2])
            else:
                print("\n\nE: found concat layer with more than 3 bottoms. This programm cannot handle it\t", len(bottoms_list))

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
