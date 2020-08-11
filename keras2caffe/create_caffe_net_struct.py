import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

import caffe
from caffe import layers

import numpy as np
import argparse


def create_caffe_net_struct(keras_model_path, prototxt_path):
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
            caffe_net.tops[name] = layers.Input()

        elif type == 'Conv2D':
            filters = layer.get_weights()[0]
            biases = layer.get_weights()[1]
            num_output = np.shape(biases)[0]
            kernel_size = np.shape(filters)[0]  # assuming only spatially square kernels
            # To get the bottom, we first access to the node which connects
            # those two layers and then takes the layer which is on its input
            bottom_name = layer._inbound_nodes[0].inbound_layers.name
            found_bottom = False
            for k in caffe_net.tops.keys():
                if k == bottom_name:
                    found_bottom = True
                    caffe_net.tops[name] = layers.Convolution(caffe_net.tops[k], num_output=num_output, kernel_size=kernel_size)

            if not found_bottom:
                print(f"Bottom NOT found for layer {name}")




    print('All types present: ', types)
    return caffe_net.to_proto()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reads the Keras model and creates the structure of an equivalent Caffe one"
    )
    parser.add_argument('keras_model', action='store', help="The filename (full path including extension) of the file that contains the Keras model.")
    parser.add_argument('prototxt', action='store', help="The filename (full path WITHOUT extension) of the output prototxt for the Caffe model.")
    args = parser.parse_args()

    args.prototxt = args.prototxt + ('' if args.prototxt[-9 : ] == '.prototxt' else '.prototxt')
    with open(args.prototxt, 'a') as f:
        f.write(str(create_caffe_net_struct(args.keras_model, args.prototxt)))

    # Delete from prototxt a redundant input layer
    with open(args.prototxt, 'r') as f:
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

    with open(args.prototxt, 'w') as f:
        for line in lines:
            f.write(line)
