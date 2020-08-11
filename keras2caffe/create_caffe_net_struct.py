import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

import caffe
from caffe import layers

import numpy as np
import argparse


def create_caffe_net_struct(keras_model_path):
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
            shape = layer.output_shape
            # TODO: scrivi manualmente l'inizio del prototxt, salvalo e importalo come fosse una rete, poi modificala

        elif type == 'Conv2D':
            filters = layer.get_weights()[0]
            biases = layer.get_weights()[1]
            num_output = np.shape(biases)[0]
            kernel_size = np.shape(filters)[0]  # assuming only spatially square kernels
            # To get the bottom, we first access to the node which connects
            # those two layers and then takes the layer which is on its input
            bottom = layer._inbound_nodes[0].inbound_layers
            caffe_net.tops[name] = layers.Convolution(bottom, num_output=20, kernel_size=5)


    print('All types present: ', types)
    return caffe_net.to_proto()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reads the Keras model and creates the structure of an equivalent Caffe one"
    )
    parser.add_argument('keras_model', action='store', help="The filename (full path including extension) of the file that contains the Keras model.")
    args = parser.parse_args()

    with open('prova.prototxt', 'w') as f:
        f.write(str(create_caffe_net_struct(args.keras_model)))
