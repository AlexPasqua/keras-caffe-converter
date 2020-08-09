import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, ReLU, MaxPooling2D, PReLU, Concatenate

import caffe
from caffe import layers as cl, params as cp

import numpy as np
import pickle
import argparse


def create_caffe_net(prototxt_path, weights_path, caffemodel_path):
    # Load weights in dictionary form
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)

    # Caffe net
    caffe.set_device(0)
    caffe.set_mode_gpu()
    caffe_model = caffe.Net(prototxt_path, caffe.TEST)

    # Load weights in the Caffe model
    for k in weights:
        name = k[0]
        type = k[1]

        # Skip if there's no layer in the caffe model with the same name
        if not np.any([x == name for x in caffe_model._layer_names]): continue

        # Convert conv layers
        if type == 'Conv2D':
            data = weights[k]
            w,b = data if len(data) > 1 else [data[0], np.zeros(np.shape(data)[-1])]  # the bias might not be present
            caffe_model.params[name][0].data[...] = np.transpose(w, (3,2,0,1))  # Caffe wants (c_out, c_in, h, w)
            caffe_model.params[name][1].data[...] = b

        # Convert PReLU layers
        elif type == 'PReLU':
            caffe_model.params[name][0].data[...] = weights[k][0][...][0][0]

        # Convert batchnorm layers
        elif type == 'BatchNormalization':
            gamma, beta, mean, variance = weights[k]
            caffe_model.params[name][0].data[...] = mean
            caffe_model.params[name][1].data[...] = variance + 1e-3
            caffe_model.params[name][2].data[...] = 1  # always set scale factor to 1
            caffe_model.params['{}_sc'.format(name)][0].data[...] = gamma  # scale
            caffe_model.params['{}_sc'.format(name)][1].data[...] = beta  # bias

    # Save the net into caffemodel
    caffemodel_path = caffemodel_path + ('' if caffemodel_path[-11 : ] == '.caffemodel' else '.caffemodel')
    caffe_model.save(caffemodel_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reads the Keras model's parameters from a previously exported pickle, load them into a Caffe net and saves it"
    )
    parser.add_argument('prototxt', action='store', help="The filename (full path including file extension) of the '.prototxt' file that defines the Caffe model.")
    parser.add_argument('weights_path', action='store', help="The filename (full path WITHOUT extension) of the file that will contain the Keras model's weights.")
    parser.add_argument('caffemodel', action='store', help="The filename (full path WITHOUT extension) of the file where to save the Caffe model")
    args = parser.parse_args()
    create_caffe_net(args.prototxt, args.weights_path, args.caffemodel)
