"""
Script to create the NN's structure that should be written in keras.
It reads the prototxt model definition file and write a draft of the code that should
create the net's structure in Keras
"""


import argparse


def write_nn_struct_code_keras(argv):
    known_types = ['Convolution', 'ReLU', 'Pooling', 'PReLU', 'Concat']
    # Dictionary containing the base strings for each type of known layer
    layers_strings = {
        'Convolution': "{} = Conv2D(name='{}', filters={}, kernel_size={}, strides={}, padding='same')({})\n",
        'ReLU': "{} = ReLU()({})\n",
        'PReLU': "{} = PReLU(name='{}')({})\n",
        'Concat': "{} = Concatenate(name='{}')([{}])\n\n",
        'Pooling': "{} = {}Pooling2D(pool_size={}, strides={}, padding='valid')({})\n",
        'unknown': "\nUNKNOWN Layer --> line: {}\tname: {}\t type: {}\n\n",
    }
    pool_type = {'MAX':'Max', 'AVG':'Average'}
    # Dictionary containing one layer's data
    layer = {
        'name': '', 'type': '', 'bottoms': [], 'top': '',
        'conv_params': {
            'num_output': 0,
            'kernel_size': 0,
            'stride': 1
        },
        'pool_params': {
            'pool_type': 'MAX',
            'kernel_size': 2,
            'stride': 2
        }
    }

    # Read the prototxt
    with open(argv.prototxt, 'r') as prototxt:
        lines = prototxt.readlines()
        with open(argv.outfile, 'w') as outfile:
            for i in range(len(lines)):
                curr = lines[i]
                if 'name:' in curr:
                    layer['name'] = curr[9 : -2]
                    continue

                elif 'type:' in curr:
                    layer['type'] = curr[9 : -2]
                    if layer['type'] not in known_types:
                        outfile.write(layers_strings['unknown'].format(i, layer['name'], layer['type']))
                    continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reads the Caffe network's defnition from its prototxt and generates the Python code \
                    to create the network's structure in Keras"
    )
    parser.add_argument('prototxt', action='store', help="The filename (full path including file extension) of the '.prototxt' file that defines the Caffe model.")
    parser.add_argument('outfile', action='store', help="The filename (full path including file extension) of the file where you want the code to be written in.")
    parser.add_argument('start_line', type=int, action='store', default=0, help="The line of [outfile] where you want the Keras code to start in.")
    args = parser.parse_args()
    write_nn_struct_code_keras(args)
