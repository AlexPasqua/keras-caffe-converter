"""
Script to create the NN's structure that should be written in keras.
It reads the prototxt model definition file and write a draft of the code that should
create the net's structure in Keras
"""

import argparse


def write_nn_struct_code_keras(argv):
    """
    Reads the Caffe model's definition from its prototxt file and writes on the outfile
    the code to generate the same model in Keras.

    Arguments: argv should contain...
        prototxt (str): The filename (full path including file extension) of the '.prototxt' file that defines the Caffe model
        outfile (str): The filename (full path including file extension) of the file where you want the code to be written in
        start_line (int): The line of [outfile] where you want the Keras code to start in

    Returns: None
    """

    known_types = ['Convolution', 'ReLU', 'Pooling', 'PReLU', 'Concat']
    # Dictionary containing the base strings for each type of known layer
    layers_strings = {
        'Input': "input = Input(shape=({},{},{}), name='{}')\n",
        'Convolution': "{} = Conv2D(name='{}', filters={}, kernel_size={}, strides={}, padding='same')({})\n",
        'ReLU': "{} = ReLU()({})\n",
        'PReLU': "{} = PReLU(name='{}')({})\n",
        'Concat': "{} = Concatenate(name='{}')([{}])\n\n",
        'Pooling': "{} = {}Pooling2D(pool_size={}, strides={}, padding='valid')({})\n",
        'unknown': "\nUNKNOWN Layer --> line: {}\tname: {}\t type: {}\n\n",
    }
    # Dictionary containing one layer's data
    layer_data = {'name': '', 'type': '', 'bottoms': [], 'top': '',
        'num_output': 0, 'pool_type': 'MAX', 'kernel_size': 0, 'stride': 1,
    }
    pool_type = {'MAX':'Max', 'AVG':'Average'}
    prev_layer_top = ''
    prev_layer_name = ''

    # Read the prototxt
    # Update layer_data
    # Basing on the layer's type, I use (only) the attributes I need of layer_data
    with open(argv.prototxt, 'r') as prototxt:
        lines = prototxt.readlines()
        with open(argv.outfile, 'w') as outfile:
            offset = 0
            while 'input:' not in lines[offset]:
                offset = offset + 1

            # Write the input layer
            if 'input:' in lines[offset]:
                input_name = lines[offset][8 : -2]
                shape = (lines[3 + offset][11], lines[4 + offset][11], lines[2 + offset][11])
                prev_layer_top = 'input'
                outfile.write(layers_strings['Input'].format(shape[0], shape[1], shape[2], input_name))

            # Scan the rest of the prototxt
            for i in range(5 + offset, len(lines)):
                curr = lines[i]
                if 'name:' in curr:
                    layer_data['name'] = curr[9 : -2]
                    continue

                elif 'type:' in curr:
                    layer_data['type'] = curr[9 : -2]
                    if layer_data['type'] not in known_types:
                        outfile.write(layers_strings['unknown'].format(i, layer_data['name'], layer_data['type']))
                    continue

                elif 'bottom:' in curr:
                    if layer_data['type'] == 'Concat':
                        layer_data['bottoms'].append(curr[11 : -2])
                    continue

                elif 'top:' in curr:
                    layer_data['top'] = curr[8 : -2]
                    continue

                elif 'convolution_param {' in curr or 'pooling_param {' in curr:
                    #print(layer_data['type'], '\t\t', layer_data['name'])
                    layer_data['stride'] = 1    # stride default value
                    # read all the convolution / pooling parameters
                    j = 1
                    while '}' not in lines[i+j]:
                        cursor = lines[i+j]
                        if 'num_output:' in cursor:
                            layer_data['num_output'] = cursor[-4 : -1].strip()
                        elif 'pool:' in cursor:
                            layer_data['pool_type'] = cursor[-4 : -1]
                        elif 'kernel_size:' in cursor:
                            layer_data['kernel_size'] = cursor[-2]
                        elif 'stride:' in cursor:
                            layer_data['stride'] = cursor[-2]
                        j = j + 1

                    if layer_data['type'] == 'Convolution':
                        if prev_layer_top == '': prev_layer_top = 'input'   # I don't know why the first time it's empty
                        outfile.write(layers_strings['Convolution'].format(layer_data['name'],
                                                                           layer_data['name'],
                                                                           layer_data['num_output'],
                                                                           layer_data['kernel_size'],
                                                                           layer_data['stride'],
                                                                           prev_layer_top))
                    elif layer_data['type'] == 'Pooling':
                        outfile.write(layers_strings['Pooling'].format(layer_data['name'],
                                                                       pool_type[layer_data['pool_type']],
                                                                       layer_data['kernel_size'],
                                                                       layer_data['stride'],
                                                                       prev_layer_top))

                if layer_data['type'] == 'ReLU':
                    outfile.write(layers_strings['ReLU'].format(prev_layer_name, prev_layer_name))
                elif layer_data['type'] == 'PReLU':
                    outfile.write(layers_strings['PReLU'].format(prev_layer_name, layer_data['name'], prev_layer_name))
                elif layer_data['type'] == 'Concat':
                    bottoms_str = ''
                    for bottom in layer_data['bottoms']:
                        bottoms_str = bottoms_str + bottom + ', '
                    outfile.write(layers_strings['Concat'].format(layer_data['name'], layer_data['name'], bottoms_str[ : -2]))
                    layer_data['bottoms'] = []

                layer_data['type'] = ''
                prev_layer_name = layer_data['name']
                prev_layer_top = layer_data['top']

            outfile.write(f"keras_model = tf.keras.Model(inputs=input, outputs={layer_data['name']})")


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
