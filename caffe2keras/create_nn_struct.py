"""
Script to create the NN's structure that should be written in keras.
It reads the prototxt model definition file and write a draft of the code that should
create the net's structure in Keras
"""

import argparse


# Dictionary containing the base strings for each type of known layer
layers_strings = {
    'Input': "\tinput = Input(shape=({},{},{}), name='{}')\n",
    'Convolution': "\t{} = Conv2D(name='{}', filters={}, kernel_size={}, strides={}, padding='same')({})\n",
    'ReLU': "\t{} = ReLU()({})\n",
    'PReLU': "\t{} = PReLU(name='{}')({})\n",
    'Concat': "\t{} = Concatenate(name='{}')([{}])\n\n",
    'Pooling': "\t{} = {}Pooling2D(pool_size={}, strides={}, padding='valid')({})\n",
    'unknown': "\t\nUNKNOWN Layer --> line: {}\tname: {}\t type: {}\n\n",
}


def write_layer(layer_data, outfile):
    """
    Writes one layer's data on the output file

    Arguments:
        layer_data: Dictionary containing one layer's data
        outfile: the output file (in order to be able to write on it)

    Returns: layer_data modified -> layer_data['bottoms'] is reset
    """

    known_types = ['Convolution', 'ReLU', 'Pooling', 'PReLU', 'Concat']
    pool_type = {'MAX':'Max', 'AVG':'Average'}

    if layer_data['type'] == 'Convolution':
        outfile.write(layers_strings['Convolution'].format(layer_data['name'],
                                                           layer_data['name'],
                                                           layer_data['num_output'],
                                                           layer_data['kernel_size'],
                                                           layer_data['stride'],
                                                           write_layer.prev_layer_top))

    elif layer_data['type'] == 'Pooling':
        outfile.write(layers_strings['Pooling'].format(layer_data['name'],
                                                       pool_type[layer_data['pool']],
                                                       layer_data['kernel_size'],
                                                       layer_data['stride'],
                                                       write_layer.prev_layer_top))

    elif layer_data['type'] == 'ReLU':
        outfile.write(layers_strings['ReLU'].format(write_layer.prev_layer_name, write_layer.prev_layer_name))

    elif layer_data['type'] == 'PReLU':
        outfile.write(layers_strings['PReLU'].format(write_layer.prev_layer_name, layer_data['name'], write_layer.prev_layer_name))

    elif layer_data['type'] == 'Concat':
        bottoms_str = ''
        for bottom in layer_data['bottoms']:
            bottoms_str = bottoms_str + bottom + ', '
        outfile.write(layers_strings['Concat'].format(layer_data['name'], layer_data['name'], bottoms_str[ : -2]))

    elif layer_data['type'] not in known_types:
        outfile.write(layers_strings['unknown'].format(i, layer_data['name'], layer_data['type']))

    layer_data['bottoms'] = []
    write_layer.prev_layer_name = layer_data['name']
    write_layer.prev_layer_top = layer_data['top']
    return layer_data


def write_nn_struct_code_keras(prototxt_path, outfile_path):
    """
    Reads the Caffe model's definition from its prototxt file and writes on the outfile
    the code to generate the same model in Keras.

    Arguments: argv should contain...
        prototxt (str): The filename (full path including file extension) of the '.prototxt' file that defines the Caffe model
        outfile (str): The filename (full path including file extension) of the file where you want the code to be written in
        start_line (int): The line of [outfile] where you want the Keras code to start in

    Returns: None
    """


    # Dictionary containing one layer's data
    layer_data = {'name': '', 'type': '', 'bottoms': [], 'top': '',
        'num_output': 0, 'pool': 'MAX', 'kernel_size': 0, 'stride': 1
    }

    # These 2 variables are persistent in write_layer: even when the function ends, they keep their value
    write_layer.prev_layer_top = ''
    write_layer.prev_layer_name = ''

    curls_count = 0     # counter of open (and not closed yet) curly brackets in the prototxt
    inparam = False     # True if the current line is inside the convolution_param / pooling_param of a layer

    # Read the prototxt
    # Update layer_data
    # Basing on the layer's type, I use (only) the attributes I need of layer_data
    with open(prototxt_path, 'r') as prototxt:
        lines = prototxt.readlines()
        with open(outfile_path, 'a') as outfile:
            offset = 0
            while 'input:' not in lines[offset]:
                if 'layer {' in lines[offset]: break    # in case there's no input
                offset = offset + 1

            # Write the input layer
            if 'input:' in lines[offset]:
                input_name = lines[offset].split()[1].strip('"')
                map = {3:0, 4:1, 2:2}
                shape = [0,0,0]
                for i in map.keys():
                    v = [int(s) for s in lines[i + offset].split() if s.isdigit()]
                    shape[map[i]] = v[1] if len(v) > 1 else v[0]
                write_layer.prev_layer_top = 'input'
                outfile.write(layers_strings['Input'].format(shape[0], shape[1], shape[2], input_name))

            # Scan the rest of the prototxt
            for i in range(5 + offset, len(lines)):
                curr = lines[i]
                if '{' in curr:     # It means that we're entering a section
                    curls_count = curls_count + 1
                elif '}' in curr:
                    curls_count = curls_count - 1
                    if curls_count == 1 and inparam: inparam = False
                    elif curls_count == 0: layer_data = write_layer(layer_data, outfile)
                    continue

                # If curr is a line which is not part of a subsection of a layer
                if curls_count == 1:
                    keywords = ('name:', 'type:', 'top:', 'bottom:')
                    for kw in keywords:
                        if kw in curr:
                            datum = curr.split()[1].strip('"')
                            if kw == 'bottom:': layer_data['bottoms'].append(datum)
                            else: layer_data[kw[:-1]] = datum

                # Otherwise it has to read the convolution_param / pooling_param
                elif curls_count == 2:
                    if 'convolution_param {' in curr or 'pooling_param {' in curr:
                        inparam = True
                        default_stride = True
                        continue

                    if inparam:
                        keywords = ('num_output:', 'pool:', 'kernel_size:', 'stride:')
                        for kw in keywords:
                            if kw in curr:
                                layer_data[kw[:-1]] = curr.split()[1]
                                if kw == 'stride:': default_stride = False

                        if default_stride: layer_data['stride'] = 1

            # After the for cycle finished...
            outfile.write(f"\tkeras_model = tf.keras.Model(inputs=input, outputs={layer_data['name']})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reads the Caffe network's defnition from its prototxt and generates the Python code \
                    to create the network's structure in Keras"
    )
    parser.add_argument('prototxt', action='store', help="The filename (full path including file extension) of the '.prototxt' file that defines the Caffe model.")
    parser.add_argument('outfile', action='store', help="The filename (full path including file extension) of the file where you want the code to be written in.")
    args = parser.parse_args()
    write_nn_struct_code_keras(args.prototxt, args.outfile)
