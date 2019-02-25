import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Model


##keract function
def _evaluate(model: Model, nodes_to_evaluate, x, y=None):
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, nodes_to_evaluate)
    x_, y_, sample_weight_ = model._standardize_user_data(x, y)
    return f(x_ + y_ + sample_weight_)

##keract function
def \
        get_activations(model, x, layer_name=None):
    nodes = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None]
    # we process the placeholders later (Inputs node in Keras). Because there's a bug in Tensorflow.
    input_layer_outputs, layer_outputs = [], []
    [input_layer_outputs.append(node) if 'input_' in node.name else layer_outputs.append(node) for node in nodes]
    activations = _evaluate(model, layer_outputs, x, y=None)
    activations_dict = dict(zip([output.name for output in layer_outputs], activations))
    activations_inputs_dict = dict(zip([output.name for output in input_layer_outputs], x))
    result = activations_inputs_dict.copy()
    result.update(activations_dict)
    return result

def _getRowsCols(size):
    sqrt =math.sqrt(size)
    ceil = math.ceil(sqrt)
    floor = math.floor(sqrt)

    if size > ceil * floor:
        return (ceil,ceil)
    else:
        return (ceil,floor)




def show_activations(activations, timeout=30, winnname='show'):
    for layer_name, first in activations.items():
        #print(layer_name, first.shape, end=' ')
        if first.shape[0] != 1:
            #print('-> Skipped. First dimension is not 1.')
            continue
        #print('')
        if len(first.shape) <= 2:
            fig = plt.figure()
            fig.add_subplot()
            plt.scatter(range(len(first[0])), first[0], s=math.ceil(64 / (math.sqrt(len(first[0])))))
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            cv2.imshow(winnname, data)
            cv2.waitKey(timeout)
            plt.close(fig)
            continue
        size = first.shape[1]
        cols, rows = _getRowsCols(size)
        wSpace = 30
        hSpace = 30
        width = cols * first.shape[3] + (cols)*wSpace
        height = rows * first.shape[2] + (rows)*hSpace
        mat = np.zeros((height, width), dtype=np.float32)

        for i in range(rows):
           for j in range(cols):
               if i*cols + j >= size:
                   break
               tempImg = first[0, i*cols+j,:,:]
               mat[i*(first.shape[2]+hSpace):i*(first.shape[2]+hSpace)+tempImg.shape[0], j*(first.shape[3]+wSpace):j*(first.shape[3]+wSpace)+ tempImg.shape[1]] = tempImg

        mat = cv2.resize(mat, (0, 0), fx=3, fy=3)
        cv2.imshow(winnname, mat)
        cv2.waitKey(timeout)



##keract function
def display_activations(activations):
    max_columns=6
    max_rows=6
    for layer_name, first in activations.items():
        print(layer_name, first.shape, end=' ')
        if first.shape[0] != 1:
            print('-> Skipped. First dimension is not 1.')
            continue
        print('')
        if len(first.shape) <= 2:
            fig = plt.figure()
            fig.add_subplot()
            plt.scatter(range(len(first[0])), first[0], s=math.ceil(64 / (math.sqrt(len(first[0])))))
            plt.show()
            continue

        fig = plt.figure(figsize=(12,12))
        plt.axis('off')
        plt.title(layer_name)



        for i in range(1, min(max_columns * max_rows + 1, first.shape[1] + 1)):
            img = first[0, i-1, :, :]
            fig.add_subplot(max_rows, max_columns, i)
            plt.imshow(img)
            plt.axis('off')
        plt.show()