import os
from IPython.core.display import Image, display
from keras.utils import np_utils
from scipy.misc import imsave
import numpy as np


# Split data into test and validation
def data_split(x_train, y_train, proportion=0.7):
    return (x_train[:int(proportion * x_train.shape[0])], y_train[:int(proportion * x_train.shape[0])]), \
           (x_train[int(proportion * x_train.shape[0]):], y_train[int(proportion * x_train.shape[0]):])


# Normalize data
def data_prepocess(data, keys):
    for key in keys:
        if 'y' in key:
            data[key + '_gt'] = np_utils.to_categorical(data[key])
        if 'x' in key:
            data[key].astype('float')
            data[key] = data[key]/255.0


# Visualize first data with labels
def visualize_data(paths, data, data_status, number=100, notebook=False):
    for data_status_item in data_status:
        for i in range(number):
            imsave(os.path.join(paths[os.path.join('visualisation',
                                                   data_status_item,
                                                   str(data['y_' + data_status_item][i]))],
                                str(i) + '.png'),
                   data['x_' + data_status_item][i, :, :])
            if notebook:
                print(data_status_item + ' sample number ' + str(i) + ' (ground truth value = ' + str(
                    data['y_' + data_status_item][i]) + ') :')
                display(Image(filename=os.path.join(paths[os.path.join('visualisation',
                                                                       data_status_item,
                                                                       str(data['y_' + data_status_item][i]))],
                                                    str(i) + '.png')))


# Save data in .npy format
def save_data(path, dictionnary):
    for key in dictionnary:
        np.save(os.path.join(path,key),
                dictionnary[key])
