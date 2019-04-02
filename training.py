#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import mnist

from model import *
from preprocess import *
from utils import *


# Defining hyperparameters
hyperparameters = {'nb_inner_layers': 4,
                   'nb_neurons_per_layers': 25,
                   'activation_function': 'softmax',
                   'batch_size': 50,
                   'nb_epochs': 250,
                   'loss': 'mean_squared_error',
                   'optimizer': 'sgd',
                   'metrics': ['categorical_accuracy'],
                   'dropout_rate': [0.1, 0.1, 0.1, 0.1, 0.1],
                   'experiment': 'MC Dropout'
                   }
K.clear_session()

# Defining global variables
data_status = ['train', 'validation', 'test']
categories = np.arange(10).tolist()

# Defining output directory structure
path_structure = {'preprocess': [],
                  'model': [['stats']],
                  'visualisation': [data_status, categories]
                  }
paths = create_paths_from_dictionnary(path_structure, output_directory=get_output_dir(hyperparameters))

# Update dashboard
if os.path.exists('../output/dashboard.csv'):
    dashboard = pd.read_csv('../output/dashboard.csv')
    dashboard = dashboard.append(hyperparameters, ignore_index=True)
else:
    print('a')
    dashboard = pd.DataFrame(hyperparameters)
dashboard.to_csv('../output/dashboard.csv', index=False)

# Loading splitting and preprocessing data
data = dict()
(data['x_train'], data['y_train']), (data['x_test'], data['y_test']) = mnist.load_data()
(data['x_train'], data['y_train']), (data['x_validation'], data['y_validation']) = data_split(data['x_train'],
                                                                                              data['y_train'])
indexes_train = np.where(data['y_train'] < 5)
indexes_validation = np.where(data['y_validation'] < 5)

data['x_train'] = data['x_train'][indexes_train]
data['y_train'] = data['y_train'][indexes_train]
data['x_validation'] = data['x_validation'][indexes_validation]
data['y_validation'] = data['y_validation'][indexes_validation]

data_prepocess(data, list(data))
save_data(paths['preprocess'], data)

# Vizualizing the first 100 images of training, validation and test set
visualize_data(paths, data, data_status, notebook=True)

# Defining callbacks
callbacks = [TensorBoard(log_dir=paths['model'], histogram_freq=1, batch_size=hyperparameters['batch_size']),
             ModelCheckpoint(os.path.join(paths['model'], 'weights.hdf5'))]

# Creating model
model = create_model(hyperparameters)

# Saving model
with open(os.path.join(paths['model'], 'model.json'), 'w') as json_file:
    json_file.write(model.to_json())

# Training model
model.fit(data['x_train'], data['y_train_gt'],
          batch_size=hyperparameters['batch_size'],
          epochs=hyperparameters['nb_epochs'],
          verbose=1,
          callbacks=callbacks,
          validation_data=(data['x_validation'], data['y_validation_gt']))
