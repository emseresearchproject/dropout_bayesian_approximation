from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.utils import plot_model
from IPython.core.display import Image, display
import os


def create_model(hyperparameters):
    model = Sequential()
    model.add(Reshape((784,), input_shape=(28,28,)))
    for i in range(hyperparameters['nb_inner_layers']):
        model.add(Dense(hyperparameters['nb_neurons_per_layers'],
                        activation=hyperparameters['activation_function']))
    model.add(Dense(10))
    model.compile(optimizer=hyperparameters['optimizer'],
                  metrics=hyperparameters['metrics'],
                  loss=hyperparameters['loss'])
    return model
