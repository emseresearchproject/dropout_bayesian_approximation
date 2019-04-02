from keras.engine import Layer
from keras.models import Sequential
from keras.layers import Dense, Reshape, Lambda
import keras.backend as K


def create_model(hyperparameters):
    model = Sequential()
    model.add(Reshape((784,), input_shape=(28,28,)))
    model.add(Lambda(lambda x: K.dropout(x, level=hyperparameters['dropout_rate'][hyperparameters['nb_inner_layers']])))
    for i in range(hyperparameters['nb_inner_layers']):
        model.add(Dense(hyperparameters['nb_neurons_per_layers'],
                        activation='relu'))
        model.add(Lambda(lambda x: K.dropout(x, level=hyperparameters['dropout_rate'][i])))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer=hyperparameters['optimizer'],
                  metrics=hyperparameters['metrics'],
                  loss=hyperparameters['loss'])
    return model
