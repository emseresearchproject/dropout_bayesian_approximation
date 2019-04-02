#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.models import model_from_json
from utils import *


# Compute the epistemic and the aleatoric matrices for a given input
def uncertainties(model, x, N):
    network_output = np.zeros((1, 5, N))
    aleatoric = np.zeros((5, 5))
    epistemic = np.zeros((5, 5))
    for i in range(N):
        network_output[:, :, i] = model.predict(x)
    network_output_mean = np.mean(network_output, axis=2)
    for i in range(N):
        network_output_centered = network_output[:, :, i] - network_output_mean
        aleatoric += network_output[:, :, i] * np.eye(5) \
                     - np.transpose(network_output[:, :, i]).dot(network_output[:, :, i])
        epistemic += np.transpose(network_output_centered).dot(network_output_centered)
    aleatoric /= 100
    epistemic /= 100
    return network_output_mean, epistemic, aleatoric


# Loading model
json_file = open(os.path.join('model/model.json'), 'r')
loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
json_file.close()
model.load_weights(os.path.join('model/weights.hdf5'))

# Load data
X = np.load('data/x_test.npy')[:100, :, :]
Y = np.load('data/y_test.npy')[:100]
output = [{'aleatoric': [], 'epistemic': [], 'values': []} for i in range(10)]

# Exploiting data
for input in range(Y.shape[0]):
    x = np.expand_dims(X[input, :, :], axis=0)
    y = Y[input]
    network_output, epistemic, aleatoric = uncertainties(model, x, 100)
    output[y]['values'].append(network_output[0, :])
    output[y]['aleatoric'].append(np.trace(aleatoric)/5)
    output[y]['epistemic'].append(np.trace(epistemic)/5)

# Displaying epistemic and aleatoric values per class
for i, dictionnary in enumerate(output):
    print('Class ' + str(i) + ' :')
    out = np.mean(np.array(dictionnary['values']), axis=0)
    out_str = '\tValues : '
    for j in range(5):
        out_str += str(j) + ' : ' + str(int(1000 * out[j])/1000.0) + ' ,'
    print(out_str)
    print('\tEpistemic : ' + str(np.mean(np.array(dictionnary['epistemic']))))
    print('\tAleatoric : ' + str(np.mean(np.array(dictionnary['aleatoric']))))

# Computing epistemic and aleatoric mean values
learnt_classes = {'epistemic': 0, 'aleatoric': 0}
unlearnt_classes = {'epistemic': 0, 'aleatoric': 0}
for i in range(10):
    if i < 5:
        learnt_classes['epistemic'] += np.mean(np.array(output[i]['aleatoric'])) / 5.0
        learnt_classes['aleatoric'] += np.mean(np.array(output[i]['epistemic'])) / 5.0
    else:
        unlearnt_classes['epistemic'] += np.mean(np.array(output[i]['aleatoric'])) / 5.0
        unlearnt_classes['aleatoric'] += np.mean(np.array(output[i]['epistemic'])) / 5.0

# Displaying epistemic and aleatoric mean values
print('Learnt classes :')
print('\tEpistemic mean : ' + str(learnt_classes['epistemic']))
print('\tleatoric mean : ' + str(learnt_classes['aleatoric']))
print('Unlearnt classes :')
print('\tEpistemic mean : ' + str(unlearnt_classes['epistemic']))
print('\tAleatoric mean : ' + str(unlearnt_classes['aleatoric']))
