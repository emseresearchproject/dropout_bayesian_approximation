import os
from IPython.core.display import Image, display
from scipy.misc import imsave
import matplotlib.pyplot as plt


# Create a dictionnary of paths from a dictionnary that contains the structure
# of the tree
def create_paths_from_dictionnary(paths_structure, output_directory='../output'):
    paths = dict()
    paths_list = list(paths_structure)
    create_paths(paths, paths_list, output_directory)
    for key in paths_list:
        add_categories(paths, key, paths_structure[key])
    return paths


# Create the structure in the output directory
def create_paths(paths, paths_list, output_directory='../output'):
    paths['output_dir'] = output_directory
    for path in paths_list:
        paths[path] = os.path.join(output_directory, path)
    update_repositories(paths)


# Create the directories contained in the dictionnary if they do not exist
def update_repositories(paths):
    for key in paths:
        if not os.path.isdir(paths[key]):
            os.makedirs(paths[key])


# Recursive function called by create_paths_from_dictionnary
def add_categories(paths, path, categories):
    if not categories:
        update_repositories(paths)
    else:
        for category in categories[0]:
            paths[path + '/' + str(category)] = os.path.join(paths[path], str(category))
            add_categories(paths, path + '/' + str(category), categories[1:])


# Obtain the experiment number
def get_output_dir(hyperparameters):
    if os.path.isdir('../output/' + hyperparameters['experiment']):
        l = [int(f) for f in os.listdir('../output/' + hyperparameters['experiment'])]
        if not l:
            hyperparameters['number'] = 1
            return '../output/' + hyperparameters['experiment'] + '/1/'
        hyperparameters['number'] = max(l) + 1
        return os.path.join('../output/', hyperparameters['experiment'], str(max(l) + 1))
    else:
        os.makedirs('../output/' + hyperparameters['experiment'] + '/1/')
        hyperparameters['number'] = 1
        return os.makedirs('../output/' + hyperparameters['experiment'] + '/1/')
