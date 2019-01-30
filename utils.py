import os
from IPython.core.display import Image, display
from scipy.misc import imsave
import matplotlib.pyplot as plt


def create_paths_from_dictionnary(paths_structure, output_directory='../output'):
    paths = dict()
    paths_list = list(paths_structure)
    create_paths(paths, paths_list, output_directory)
    for key in paths_list:
        add_categories(paths, key, paths_structure[key])
    return paths


def create_paths(paths, paths_list, output_directory='../output'):
    paths['output_dir'] = output_directory
    for path in paths_list:
        paths[path] = os.path.join(output_directory, path)
    update_repositories(paths)


def update_repositories(paths):
    for key in paths:
        if not os.path.isdir(paths[key]):
            os.makedirs(paths[key])


def add_categories(paths, path, categories):
    if not categories:
        update_repositories(paths)
    else:
        for category in categories[0]:
            paths[path + '/' + str(category)] = os.path.join(paths[path], str(category))
            add_categories(paths, path + '/' + str(category), categories[1:])