from dictionaries import Parameters
from class_velocity_neuron import LIF_simple
from class_primitive_neuron import LIF_primitive
from class_position_neuron import LIF
from class_sensory_neuron import AdEx
from class_hair_field import HairField
from plots import *
from functions import *


def convert_to_bins(old_array, n_bins):
    n_steps = old_array.shape[0]
    n_steps_bin = int(n_steps / n_bins)
    new_array = np.empty((n_bins, old_array.shape[1]), dtype=int)

    for j in range(new_array.shape[1]):
        for i in range(n_bins):
            elements = old_array[n_steps_bin*i:n_steps_bin*(i+1), j]
            elements[elements == 0] = False
            if np.any(elements):
                new_array[i, j] = 1
            else:
                new_array[i, j] = 0

    return new_array


array = np.random.choice([1, 0], size=(10, 10), p=[1./10, 9./10])
new_array = convert_to_bins(array, 2)


