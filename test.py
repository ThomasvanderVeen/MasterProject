from dictionaries import Parameters
from class_velocity_neuron import LIF_simple
from class_primitive_neuron import LIF_primitive
from class_position_neuron import LIF
from class_sensory_neuron import AdEx
from class_hair_field import HairField
from plots import *
from functions import *

N_sims = 3

w_pos = [12e-3, 0, 13e-3, 11e-3, 8e-3]
w_vel = [13.5e-3, 13e-3, 0, 12.5e-3, 13.5e-3]

permutations, synapse_type, weights_primitive, primitive_filter, primitive_filter_2 = get_encoding(w_pos, w_vel)

for i in range(60):
    print(permutations[i], synapse_type[i])

