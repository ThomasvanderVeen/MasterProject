from dictionaries import Parameters
from class_velocity_neuron import LIF_simple
from class_primitive_neuron import LIF_primitive
from class_position_neuron import LIF
from class_sensory_neuron import AdEx
from class_hair_field import HairField
from plots import *
from functions import *
import scipy.io
import numpy as np

data = pickle_open('simulation_data')

sim_list = []
for i in range(400):

    gait = np.array(data[f'simulation_{i}'][1])[0, :]

    change_index = np.where(gait[:-1] != gait[1:])[0]

    if change_index.size < 18:
        sim_list.append(i)

print(sim_list)

for j in sim_list:
    gait = np.array(data[f'simulation_{j}'][1])[0, :]
    init = gait[0]
    k = 0
    if gait[0] == 0: k = 1

    change_index = np.where(gait[:-1] != gait[1:])[0]
    x = np.linspace(0, 1, num=gait.shape[0])
    plt.plot(x, gait)
    plt.scatter([x[change_index[k]], x[change_index[k+1]]], [1, 1])
    print(j)
    plt.show()

