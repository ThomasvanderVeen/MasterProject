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
gait = np.array(data[f'simulation_1'][1])[0, :]

gait_inverted = np.logical_not(gait).astype(int)
random_spike_train = np.random.randint(0, 2, gait.size)
x = np.linspace(0, 1, num=30)

N_swing = np.sum(random_spike_train*gait)
N_stance = np.sum(random_spike_train*gait_inverted)
dt = 0.001

change_index = np.where(gait[:-1] != gait[1:])[0]

N = int(change_index.size/2)-1
swing_bin_rate, swing_bin_likelihood, stance_bin_rate, stance_bin_likelihood = np.zeros((N, 10)), np.zeros((N, 10)),\
                                                                                np.zeros((N, 20)), np.zeros((N, 20))
for i in range(N):
    spikes_swing = np.array_split(random_spike_train[change_index[i]:change_index[i+1]], 10)
    spikes_stance = np.array_split(random_spike_train[change_index[i]:change_index[i+1]], 20)

    for j in range(10):
        swing_bin_rate[i, j] = np.sum(spikes_swing[j])

    for k in range(20):
        stance_bin_rate[i, k] = np.sum(spikes_stance[k])

    stance_bin_likelihood[stance_bin_rate > 0.5] = 1
    swing_bin_likelihood[swing_bin_rate > 0.5] = 1

swing_bin_rate = np.sum(swing_bin_rate, axis=0)/N
stance_bin_rate = np.sum(stance_bin_rate, axis=0)/N
swing_bin_likelihood = np.sum(swing_bin_likelihood, axis=0)/N
stance_bin_likelihood = np.sum(stance_bin_likelihood, axis=0)/N

plt.scatter(x, np.append(swing_bin_rate, stance_bin_rate))
plt.show()

plt.scatter(x, np.append(swing_bin_likelihood, stance_bin_likelihood))
plt.show()

x = np.linspace(0, 1, num=gait.shape[0])
plt.plot(x, gait)
plt.scatter(x[change_index], 2*gait[change_index])
plt.show()