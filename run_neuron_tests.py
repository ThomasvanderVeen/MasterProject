from dictionaries import Parameters
from class_sensory_neuron import AdEx
from tqdm import tqdm
import torch
import numpy as np
from functions import *
import matplotlib.pyplot as plt

parameters_sensory = {'C': 200e-12, 'g_L': 12e-9, 'E_L': -70e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'V_T': -50e-3,
                      'tau_W': 300e-3, 'b': 60e-12, 'V_R': -58e-3, 'V_cut': 20e-3, 'refrac': 0.00,
                      'n': 100, 'dt': 0.0001}

t_total = 5
N_steps = int(t_total/parameters_sensory['dt'])

for k in [100e-3, 200e-3, 300e-3, 400e-3, 500e-3, 600e-3]:
    parameters_sensory['tau_W'] = k
    sensory_neuron = AdEx(parameters_sensory)
    sensory_neuron.initialize_state()

    spike_list = torch.empty((N_steps, parameters_sensory['n']))
    input = torch.linspace(0, 5e-9, steps=parameters_sensory['n'])
    input = torch.tile(input, (N_steps, 1))
    firing_rate_list = []

    for i in tqdm(range(N_steps)):
        voltage, spike_list[i, :] = sensory_neuron.forward(input[i, :])

    for j in range(parameters_sensory['n']):
        firing_rate, spike_index = get_firing_rate(spike_list[:, j], parameters_sensory['dt'])
        firing_rate_list.append(firing_rate[-1])

    plt.plot(input[0, :]*1e9, firing_rate_list, label=f'{int(1000*k)} ms')
plt.legend()
plt.xlabel('Input[nA]')
plt.ylabel('Spike Rate [imp/s]')
plt.show()