from dictionaries import Parameters
from class_sensory_neuron import AdEx
from class_position_neuron import LIF
from tqdm import tqdm
import torch
import numpy as np
from functions import *
import matplotlib.pyplot as plt
import os

if not os.path.exists("Images_testing"):
    os.makedirs("Images_testing")

'''
Testing for the AdEx neuron, table 1 (thesis) is correct except for V_r =! -58 mV, should be V_r = -70mV
'''

'''
parameters_sensory = {'C': 200e-12, 'g_L': 12e-9, 'E_L': -70e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'V_T': -50e-3,
                      'tau_W': 300e-3, 'b': 60e-12, 'V_R': -70e-3, 'V_cut': -40e-3, 'refrac': 0.00,
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
plt.xlim(0, 5)
plt.ylim(0, 500)
plt.ylabel('Spike Rate [imp/s]')
plt.savefig('Images_testing/AdEx_testing.png')
'''

'''
Testing for the position neuron
'''

parameters_position = {'E_L': -70e-3, 'V_T': -50e-3, 'tau': 50e-3, 'tau_W': 10e-3, 'tau_epsp': 50e-3, 'b': 10e-3,
                 'V_R': -70e-3, 'n': 3, 'N_input': 1, 'dt': 0.001, 'refrac': 0}

t_total = 1
N_steps = int(t_total/parameters_position['dt'])
N_interval = int(0.2/parameters_position['dt'])

sensory_neuron = LIF(parameters_position)
sensory_neuron.initialize_state()

voltage = torch.empty((N_steps, parameters_position['n']))
input = torch.zeros(N_steps)
input[int(N_interval/2)::N_interval] = 5
input = torch.tile(torch, (3, 1))

for i in tqdm(range(N_steps)):
    voltage, _ = sensory_neuron[input[i]]


