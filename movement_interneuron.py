from LIF_simple_class import LIF_simple
import numpy as np
import torch
import matplotlib.pyplot as plt
from HairField_class import HairField
from ramp_generator import RampGenerator
from AdEx_class import AdEx

dt = 0.001
t_total = 25
N_steps = round(t_total/dt)
N = 5

ramp = np.linspace(90/(dt*10), 90/(dt*190), num=N).astype(int)

parameters_hair_field = {'N_hairs': 7, 'min_joint_angle': 0, 'max_joint_angle': 90, 'max_angle': 90, 'overlap': 4,
                         'overlap_bi': 18}

parameters_AdEx = {'C': 200e-12, 'g_L': 12e-9, 'E_L': -70e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'V_T': 50e-3,
              'tau_W': 600e-3, 'b': 8e-12, 'V_R': -58e-3, 'V_cut': 50e-3, 'refrac': 0.00, 'n': 7,
              'dt': dt}

parameters_LIF_simple = {'tau': 5e-3, 'tau_G': 500e-3, 'G_r': 250e-3, 'p': 0.1, 'V_T': -30e-3,
                         'V_R': -70e-3, 'n': 1, 'N_input': 7, 'dt': dt, 'refrac': 0}

spikes_rates = []
hair_field = HairField(parameters_hair_field)
hair_field.get_receptive_field()

for j in range(N):
    t1 = torch.linspace(0, 90, steps=ramp[j])
    t2 = torch.linspace(90, 90, steps=N_steps-ramp[j])
    ramp_angles = torch.cat((t1, t2))
    hair_angles = hair_field.get_hair_angle(torch.Tensor.numpy(ramp_angles))/37e9

    neuron = AdEx(parameters_AdEx)
    neuron.initialize_state()

    lif = LIF_simple(parameters_LIF_simple)
    lif.initialize_state()

    voltage, time, spike_list = np.empty(hair_angles.shape), np.array([]), torch.empty(hair_angles.shape)
    voltage_inter, spike_inter = np.empty([N_steps]), np.empty([N_steps])

    for i in range(N_steps):
        voltage[i, :], spike_list[i, :] = neuron.forward(hair_angles[i, :])
        voltage_inter[i], spike_inter[i] = lif.forward(spike_list[i, :])
        time = np.append(time, i * parameters_AdEx['dt'])
        print(time[-1])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(time, spike_inter)
    ax2.plot(time, ramp_angles)
    plt.show()
    spikes = np.where(spike_inter == 1)[0]
    ISI = np.diff(spikes)*dt
    firing_rate = 1/ISI
    firing_rate = np.append(firing_rate, firing_rate[-1])
    spikes_rates.append(np.max(firing_rate))

print(spikes_rates)

'''
dt = 0.0001
t_total = 1
N_steps = round(t_total/dt)

for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    parameters_LIF_simple = {'tau': 5e-3, 'tau_G': 500e-3, 'G_r': 25e-3, 'p': p, 'V_T': -50e-3,
                      'V_R': -70e-3, 'n': 1, 'N_input': 1, 'dt': dt, 'refrac': 0}

    isi = int((t_total/dt)/125)
    input = torch.full((N_steps, 1), 0)



    input[int(0.1/dt)::isi] = 1

    lif = LIF_simple(parameters_LIF_simple)
    lif.initialize_state()

    voltage, time, spike_list = np.empty(input.shape), np.array([]), np.empty(input.shape)
    for i in range(N_steps):
        voltage[i, :], spike_list[i, :] = lif.forward(input[i])
        time = np.append(time, i * dt)

    plt.plot(time, voltage)
plt.plot(time, np.full(time.shape, -0.050), linestyle='dotted', color='black')
plt.show()
'''

