from HairField_class import HairField
from AdEx_class import AdEx
from LIF_class import LIF
from plots import *
import numpy as np
import matplotlib.pyplot as plt
import torch

dt = 0.00001

parameters_hair_field = {'N_hairs': 5, 'max_joint_angle': 180, 'min_joint_angle': 70, 'max_angle': 90, 'overlap': 4}
hair_field = HairField(parameters_hair_field)
hair_field.get_receptive_field()
hair_field_2 = HairField(parameters_hair_field)
hair_field_2.receptive_field = parameters_hair_field['max_joint_angle']-hair_field.receptive_field

parameters_AdEx = {'C': 200e-12, 'g_L': 2e-9, 'E_L': -70e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'V_T': -50e-3,
              'tau_W': 6e-3, 'b': 8e-12, 'V_R': -58e-3, 'V_cut': -30e-3, 'refrac': 0.00, 'n': 10,
              'dt': dt}

neuron = AdEx(parameters_AdEx)
neuron.initialize_state()

parameters_LIF = {'E_L': -70e-3, 'V_T': -50e-3, 'tau': 50e-3, 'tau_W': 10e-3, 'tau_epsp': 50e-3, 'b': 15e-3, 'V_R': -70e-3, 'n': 1, 'N_input': 10, 'dt': dt, 'refrac': 0}
lif = LIF(parameters_LIF)
lif.initialize_state()
lif_2 = LIF(parameters_LIF)
lif_2.initialize_state()

t_total = 10
N_steps = round(t_total/dt)
joint_angle = np.linspace(1, 0.1, num=N_steps)*90*np.sin(np.linspace(1.5*np.pi, 9*np.pi, num=N_steps)) + 90
hair_angles_1 = hair_field.get_hair_angle(joint_angle)
hair_angles_2 = hair_field_2.get_hair_angle(joint_angle)

hair_angles = torch.from_numpy(np.hstack((hair_angles_1, hair_angles_2))/37e9)

voltage, time, spike_list = np.empty(hair_angles.shape), np.array([]), torch.empty(hair_angles.shape)
voltage_inter, spike_inter = np.empty([N_steps]), np.empty([N_steps])
voltage_inter_2, spike_inter_2 = np.empty([N_steps]), np.empty([N_steps])

for i in range(N_steps):
    voltage[i, :], spike_list[i, :] = neuron.forward(hair_angles[i, :])
    voltage_inter[i], spike_inter[i] = lif.forward(spike_list[i, 5:])
    voltage_inter_2[i], spike_inter_2[i] = lif_2.forward(spike_list[i, :5])
    time = np.append(time, i * parameters_AdEx['dt'])
    print(time[-1])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

for j in [spike_inter, spike_inter_2]:
    spikes = np.where(j == 1)[0]
    ISI = np.diff(spikes)*parameters_LIF['dt']
    firing_rate = 1/ISI
    firing_rate = np.append(firing_rate, firing_rate[-1])
    firing_rate[firing_rate < 40] = 0

    ax1.plot(time[spikes], firing_rate)

ax2.plot(time, joint_angle, color='blue')
ax2.plot(time, np.full(time.shape, 90), linestyle='dotted', color='black')
plot_position_interneuron(ax1, ax2, fig, 'bi')




