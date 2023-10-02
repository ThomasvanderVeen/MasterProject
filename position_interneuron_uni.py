from HairField_class import HairField
from AdEx_class import AdEx
from LIF_class import LIF
from plots import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

dt = 0.001
t_total = 10
N_steps = round(t_total/dt)

file = open('simulation_data', 'rb')
data = pickle.load(file)
file.close()

y = data[f'simulation_0'][0][:1000]
x = np.linspace(0, t_total, num=y.shape[0])
xvals = np.linspace(0, t_total, num=N_steps)
joint_angle = np.interp(xvals, x, y)

parameters_hair_field = {'N_hairs': 5, 'max_joint_angle': np.max(joint_angle), 'min_joint_angle': np.min(joint_angle),
                         'max_angle': 90, 'overlap': 4, 'overlap_bi': 18}

hair_field = HairField(parameters_hair_field)
hair_field.get_receptive_field()
hair_angles = hair_field.get_hair_angle(joint_angle)/37e9

parameters_AdEx = {'C': 200e-12, 'g_L': 2e-9, 'E_L': -70e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'V_T': -50e-3, 'tau_W': 6e-3,
                   'b': 8e-12, 'V_R': -58e-3, 'V_cut': -30e-3, 'refrac': 0.00, 'n': hair_field.N_hairs, 'dt': dt}

neuron = AdEx(parameters_AdEx)
neuron.initialize_state()

parameters_LIF = {'E_L': -70e-3, 'V_T': -50e-3, 'tau': 50e-3, 'tau_W': 10e-3, 'tau_epsp': 50e-3, 'b': 15e-3,
                  'V_R': -70e-3, 'n': 1, 'N_input': hair_field.N_hairs, 'dt': dt, 'refrac': 0}
lif = LIF(parameters_LIF)
lif.initialize_state()

voltage, time, spike_list = np.empty(hair_angles.numpy().shape), np.array([]), torch.empty(hair_angles.numpy().shape)
voltage_inter, spike_inter = np.empty([N_steps]), np.empty([N_steps])

for i in range(N_steps):
    voltage[i, :], spike_list[i, :] = neuron.forward(hair_angles[i, :])
    voltage_inter[i], spike_inter[i] = lif.forward(spike_list[i, :])
    time = np.append(time, i * dt)
    print(time[-1])

spikes = np.where(spike_inter == 1)[0]
ISI = np.diff(spikes)
firing_rate = 1/(ISI*parameters_LIF['dt'])
firing_rate = np.append(firing_rate, firing_rate[-1])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(time[spikes], firing_rate-np.min(firing_rate), color='red')
ax2.plot(time, joint_angle-np.min(joint_angle), color='blue')
plot_position_interneuron(ax1, ax2, fig, 'uni')

plt.scatter(time, spike_list[:, 0])
plt.show()
