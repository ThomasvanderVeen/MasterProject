from AdEx_class import AdEx
from ramp_generator import RampGenerator
import numpy as np
from plots import *

v_var = np.array([24, 47, 88, 151, 245])
max_angle_var = np.array([15, 23, 34, 46, 60])
v_stat = 47
max_angle_stat = 37
parameters = {'C': 200e-12, 'g_L': 2e-9, 'E_L': -70e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'V_T': -50e-3,
              'tau_W': 600e-3, 'b': 8e-12, 'V_R': -58e-3, 'V_cut': -30e-3, 'refrac': 0.00, 'n': 5,
              'dt': 0.0001}
t_total = 10
colors = ['green', 'yellow', 'blue', 'black', 'red']

for [v, max_angle] in [[v_var, max_angle_stat], [v_stat, max_angle_var]]:

    N_steps = round(t_total/parameters['dt'])
    N_ramp = np.around(max_angle/(v*parameters['dt'])).astype(int)
    height = np.empty([parameters['n']])
    height[:] = max_angle/37e9

    parameters_ramp = {'n_ramp': N_ramp, 'n_steps': N_steps, 'height': height}
    ramp_generator = RampGenerator(parameters_ramp)
    input_ramp = ramp_generator.ramp()

    neuron = AdEx(parameters)
    neuron.initialize_state()

    voltage, time, spike_list = np.empty(input_ramp.numpy().shape), np.array([]), np.empty(input_ramp.numpy().shape)
    for i in range(N_steps):
        voltage[i, :], spike_list[i, :] = neuron.forward(input_ramp[i])
        time = np.append(time, i*parameters['dt'])

    for j in range(parameters['n']):
        spikes = np.where(spike_list[:, j] == 1)

        ISI = np.diff(spikes)

        t_spikes = np.append(0, time[spikes])
        spike_rate = np.append(0, 1/(ISI*parameters['dt']))
        spike_rate = np.append(spike_rate, 0)
        spike_rate = np.append(spike_rate, 0)
        t_spikes = np.append(t_spikes, time[-1])

        plt.plot(t_spikes, spike_rate)

    plot_single_hair(plt.gca(), v)
