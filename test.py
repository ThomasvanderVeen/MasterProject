import torch
from ramp_generator import RampGenerator
import matplotlib.pyplot as plt
from LIF_class import LIF
import numpy as np

parameters_LIF = {'E_L': -70e-3, 'V_T': -50e-3, 'tau': 50e-3, 'tau_W': 10e-3, 'tau_epsp': 50e-3, 'b': 20e-3, 'V_R': -70e-3, 'n': 1, 'N_input': 5, 'dt': 0.0001, 'refrac': 0}
lif = LIF(parameters_LIF)
lif.initialize_state()
N_steps = 10000
voltage_inter, spike_inter, w, I, time = np.empty([N_steps]), np.empty([N_steps]), np.empty([N_steps]), np.empty([N_steps]), np.array([])

spikes = np.zeros([N_steps, parameters_LIF['N_input']])
spikes[1000::2000, :] = 1
spikes = torch.from_numpy(spikes)

for i in range(N_steps):
    voltage_inter[i], spike_inter[i], w[i], I[i] = lif.forward(spikes[i])
    time = np.append(time, i * parameters_LIF['dt'])

plt.plot(time, w)
plt.show()
plt.plot(time, voltage_inter)
plt.show()
plt.plot(time, spike_inter)
plt.show()
plt.plot(time, I)
plt.show()
