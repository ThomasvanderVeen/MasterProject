from LIF_simple_class import LIF_simple
import numpy as np
import torch
import matplotlib.pyplot as plt



dt = 0.0001
t_total = 1
N_steps = round(t_total/dt)

for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    parameters_LIF_simple = {'tau': 5e-3, 'tau_G': 500e-3, 'G_r': 25e-3, 'p': p, 'V_T': 50e-3,
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


