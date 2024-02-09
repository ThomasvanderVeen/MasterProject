from class_velocity_neuron import LIF_simple
from dictionaries import Parameters
from functions import *
import matplotlib.pyplot as plt


parameters = Parameters(t_total=5, dt=0.001, n_hairs=1)

N = 10

parameters.velocity['n'] = 1
parameters.velocity['N_input'] = N
parameters.velocity['p'] = 0.1
parameters.velocity['tau_min'] = 10e-3

velocity_neuron = define_and_initialize(LIF_simple, parameters.velocity)

N_steps = 300

input_1, input_2 = torch.zeros(N_steps), torch.zeros(N_steps)

input_1[100, 0] = 1
input_1[103, 1] = 1
input_1[106, 2] = 1
input_1[109, 3] = 1
input_1[112, 4] = 1
input_1[115, 5] = 1

voltage, spikes, h = np.zeros(N_steps), np.zeros(N_steps), np.zeros(N_steps)

for i in range(N_steps):
    voltage[i], spikes[i], h = velocity_neuron.forward(input_1[i], input_2[i])

plt.scatter(range(N_steps), 2*input_1[:, 0])


plt.scatter(range(N_steps), spikes)
#plt.plot(range(N_steps), h)
plt.show()
#plt.plot(range(N_steps), voltage)
#plt.show()

print(h)