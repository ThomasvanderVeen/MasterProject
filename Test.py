from class_velocity_neuron import LIF_simple
from dictionaries import Parameters
from functions import *
import matplotlib.pyplot as plt


a = torch.Tensor([[0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 0, 0]])
b = torch.Tensor([0, 0, 0])
print(a, a.shape)

print(fill_with_ones(a))



parameters = Parameters(t_total=5, dt=0.001, n_hairs=1)

parameters.velocity['n'] = 1
parameters.velocity['p'] = 0.1
parameters.velocity['tau_i'] = 3e-3




velocity_neuron = define_and_initialize(LIF_simple, parameters.velocity)

N_steps = 100

input_1, input_2 = torch.zeros(N_steps), torch.zeros(N_steps)

input_1[10::5] = 1

voltage, spikes = np.zeros(N_steps), np.zeros(N_steps)

for i in range(N_steps):
    voltage[i], spikes[i] = velocity_neuron.forward(input_1[i], input_2[i])

plt.scatter(range(N_steps), 2*input_1)


plt.scatter(range(N_steps), spikes)
plt.show()

print(np.sum(spikes))