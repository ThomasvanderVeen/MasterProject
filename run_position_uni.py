from dictionaries import Parameters
from class_hair_field import HairField
from class_sensory_neuron import AdEx
from class_position_neuron import LIF
from plots import *
from functions import *
import matplotlib.pyplot as plt

variables = {'dt': 0.0001, 't_total': 10, 'N_steps': None}
variables['N_steps'] = round(variables['t_total']/variables['dt'])

data = pickle_open('simulation_data')

joint_angle = data[f'simulation_0'][0][:1000]
joint_angle = interpolate(joint_angle, variables['t_total'], variables['N_steps'])

parameters = Parameters(np.max(joint_angle), np.min(joint_angle), variables['dt'], N_hairs=10)
parameters.sensory['n'] = int(parameters.sensory['n']/2)
parameters.position['n'] = 1

hair_field = HairField(parameters.hair_field)
hair_field.get_receptive_field()
hair_angles = hair_field.get_hair_angle(joint_angle)/37e9

neurons = [AdEx, LIF]
parameters_list = [parameters.sensory, parameters.position]
sensory_neuron, position_neuron = [define_and_initialize(neurons[i], parameters_list[i]) for i in range(len(neurons))]

time, spike_list, spike_inter = np.array([]), torch.empty(hair_angles.numpy().shape), np.empty([variables['N_steps']])

for i in tqdm(range(variables['N_steps'])):
    _, spike_list[i, :] = sensory_neuron.forward(hair_angles[i, :])

    _, spike_inter[i] = position_neuron.forward(spike_list[i, :])
    time = np.append(time, i * variables['dt'])

firing_rate, spike_index = get_firing_rate(spike_inter, variables['dt'])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(time[spike_index], firing_rate, color='blue')
ax2.plot(time, joint_angle, color='red')
plot_position_interneuron(ax1, ax2, fig, 'uni')
