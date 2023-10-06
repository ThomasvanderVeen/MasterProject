from dictionaries import Parameters
from class_hair_field import HairField
from class_sensory_neuron import AdEx
from class_position_neuron import LIF
from plots import *
import matplotlib.pyplot as plt
from functions import *

variables = {'dt': 0.0001, 't_total': 10, 'N_steps': None}
variables['N_steps'] = round(variables['t_total']/variables['dt'])

data = pickle_open('simulation_data')

joint_angle = data[f'simulation_0'][0][:1000]
joint_angle = interpolate(joint_angle, variables['t_total'], variables['N_steps'])

parameters = Parameters(np.max(joint_angle), np.min(joint_angle), variables['dt'], N_hairs=10)

hair_field = HairField(parameters.hair_field)
hair_field.get_binary_receptive_field()

neurons = [AdEx, LIF]
parameters_list = [parameters.sensory, parameters.position]
sensory_neuron, position_neuron = [define_and_initialize(neurons[i], parameters_list[i]) for i in range(len(neurons))]

hair_angles = hair_field.get_hair_angle(joint_angle)/37e9

time, spike_list, spike_inter = np.array([]), torch.empty(hair_angles.shape),\
                                np.empty([variables['N_steps'], parameters.position['n']])

for i in tqdm(range(variables['N_steps'])):
    _, spike_list[i, :] = sensory_neuron.forward(hair_angles[i, :])
    reshaped_spikes = torch.reshape(spike_list[i, :], (parameters.position['n'], (parameters.hair_field['N_hairs'])))
    _, spike_inter[i, :] = position_neuron.forward(reshaped_spikes)
    time = np.append(time, i * parameters.sensory['dt'])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

for j in range(parameters.position['n']):
    firing_rate, spikes_index = get_firing_rate(spike_inter[:, j], variables['dt'])

    ax1.plot(time[spikes_index], firing_rate)

ax2.plot(time, joint_angle, color='black')
ax2.plot(time, np.full(time.shape, np.max(joint_angle)/2 + np.min(joint_angle)/2), linestyle='dotted', color='black')
plot_position_interneuron(ax1, ax2, fig, 'bi')

fig2, ax3 = plt.subplots()
ax4 = ax3.twinx()

for i in range(hair_field.N_hairs):
    height = np.mean(hair_field.receptive_field[:, i])
    spike_list[spike_list == 0] = np.nan
    ax3.scatter(time, i*spike_list[:, i], color='dodgerblue', s=1)
ax4.plot(time, joint_angle, color='black')
ax4.plot(time, np.full(time.shape, np.max(joint_angle)/2 + np.min(joint_angle)/2), linestyle='dotted', color='red')
plot_spike_timing(ax3, ax4, fig2, hair_field.N_hairs)

