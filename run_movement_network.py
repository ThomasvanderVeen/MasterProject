from dictionaries import Parameters
from class_velocity_neuron import LIF_simple
from class_sensory_neuron import AdEx
from class_hair_field import HairField
from plots import *
from functions import *

variables = {'dt': 0.001, 't_total': 10, 'N_steps': None}
variables['N_steps'] = round(variables['t_total']/variables['dt'])

data = pickle_open('simulation_data')

joint_angle = data[f'simulation_0'][0][:1000]
joint_angle = interpolate(joint_angle, variables['t_total'], variables['N_steps'])

parameters = Parameters(np.max(joint_angle), np.min(joint_angle), variables['dt'], N_hairs=20)

hair_field = HairField(parameters.hair_field)
hair_field.get_double_receptive_field()

neurons = [AdEx, LIF_simple]
parameters_list = [parameters.sensory, parameters.velocity]
sensory_neuron, velocity_neuron = [define_and_initialize(neurons[i], parameters_list[i]) for i in
                                   range(len(neurons))]

hair_angles = hair_field.get_hair_angle(joint_angle)

time, spike_list, spike_inter = np.array([]), torch.empty(hair_angles.shape), np.empty([variables['N_steps'],
                                                                                        parameters.velocity['n']])
for i in tqdm(range(variables['N_steps'])):
    _, spike_list[i, :] = sensory_neuron.forward(hair_angles[i, :])
    reshaped_spikes = torch.reshape(spike_list[i, :], (parameters.velocity['n'], (parameters.hair_field['N_hairs'])))
    _, spike_inter[i, :] = velocity_neuron.forward(reshaped_spikes)
    time = np.append(time, i * parameters.sensory['dt'])

spike_inter[spike_inter == 0] = np.nan

fig, ax = plt.subplots()

plt.plot(time, joint_angle, color='black')
plt.scatter(time, joint_angle*spike_inter[:, 0], color='blue')
plt.scatter(time, joint_angle*spike_inter[:, 1], color='red')

plot_movement_interneuron_network(ax, fig)
