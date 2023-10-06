from dictionaries import Parameters
from class_velocity_neuron import LIF_simple
from class_primitive_neuron import LIF_primitive
from class_position_neuron import LIF
from class_sensory_neuron import AdEx
from class_hair_field import HairField
from plots import *
from functions import *

variables = {'dt': 0.001, 't_total': 10, 'N_steps': None}
variables['N_steps'] = round(variables['t_total']/variables['dt'])

data = pickle_open('simulation_data')

joint_angle = data[f'simulation_0'][0][:1000]
joint_angle = interpolate(joint_angle, variables['t_total'], variables['N_steps'])

parameters = Parameters(np.max(joint_angle), np.min(joint_angle), variables['dt'], 20)

hair_field = HairField(parameters.hair_field)
hair_field.get_double_receptive_field()

neurons = [AdEx, LIF, LIF_simple, LIF_primitive]
parameters_list = [parameters.sensory, parameters.position, parameters.velocity, parameters.primitive]
sensory_neuron, position_neuron, velocity_neuron, primitive_neuron = [define_and_initialize(neurons[i], parameters_list[i])
                                                                      for i in range(len(neurons))]

hair_angles = hair_field.get_hair_angle(joint_angle)

time, spike_list, spike_inter, spike_primitive, spike_position = np.array([]), torch.empty(hair_angles.shape),\
                                                 torch.empty([variables['N_steps'], parameters.velocity['n']]),\
                                                 torch.empty([variables['N_steps'], parameters.primitive['n']]),\
                                                 torch.empty([variables['N_steps'], parameters.position['n']])

for i in tqdm(range(variables['N_steps'])):
    _, spike_list[i, :] = sensory_neuron.forward(hair_angles[i, :])

    reshaped_spikes = torch.reshape(spike_list[i, :], (parameters.velocity['n'], (parameters.hair_field['N_hairs'])))

    _, spike_inter[i, :] = velocity_neuron.forward(reshaped_spikes)
    _, spike_position[i, :] = position_neuron.forward(reshaped_spikes[:, 10:])

    pos_vel_spikes = torch.Tensor((spike_inter[i, 0], spike_position[i, 0]))

    _, spike_primitive[i, :] = primitive_neuron.forward(pos_vel_spikes)
    time = np.append(time, i * parameters.sensory['dt'])

print(spike_primitive.shape)


#plt.plot(time, joint_angle, color='black')
#plt.scatter(time, joint_angle*spike_position[:, 1].numpy(), s=5, color = 'blue')
#plt.scatter(time, joint_angle*spike_position[:, 0].numpy(), s=5, color = 'red'

plt.scatter(time, 50*spike_inter[:, 0]*spike_position[:, 0])
plt.scatter(time, 25*spike_primitive)
plt.scatter(time, 15*spike_inter[:, 0])
plt.scatter(time, 7.5*spike_position[:, 0])


plt.show()
