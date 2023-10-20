from dictionaries import Parameters
from class_velocity_neuron import LIF_simple
from class_primitive_neuron import LIF_primitive
from class_position_neuron import LIF
from class_sensory_neuron import AdEx
from class_hair_field import HairField
from plots import *
from functions import *

w_pos = [11e-3, 0, 12e-3, 10e-3, 7.5e-3]
w_vel = [13e-3, 13e-3, 0, 11e-3, 12e-3]
_, _, weights_primitive, _, primitive_filter = get_encoding(w_pos, w_vel)
permutations = get_primitive_indexes(6)
data = pickle_open('simulation_data')
joint_angles = np.array(data[f'simulation_{0}'][0]).T

parameters = Parameters(max_joint_angle=np.amax(joint_angles, axis=0), min_joint_angle=np.amin(joint_angles, axis=0),
                        N_hairs=10, t_total=8, dt=0.001, N_sims=18)
parameters.position['N_input'] = int(parameters.position['N_input']/2)
parameters.primitive['n'] = 360
parameters.primitive['w'] = weights_primitive

joint_angles = joint_angles[:parameters.general['N_frames']]
joint_angles = interpolate(joint_angles, parameters.general['t_total'], parameters.general['N_steps'])

hair_angles = np.zeros((joint_angles.shape[0], 2*joint_angles.shape[1]*parameters.hair_field['N_hairs']))

for i in range(parameters.general['N_sims']):
    hair_field = HairField(parameters.hair_field)
    hair_field.reset_max_min(i)
    hair_field.get_double_receptive_field()
    hair_angles[:, i * 2 * parameters.hair_field['N_hairs']: 2 * parameters.hair_field['N_hairs']
                + i * 2 * parameters.hair_field['N_hairs']] = hair_field.get_hair_angle(joint_angles[:, i])

neurons = [AdEx, LIF, LIF_simple, LIF_primitive]
parameters_list = [parameters.sensory, parameters.position, parameters.velocity, parameters.primitive]
sensory_neuron, position_neuron, velocity_neuron, primitive_neuron = \
    [define_and_initialize(neurons[i], parameters_list[i]) for i in range(len(neurons))]

time, spike_sensory = np.array([]), torch.empty(hair_angles.shape)
spike_position, spike_velocity, spike_primitive = \
    [torch.empty((parameters.general['N_steps'], par['n'])) for par in parameters_list[1:]]

for i in tqdm(range(parameters.general['N_steps'])):
    time = np.append(time, i)
    _, spike_sensory[i, :] = sensory_neuron.forward(torch.from_numpy(hair_angles[i, :]))

    reshaped_spikes = torch.reshape(spike_sensory[i, :], (parameters.velocity['n'], (parameters.hair_field['N_hairs'])))

    _, spike_velocity[i, :] = velocity_neuron.forward(reshaped_spikes)
    _, spike_position[i, :] = position_neuron.forward(reshaped_spikes[:, 5:])

    pos_vel_spikes = prepare_spikes_primitive(spike_velocity, spike_position, permutations, primitive_filter)

    _, spike_primitive[i, :] = primitive_neuron.forward(pos_vel_spikes)

'''
position neuron testing
'''

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

for i in range(2):
    firing_rate, spikes_index = get_firing_rate(spike_position[:, i], parameters.general['dt'])

    ax1.plot(time[spikes_index], firing_rate)

ax2.plot(time, joint_angles[:, 0], color='black')
ax2.plot(time, np.full(len(time), np.max(joint_angles[:, 0])/2 + np.min(joint_angles[:, 0])/2), linestyle='dotted',
         color='black')
plot_position_interneuron(ax1, ax2, fig, 'bi')

fig2, ax3 = plt.subplots()
ax4 = ax3.twinx()

for i in range(20):
    height = np.mean(hair_field.receptive_field[:, i])
    spike_sensory[spike_sensory == 0] = np.nan
    ax3.scatter(time, i*spike_sensory[:, i], color='dodgerblue', s=1)
    ax3.scatter(time, i * spike_sensory[:, i], color='red', s=1)
    #ax3.scatter(time, i * spike_sensory[:, i], color='dodgerblue', s=1)
ax4.plot(time, joint_angles[:, 0], color='black')
ax4.plot(time, np.full(time.shape, np.max(joint_angles[:, 0])/2 + np.min(joint_angles[:, 0])/2), linestyle='dotted', color='red')
plot_spike_timing(ax3, ax4, fig2, hair_field.N_hairs)

'''
velocity neuron testing
'''

spike_velocity = spike_velocity.numpy()
spike_velocity[spike_velocity == 0] = np.nan

fig, ax = plt.subplots()

plt.plot(time, joint_angles[:, 0], color='black')
plt.scatter(time, joint_angles[:, 0]*spike_velocity[:, 0], color='blue')
plt.scatter(time, joint_angles[:, 0]*spike_velocity[:, 1], color='red')

plot_movement_interneuron_network(ax, fig)