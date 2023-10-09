from dictionaries import Parameters
from class_velocity_neuron import LIF_simple
from class_primitive_neuron import LIF_primitive
from class_position_neuron import LIF
from class_sensory_neuron import AdEx
from class_hair_field import HairField
from plots import *
from functions import *
import itertools


allCombo = np.array(list(itertools.permutations([-np.inf, 0, 1, 2, 3], 3)))
extra = np.array([0, 4, 8])
extra = np.tile(extra, (allCombo.shape[0], 1))
allCombo = allCombo.astype(int) + extra

indexes_two = np.where(allCombo < 0)[0]
indexes_three = np.where(allCombo > -1)[0]
two = allCombo[indexes_two, :]
two = two[np.where(two > 0)].reshape(33, 2)
print(indexes_three)

three = allCombo[indexes_three, :].reshape(27, 2)

print(allCombo, two, three)

allCombo = np.ndarray.flatten(allCombo)

N_sims = 3

data = pickle_open('simulation_data')

joint_angle = np.array(data[f'simulation_0'][:N_sims]).T

parameters = Parameters(max_joint_angle=np.amax(joint_angle, axis=0), min_joint_angle=np.amin(joint_angle, axis=0),
                        N_hairs=20, t_total=7.5, dt=0.001, N_sims=N_sims)
parameters.position['N_input'] = int(parameters.position['N_input']/2)

joint_angle = joint_angle[:parameters.general['N_frames']]
joint_angles = np.empty((parameters.general['N_steps'], N_sims))
hair_angles = np.empty((parameters.general['N_steps'], 2*N_sims*parameters.hair_field['N_hairs']))

for i in range(N_sims):
    joint_angles[:, i] = interpolate(joint_angle[:, i], parameters.general['t_total'], parameters.general['N_steps'])
    hair_field = HairField(parameters.hair_field)
    hair_field.reset_max_min(i)
    hair_field.get_double_receptive_field()
    hair_angles[:, i*2*parameters.hair_field['N_hairs']: 2*parameters.hair_field['N_hairs']+i*2*parameters.hair_field['N_hairs']]\
        = hair_field.get_hair_angle(joint_angles[:, i])

hair_angles = torch.from_numpy(hair_angles)

neurons = [AdEx, LIF, LIF_simple, LIF_primitive]
parameters_list = [parameters.sensory, parameters.position, parameters.velocity, parameters.primitive]
sensory_neuron, position_neuron, velocity_neuron, primitive_neuron = [define_and_initialize(neurons[i], parameters_list[i])
                                                                      for i in range(len(neurons))]

time, spike_sensory, spike_velocity, spike_primitive, spike_position = np.array([]), torch.empty(hair_angles.shape),\
                                                 torch.empty([parameters.general['N_steps'], parameters.velocity['n']]),\
                                                 torch.empty([parameters.general['N_steps'], parameters.primitive['n']]),\
                                                 torch.empty([parameters.general['N_steps'], parameters.position['n']])

for i in tqdm(range(parameters.general['N_steps'])):
    _, spike_sensory[i, :] = sensory_neuron.forward(hair_angles[i, :])

    reshaped_spikes = torch.reshape(spike_sensory[i, :], (parameters.velocity['n'], (parameters.hair_field['N_hairs'])))

    _, spike_velocity[i, :] = velocity_neuron.forward(reshaped_spikes)
    _, spike_position[i, :] = position_neuron.forward(reshaped_spikes[:, 10:])
    pos_vel_spikes = torch.concat((spike_velocity[i, [0, 1]], spike_position[i, [0, 1]],
                                  spike_velocity[i, [2, 3]], spike_position[i, [2, 3]],
                                  spike_velocity[i, [4, 5]], spike_position[i, [4, 5]]))

    pos_vel_spikes = pos_vel_spikes[allCombo]
    pos_vel_spikes = np.reshape(pos_vel_spikes, (60, 3))

    _, spike_primitive[i, :] = primitive_neuron.forward(pos_vel_spikes)
    time = np.append(time, i * parameters.sensory['dt'])


plt.plot(time, joint_angles[:, 0], color='black')
plt.scatter(time, joint_angles[:, 0]*spike_position[:, 0].numpy(), s=5, color = 'blue')
plt.scatter(time, joint_angles[:, 0]*spike_position[:, 1].numpy(), s=5, color = 'red')
plt.scatter(time, joint_angles[:, 0]*spike_velocity[:, 0].numpy(), s=25, color = 'yellow')
plt.scatter(time, joint_angles[:, 0]*spike_velocity[:, 1].numpy(), s=25, color = 'green')

#plt.scatter(time, 50*spike_velocity[:, 0]*spike_position[:, 0], s=1)
#plt.scatter(time, 25*spike_primitive, s=1)
#plt.scatter(time, 15*spike_velocity[:, 0], s=1)
#plt.scatter(time, 7.5*spike_position[:, 0], s=1)


plt.show()
