from dictionaries import Parameters
from class_hair_field import HairField
from class_sensory_neuron import AdEx
from class_position_neuron import LIF
from plots import *
from functions import *
import matplotlib.pyplot as plt

data = pickle_open('simulation_data')

joint_angle = np.array(data[f'simulation_3'][0][0]).T

parameters = Parameters(max_joint_angle=np.max(joint_angle), min_joint_angle=np.min(joint_angle),
                        N_hairs=20, t_total=7.5, dt=0.0001, N_sims=1)

joint_angle = joint_angle[:parameters.general['N_frames']]
joint_angle = interpolate(joint_angle, parameters.general['t_total'], parameters.general['N_steps'])
parameters.sensory['n'] = int(parameters.sensory['n']/2)
parameters.position['n'] = 1

hair_field = HairField(parameters.hair_field)
hair_field.get_receptive_field()
hair_angles = hair_field.get_hair_angle(joint_angle)/37e9

neurons = [AdEx, LIF]
parameters_list = [parameters.sensory, parameters.position]
sensory_neuron, position_neuron = [define_and_initialize(neurons[i], parameters_list[i]) for i in range(len(neurons))]

time, spike_list, spike_inter = np.array([]), torch.empty(hair_angles.numpy().shape), np.empty([parameters.general['N_steps']])

for i in tqdm(range(parameters.general['N_steps'])):
    _, spike_list[i, :] = sensory_neuron.forward(hair_angles[i, :])

    _, spike_inter[i] = position_neuron.forward(spike_list[i, :])
    time = np.append(time, i * parameters.general['dt'])

firing_rate, spike_index = get_firing_rate(spike_inter, parameters.general['dt'])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(time[spike_index], firing_rate, color='blue')
ax2.plot(time, joint_angle, color='red')
plot_position_interneuron(ax1, ax2, fig, 'uni')
