import matplotlib.pyplot as plt

from class_hair_field import HairField
from class_position_neuron import LIF
from class_sensory_neuron import AdEx
from dictionaries import Parameters
from functions import *
from plots import *

data = pickle_open('Data/simulation_data')

joint_angle = np.array(data[f'simulation_0'][0][0]).T

parameters = Parameters(max_joint_angle=np.max(joint_angle), min_joint_angle=np.min(joint_angle),
                        n_hairs=100, t_total=7.5, dt=0.0002, n_angles=1)

joint_angle = joint_angle[:parameters.general['N_frames']]
joint_angle = interpolate(joint_angle, parameters.general['t_total'], parameters.general['N_steps'])

parameters.sensory['n'] = int(parameters.sensory['n'] / 2)
parameters.position['N_input'] = int(parameters.hair_field['N_hairs'])
parameters.position['n'] = 1

hair_field = HairField(parameters.hair_field)
hair_field.get_receptive_field()
hair_angles = hair_field.get_hair_angle(joint_angle) / 37e9

neurons = [AdEx, LIF]
parameters_list = [parameters.sensory, parameters.position]
sensory_neuron, position_neuron = [define_and_initialize(neurons[i], parameters_list[i]) for i in range(len(neurons))]

time, spike_sensory, spike_position = np.array([]), torch.empty(hair_angles.shape), np.empty(
    [parameters.general['N_steps']])

for i in tqdm(range(parameters.general['N_steps'])):
    _, spike_sensory[i, :] = sensory_neuron.forward(torch.from_numpy(hair_angles[i, :]))
    _, spike_position[i] = position_neuron.forward(spike_sensory[i, :])

    time = np.append(time, i * parameters.general['dt'])

firing_rate = get_firing_rate_2(spike_position, parameters.general['dt'], t=0.05)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(time[::100], joint_angle[::100], color=parameters.general['colors'][0], linestyle=parameters.general['linestyles'][0])
ax2.plot(time[::100], firing_rate[::100], color=parameters.general['colors'][1], linestyle=parameters.general['linestyles'][1])
plot_position_interneuron(ax1, ax2, fig, 'uni')
