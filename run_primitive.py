from dictionaries import Parameters
from class_velocity_neuron import LIF_simple
from class_primitive_neuron import LIF_primitive
from class_position_neuron import LIF
from class_sensory_neuron import AdEx
from class_hair_field import HairField
from plots import *
from functions import *

N_simulations = 2
w_pos = [11e-3, 0, 12e-3, 10e-3, 7.5e-3]
w_vel = [13e-3, 13e-3, 0, 11e-3, 12e-3]
colors = ['blue', 'black', 'green', 'yellow', 'orange']
data = pickle_open('simulation_data')
for m in range(1):
    permutations, synapse_type, weights_primitive, primitive_filter, primitive_filter_2 = get_encoding(w_pos, w_vel)
    true_positive, false_positive, true_negative, false_negative = [np.empty((N_simulations, 60)) for _ in range(4)]
    swing_bin_rate, stance_bin_rate, swing_bin_likelihood, stance_bin_likelihood = [np.empty((N_simulations, 60, i)) for
                                                                                    i in [10, 20, 10, 20]]

    for k in tqdm(range(N_simulations)):

        joint_angle, gait = np.array(data[f'simulation_{k}'][0][3*m:3+3*m]).T, np.array(data[f'simulation_{k}'][1])[m, :]
        #print(np.amax(joint_angle, axis=0), np.amin(joint_angle, axis=0))
        #plt.plot(np.linspace(0, 4, num=joint_angle.shape[0]), joint_angle)
        #plt.show()
        parameters = Parameters(max_joint_angle=np.amax(joint_angle, axis=0), min_joint_angle=np.amin(joint_angle, axis=0),
                                N_hairs=20, t_total=8, dt=0.001, N_sims=3)
        parameters.position['N_input'] = int(parameters.position['N_input']/2)
        parameters.primitive['w'] = weights_primitive

        ground_truth = np.zeros([12, parameters.general['N_steps']])

        joint_angle = joint_angle[:parameters.general['N_frames']]
        gait = gait[:parameters.general['N_frames']]

        gait = interpolate(gait, parameters.general['t_total'], parameters.general['N_steps'], True)

        joint_angles = np.empty((parameters.general['N_steps'], parameters.general['N_sims']))
        hair_angles = np.empty((parameters.general['N_steps'], 2*parameters.general['N_sims']*parameters.hair_field['N_hairs']))

        for i in range(parameters.general['N_sims']):
            joint_angles[:, i] = interpolate(joint_angle[:, i], parameters.general['t_total'], parameters.general['N_steps'])

            mid = np.max(joint_angles[:, i])/2 + np.min(joint_angles[:, i])/2
            diff = np.diff(joint_angles[:, i])
            ground_truth[0 + 4 * i, np.where(diff > 0)] = 1
            ground_truth[1 + 4 * i, np.where(diff < 0)] = 1
            ground_truth[2 + 4 * i, np.where(joint_angles[:, i] > mid)] = 1
            ground_truth[3 + 4 * i, np.where(joint_angles[:, i] < mid)] = 1

            hair_field = HairField(parameters.hair_field)
            hair_field.reset_max_min(i)
            hair_field.get_double_receptive_field()
            hair_angles[:, i*2*parameters.hair_field['N_hairs']: 2*parameters.hair_field['N_hairs']+i*2*parameters.hair_field['N_hairs']]\
                = hair_field.get_hair_angle(joint_angles[:, i])/37e9

        hair_angles = torch.from_numpy(hair_angles)

        neurons = [AdEx, LIF, LIF_simple, LIF_primitive]
        parameters_list = [parameters.sensory, parameters.position, parameters.velocity, parameters.primitive]
        sensory_neuron, position_neuron, velocity_neuron, primitive_neuron = [define_and_initialize(neurons[i], parameters_list[i])
                                                                              for i in range(len(neurons))]

        time, spike_sensory, spike_velocity, spike_primitive, spike_position = np.array([]), torch.empty(hair_angles.shape),\
                                                         torch.empty([parameters.general['N_steps'], parameters.velocity['n']]),\
                                                         torch.empty([parameters.general['N_steps'], parameters.primitive['n']]),\
                                                         torch.empty([parameters.general['N_steps'], parameters.position['n']])

        permutations = get_primitive_indexes(1)
        spike_timings = torch.empty([parameters.general['N_steps'], parameters.primitive['n']])

        for i in range(parameters.general['N_steps']):
            _, spike_sensory[i, :] = sensory_neuron.forward(hair_angles[i, :])

            reshaped_spikes = torch.reshape(spike_sensory[i, :], (parameters.velocity['n'], (parameters.hair_field['N_hairs'])))

            _, spike_velocity[i, :] = velocity_neuron.forward(reshaped_spikes)
            _, spike_position[i, :] = position_neuron.forward(reshaped_spikes[:, int(parameters.hair_field['N_hairs']/2):])

            pos_vel_spikes = torch.concat((spike_velocity[i, [0, 1]], spike_position[i, [0, 1]],
                                          spike_velocity[i, [2, 3]], spike_position[i, [2, 3]],
                                          spike_velocity[i, [4, 5]], spike_position[i, [4, 5]]))

            pos_vel_spikes = pos_vel_spikes[permutations].reshape((parameters.primitive['n'], 3))*primitive_filter_2
            ground_truth_i = ground_truth[permutations, i].reshape((parameters.primitive['n'], 3))*primitive_filter_2
            #spike_combinations = torch.sum(pos_vel_spikes + primitive_filter, dim=1) / 3
            spike_combinations = torch.sum(torch.from_numpy(ground_truth_i)+primitive_filter, dim=1)/3
            spike_combinations[spike_combinations < 0.9] = 0
            spike_timings[i, :] = spike_combinations

            _, spike_primitive[i, :] = primitive_neuron.forward(pos_vel_spikes)

            time = np.append(time, i * parameters.sensory['dt'])

        spike_primitive_bins = convert_to_bins(spike_primitive.numpy(), 100)
        spike_timings_bins = convert_to_bins(spike_timings.numpy(), 100)

        #plt.plot(time, joint_angles[:, 0])
        #plt.scatter(time, joint_angles[:, 0]*spike_velocity[:, 0].numpy())
        #plt.show()

        for i in range(60):
            swing_bin_rate[k, i, :], stance_bin_rate[k, i, :], swing_bin_likelihood[k, i, :], stance_bin_likelihood[k, i, :] = get_stance_swing_bins(gait, spike_primitive[:, i].numpy())

            intersect = spike_primitive_bins[:, i] + spike_timings_bins[:, i]
            difference = spike_primitive_bins[:, i] - spike_timings_bins[:, i]

            true_positive[k, i] = intersect[intersect > 1.5].size
            false_positive[k, i] = difference[difference > 0.5].size
            true_negative[k, i] = intersect[intersect < 0.5].size
            false_negative[k, i] = difference[difference < -0.5].size


    swing_bin_likelihood, stance_bin_likelihood = np.mean(swing_bin_likelihood, axis=0), np.mean(stance_bin_likelihood, axis=0)


    for i in range(60):
        fig, ax = plt.subplots()
        ax.scatter(np.linspace(0, .475, num=10), swing_bin_likelihood[i, :], color='red', marker='^')
        ax.scatter(np.linspace(.525, 1.5, num=20), stance_bin_likelihood[i, :], color='blue', marker='^')
        ax.set_ylabel('Likelihood of Spiking')
        fig.text(0.33, 0.04, 'Swing', ha='center')
        fig.text(0.66, 0.04, 'Stance')
        ax.set_xticks([])
        fig.savefig(f'Images_PSTH/neuron_{i}_leg_{m}')
        ax.cla()

    fig, ax = plt.subplots()

    for i in range(60):

        true_pos_sum = np.sum(true_positive, axis=0)
        false_pos_sum = np.sum(false_positive, axis=0)
        true_neg_sum = np.sum(true_negative, axis=0)
        false_neg_sum = np.sum(false_negative, axis=0)
        plt.scatter(false_pos_sum[i]/(false_pos_sum[i] + true_neg_sum[i] + 0.0000001),
                    true_pos_sum[i]/(true_pos_sum[i] + false_neg_sum[i] + 0.0000001), color=colors[synapse_type[i]])


    plt.plot([0, 1], [0, 1], color='red', linestyle='dotted')
    plot_primitive_interneuron(ax, fig)

'''
for i in range(0, 60):

    plt.scatter(np.linspace(0, 10, num=100), 12*spike_timings[:, i], s=1)
    plt.scatter(np.linspace(0, 10, num=100), 25*spike_primitive[:, i], s=1)

    plt.show()
'''