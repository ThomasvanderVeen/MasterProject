from dictionaries import Parameters
from class_velocity_neuron import LIF_simple
from class_primitive_neuron import LIF_primitive
from class_position_neuron import LIF
from class_sensory_neuron import AdEx
from class_hair_field import HairField
from plots import *
from functions import *

N_simulations = 2

true_positive, false_positive = np.empty((N_simulations, 60)), np.empty((N_simulations, 60))

for k in range(N_simulations):

    N_sims = 3
    colors = ['r', 'g', 'b', 'black', 'purple', 'purple']
    w_pos = [2e-3, 0, 15e-3, 10e-3, 5e-3]
    w_vel = [19e-3, 19e-3, 0, 11e-3, 9e-3]

    #w_pos = [12e-3, 0, 13e-3, 11e-3, 8e-3]
    #w_vel = [13.5e-3, 13e-3, 0, 12.5e-3, 13.5e-3]

    permutations, synapse_type, weights_primitive, primitive_filter, primitive_filter_2 = get_encoding(w_pos, w_vel)

    data = pickle_open('simulation_data')

    joint_angle = np.array(data[f'simulation_{k}'][:N_sims]).T

    parameters = Parameters(max_joint_angle=np.amax(joint_angle, axis=0), min_joint_angle=np.amin(joint_angle, axis=0),
                            N_hairs=20, t_total=7.5, dt=0.0003, N_sims=N_sims)
    parameters.position['N_input'] = int(parameters.position['N_input']/2)
    parameters.primitive['w'] = weights_primitive

    joint_angle = joint_angle[:parameters.general['N_frames']]
    joint_angles = np.empty((parameters.general['N_steps'], N_sims))
    hair_angles = np.empty((parameters.general['N_steps'], 2*N_sims*parameters.hair_field['N_hairs']))

    ground_truth = np.zeros([12, 75000])

    for i in range(N_sims):
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

    permutations = get_primitive_indexes(1)
    spike_timings = torch.empty([parameters.general['N_steps'], parameters.primitive['n']])

    for i in tqdm(range(parameters.general['N_steps'])):
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



    #plt.plot(time, joint_angles[:, 0])
    #plt.scatter(time, joint_angles[:, 0]*spike_position[:, 0].numpy())
    #plt.scatter(time, joint_angles[:, 0]*spike_velocity[:, 1].numpy())
    #plt.show()
    for i in range(10, 60):
        intersect = spike_primitive.numpy()[:, i] + spike_timings.numpy()[:, i]
        difference = spike_primitive.numpy()[:, i] - spike_timings.numpy()[:, i]
        true_positive[k, i] = intersect[intersect > 1.5].size
        false_positive[k, i] = difference[difference > 0.5].size

        #print(true_positive[, ki], false_positive[i], synapse_type[i])
        #print(true_positive/(true_positive + false_positive + 1), false_positive/(false_positive + true_positive + 1))
        #

for i in range(60):
    true_pos_sum = np.sum(true_positive, axis=0)
    false_pos_sum = np.sum(false_positive, axis=0)
    plt.scatter(i, true_pos_sum[i]/(true_pos_sum[i] + false_pos_sum[i] + 0.00001), color=colors[synapse_type[i]])

plt.show()

#intersect = spike_primitive.numpy() + spike_timings.numpy()
#difference = spike_primitive.numpy() - spike_timings.numpy()
#true_positive = intersect[intersect > 1.5].size
#false_positive = difference[difference > 0.5].size

print((np.sum(true_positive, axis=0)/(0.00001+np.sum(true_positive, axis=0)+np.sum(false_positive, axis=0))))


for i in range(60):
    print(synapse_type[i])
    #plt.scatter(time, 50*spike_velocity[:, 0]*spike_position[:, 0], s=1)
    plt.scatter(time, 12*spike_timings[:, i], s=1)
    plt.scatter(time, 25*spike_primitive[:, i], s=1)
    #plt.scatter(time, 15*spike_velocity[:, 0], s=1)
    #plt.scatter(time, 7.5*spike_position[:, 0], s=1)
    #print(permutations)


    plt.show()
