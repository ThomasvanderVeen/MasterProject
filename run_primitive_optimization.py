from functions import *
from class_primitive_neuron import LIF_primitive
from dictionaries import Parameters
from plots import *
import time
from tqdm.contrib import itertools

joint_angles_list = pickle_open('Data/joint_angles_list')
velocity_list = pickle_open('Data/velocity_list')
position_list = pickle_open('Data/position_list')
ground_truth_list = pickle_open('Data/ground_truth')

parameters = Parameters(t_total=5, dt=0.001)
N_SIMULATIONS = len(velocity_list)
N_SIMULATIONS = 20
N_LEGS = 6
N_WEIGHTS = 7

tau_list = np.linspace(0e-3, 10e-3, num=5)
w_1 = np.linspace(6e-3, 15e-3, num=N_WEIGHTS)
w_2 = np.linspace(6e-3, 15e-3, num=N_WEIGHTS)
w_1_opt, w_2_opt, accuracy_opt = \
    np.zeros([tau_list.size, N_WEIGHTS]), np.zeros([tau_list.size, N_WEIGHTS]), np.zeros([tau_list.size, N_WEIGHTS])
accuracy_matrix = np.zeros((N_WEIGHTS, N_WEIGHTS, N_WEIGHTS))

for p, l, m in itertools.product(range(tau_list.size), range(N_WEIGHTS), range(N_WEIGHTS)):
    w_pos = [w_1[l]] * N_WEIGHTS
    w_vel = [w_2[m]] * N_WEIGHTS
    permutations_name, synapse_type, weights_primitive, primitive_filter_2, primitive_filter, permutations, _ = get_encoding(
        w_pos, w_vel, 6)
    parameters.primitive['n'] = permutations_name.shape[0] * N_LEGS
    parameters.primitive['w'] = weights_primitive
    parameters.primitive['tau'] = tau_list[p]
    primitive_neuron = define_and_initialize(LIF_primitive, parameters.primitive)
    spike_primitive = np.zeros((parameters.general['N_steps'], parameters.primitive['n']))
    true_positive, false_positive, true_negative, false_negative = \
        [np.empty((N_SIMULATIONS, parameters.primitive['n'])) for _ in range(4)]

    for i in range(N_SIMULATIONS):
        spike_velocity, spike_position, ground_truth = torch.from_numpy(velocity_list[i]), \
                                                       torch.from_numpy(position_list[i]), ground_truth_list[i]

        for j in range(parameters.general['N_steps']):
            pos_vel_spikes = prepare_spikes_primitive(spike_velocity[j, :], spike_position[j, :], permutations,
                                                      primitive_filter)

            _, spike_primitive[j, :] = primitive_neuron.forward(pos_vel_spikes)

        ground_truth_bins = convert_to_bins(ground_truth, 200)
        spike_primitive_bins = convert_to_bins(spike_primitive, 200)

        for j in range(parameters.primitive['n']):
            intersect = spike_primitive_bins[:, j] + ground_truth_bins[:, j]
            difference = spike_primitive_bins[:, j] - ground_truth_bins[:, j]

            true_positive[i, j] = intersect[intersect > 1.5].size
            false_positive[i, j] = difference[difference > 0.5].size
            true_negative[i, j] = intersect[intersect < 0.5].size
            false_negative[i, j] = difference[difference < -0.5].size

    true_pos_sum = np.sum(true_positive, axis=0)
    false_pos_sum = np.sum(false_positive, axis=0)
    true_neg_sum = np.sum(true_negative, axis=0)
    false_neg_sum = np.sum(false_negative, axis=0)
    accuracy, accuracy_types = np.array([]), np.zeros([N_WEIGHTS])
    N_types = np.bincount(synapse_type)

    for i in range(parameters.primitive['n']):
        TPR = true_pos_sum[i] / (true_pos_sum[i] + false_neg_sum[i] + 0.0000001)
        TNR = true_neg_sum[i] / (true_neg_sum[i] + false_pos_sum[i] + 0.0000001)
        PPV = true_pos_sum[i] / (true_pos_sum[i] + false_pos_sum[i] + 0.0000001)
        NPV = true_neg_sum[i] / (true_neg_sum[i] + false_neg_sum[i] + 0.0000001)
        FNR = 1 - TPR
        FPR = 1 - TNR
        FDR = 1 - PPV
        FOR = 1 - NPV

        ACC_balanced = (TPR + TNR) / 2
        F1 = 2 * true_pos_sum[i] / (2 * true_pos_sum[i] + false_pos_sum[i] + false_neg_sum[i] + 0.0000001)
        MCC = np.sqrt(TPR * TNR * PPV * NPV) - np.sqrt(FNR * FPR * FOR * FDR)
        x = np.linspace(0, 1, num=ground_truth_bins[:, i].shape[0])

        # print(synapse_type[i], ACC_balanced)
        # plt.scatter(x, ground_truth_bins[:, i])
        # plt.scatter(x, spike_primitive_bins[:, i] * 2)
        # plt.show()

        accuracy_types[synapse_type[i]] += ACC_balanced / N_types[synapse_type[i]]

        # ACC = (true_pos_sum[i] + true_neg_sum[i])/(true_pos_sum[i] + true_neg_sum[i] + false_pos_sum[i] +
        #                                          false_neg_sum[i] + 0.0000001)
        # accuracy_types[synapse_type[i]] += ACC/N_types[synapse_type[i]]
    accuracy_matrix[l, m, :] = accuracy_types

    for i in range(N_WEIGHTS):
        indexes = np.where(accuracy_matrix[:, :, i] == np.max(accuracy_matrix[:, :, i]))
        w_1_opt[p, i], w_2_opt[p, i] = w_1[indexes[0][0]], w_2[indexes[1][0]]
        accuracy_opt[p, i] = accuracy_matrix[indexes[0][-1], indexes[1][-1], i]

fig, ax = plt.subplots(figsize=(1.5 * 3.54, 3.54), dpi=600)

for i in range(N_WEIGHTS):
    ax.plot(tau_list * 1000, accuracy_opt[:, i], color=parameters.general['colors'][i], linestyle='dotted')
ax.plot(tau_list * 1000, np.mean(accuracy_opt, axis=1), color='black')
for i in range(N_WEIGHTS):
    ax.scatter(tau_list * 1000, accuracy_opt[:, i], color=parameters.general['colors'][i], s=8)
ax.scatter(tau_list * 1000, np.mean(accuracy_opt, axis=1), color='black', s=10)

plot_primitive_accuracy(ax, fig, tau_list)

fig, ax = plt.subplots(2, figsize=(1.5 * 3.54, 3.54), dpi=600)

width = 100 * (tau_list[1] - tau_list[0])
for i in range(tau_list.size):
    for j in range(N_WEIGHTS):
        ax[0].bar(tau_list[i] * 1000 + (-2 * width + width * j), w_1_opt[i, j] * 1000, width, color=colors[j])
        ax[1].bar(tau_list[i] * 1000 + (-2 * width + width * j), w_2_opt[i, j] * 1000, width, color=colors[j])

plot_primitive_weights(ax, fig, tau_list, w_1, w_2)
