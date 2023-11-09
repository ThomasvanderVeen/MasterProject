from functions import *
from class_primitive_neuron import LIF_primitive
from dictionaries import Parameters
from plots import *

velocity_list = pickle_open('Data/velocity_list')
position_list = pickle_open('Data/position_list')
ground_truth_list = pickle_open('Data/ground_truth')

parameters = Parameters(t_total=6, dt=0.001)
N_simulations = len(velocity_list)
permutations = get_primitive_indexes(6)

n_w = 2
w_1 = np.linspace(4e-3, 15e-3, num=n_w)
w_2 = np.linspace(4e-3, 15e-3, num=n_w)

accuracy_matrix = np.zeros((n_w, n_w, 6))

for l in range(n_w):
    for m in range(n_w):
        w_pos = [w_1[l]]*5
        w_vel = [w_2[m]]*5
        _, synapse_type, weights_primitive, primitive_filter_2, primitive_filter = get_encoding(w_pos, w_vel)

        primitive_neuron = define_and_initialize(LIF_primitive, parameters.primitive)
        parameters.primitive['w'] = weights_primitive
        spike_primitive = np.zeros((parameters.general['N_steps'], 360))
        true_positive, false_positive, true_negative, false_negative = [np.empty((N_simulations, 360)) for _ in range(4)]

        for i in range(N_simulations):
            spike_velocity, spike_position, ground_truth = torch.from_numpy(velocity_list[i]), torch.from_numpy(position_list[i]), ground_truth_list[i]

            for j in range(parameters.general['N_steps']):
                pos_vel_spikes = prepare_spikes_primitive(spike_velocity[j, :], spike_position[j, :], permutations,
                                                          primitive_filter)

                _, spike_primitive[j, :] = primitive_neuron.forward(pos_vel_spikes)

            ground_truth_bins = convert_to_bins(ground_truth, 100)
            spike_primitive_bins = convert_to_bins(spike_primitive, 100)

            for j in range(360):
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
        accuracy, accuracy_types = np.array([]), np.zeros([5])
        N_types = np.bincount(synapse_type)

        for i in range(360):
            ACC = (true_pos_sum[i] + true_neg_sum[i])/(true_pos_sum[i] + true_neg_sum[i] + false_pos_sum[i] + false_neg_sum[i] + 0.0000001)
            accuracy = np.append(accuracy, ACC)
            accuracy_types[synapse_type[i]] += ACC/N_types[synapse_type[i]]

        accuracy_mean = np.mean(accuracy)
        accuracy_matrix[m, l, :-1] = accuracy_types
        accuracy_matrix[m, l, -1] = accuracy_mean

        print(f'[Primitive neuron] Mean accuracy of primitive neurons: {accuracy_mean}, w_1 = {w_1[m]}, w_1 = {w_2[l]}.')
        #print(f'[Primitive neuron] and of type vel-pos: {np.around(accuracy_types[0], 3)}, vel-vel: {np.around(accuracy_types[1], 3)}, pos-pos: '
        #      f'{np.around(accuracy_types[2], 3)}, pos-vel-vel: {np.around(accuracy_types[3], 3)}, vel-pos-pos: {np.around(accuracy_types[4], 3)}.')

v_max = np.max(accuracy_matrix)
v_min = np.min(accuracy_matrix)

fig, ax = plt.subplots(2, 3)

df1 = pd.DataFrame(accuracy_matrix[:, :, 0], columns=np.around(w_1, 4), index=np.around(w_2, 4))
df2 = pd.DataFrame(accuracy_matrix[:, :, 1], columns=np.around(w_1, 4), index=np.around(w_2, 4))
df3 = pd.DataFrame(accuracy_matrix[:, :, 2], columns=np.around(w_1, 4), index=np.around(w_2, 4))
df4 = pd.DataFrame(accuracy_matrix[:, :, 3], columns=np.around(w_1, 4), index=np.around(w_2, 4))
df5 = pd.DataFrame(accuracy_matrix[:, :, 4], columns=np.around(w_1, 4), index=np.around(w_2, 4))
df6 = pd.DataFrame(accuracy_matrix[:, :, 5], columns=np.around(w_1, 4), index=np.around(w_2, 4))

sns.heatmap(data=df1, cbar=False, annot=True, ax=ax[0, 0], xticklabels=False, vmax=v_max, vmin=v_min)
sns.heatmap(data=df2, cbar=False, annot=True, ax=ax[0, 1], yticklabels=False, xticklabels=False, vmax=v_max, vmin=v_min)
sns.heatmap(data=df3, cbar=False, annot=True, ax=ax[1, 0], vmax=v_max, vmin=v_min)
sns.heatmap(data=df4, cbar=False, annot=True, ax=ax[1, 1], yticklabels=False, vmax=v_max, vmin=v_min)
sns.heatmap(data=df5, cbar=False, annot=True, ax=ax[0, 2], yticklabels=False, xticklabels=False, vmax=v_max, vmin=v_min)
sns.heatmap(data=df6, cbar=False, annot=True, ax=ax[1, 2], yticklabels=False, vmax=v_max, vmin=v_min)

fig.text(0.5, 0.04, 'common X', ha='center')
fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')

plt.show()