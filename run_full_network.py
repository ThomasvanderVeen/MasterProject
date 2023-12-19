import plots
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from class_hair_field import HairField
from class_position_neuron import LIF
from class_primitive_neuron import LIF_primitive
from class_sensory_neuron import AdEx
from class_velocity_neuron import LIF_simple
from dictionaries import Parameters
from functions import *

N_LEGS = 6
N_SIMULATIONS = 2
W_POS = [11.43e-3, 0, 11.43e-3, 11.43e-3, 14e-3, 8e-3, 0e-3]
W_VEL = [0e-3, 17.4e-3, 14e-3, 2.5e-3, 5.71e-3, 0e-3, 14e-3]

permutations_name, synapse_type, weights_primitive, primitive_filter_2, primitive_filter, permutations, base_perm = \
    get_encoding(W_POS, W_VEL, N_LEGS)

data = pickle_open('Data/simulation_data')

joint_angles_list, primitive_list, position_list, velocity_list, sensory_list = [], [], [], [], []

for k in tqdm(range(N_SIMULATIONS), desc='Network progress'):

    joint_angles = np.array(data[f'simulation_{k}'][0]).T

    parameters = Parameters(
        max_joint_angle=np.amax(joint_angles, axis=0),
        min_joint_angle=np.amin(joint_angles, axis=0),
        n_hairs=100,
        t_total=5,
        dt=0.001,
        n_angles=18
    )

    parameters.primitive['w'] = weights_primitive
    parameters.primitive['n'] = permutations_name.shape[0] * N_LEGS

    N_frames = parameters.general['N_frames']
    if joint_angles.shape[0] < N_frames:
        raise Exception(f'Total frames exceeds maximum: {joint_angles.shape[0]} < {N_frames}')

    joint_angles = joint_angles[:N_frames]
    joint_angles = interpolate(joint_angles, parameters.general['t_total'], parameters.general['N_steps'])

    hair_angles = np.zeros((joint_angles.shape[0], 2 * joint_angles.shape[1] * parameters.hair_field['N_hairs']))

    for i in range(parameters.general['n_angles']):
        hair_field = HairField(parameters.hair_field)
        hair_field.reset_max_min(i)
        hair_field.get_double_receptive_field()

        hair_angles[:, i * 2 * parameters.hair_field['N_hairs']: 2 * parameters.hair_field['N_hairs']
                                                                 + i * 2 * parameters.hair_field[
                                                                     'N_hairs']] = hair_field.get_hair_angle(
            joint_angles[:, i]) / 37e9

    neurons = [AdEx, LIF, LIF_simple, LIF_primitive]
    parameters_list = [parameters.sensory, parameters.position, parameters.velocity, parameters.primitive]
    sensory_neuron, position_neuron, velocity_neuron, primitive_neuron = \
        [define_and_initialize(neuron_class, params) for neuron_class, params in zip(neurons, parameters_list)]

    time, spike_sensory = np.array([]), torch.empty(hair_angles.shape)
    spike_position, spike_velocity, spike_primitive = \
        [torch.empty((parameters.general['N_steps'], par['n'])) for par in parameters_list[1:]]

    for i in range(parameters.general['N_steps']):
        time = np.append(time, i * parameters.general['dt'])
        _, spike_sensory[i, :] = sensory_neuron.forward(torch.from_numpy(hair_angles[i, :]))

        reshaped_spikes = torch.reshape(spike_sensory[i, :],
                                        (parameters.velocity['n'], (parameters.hair_field['N_hairs'])))

        _, spike_velocity[i, :] = velocity_neuron.forward(reshaped_spikes)
        _, spike_position[i, :] = position_neuron.forward(
            reshaped_spikes[:, int(parameters.hair_field['N_hairs'] / 2) - 20:])

        pos_vel_spikes = prepare_spikes_primitive(spike_velocity[i, :], spike_position[i, :], permutations,
                                                  primitive_filter)
        _, spike_primitive[i, :] = primitive_neuron.forward(pos_vel_spikes)

    primitive_list.append(spike_primitive.numpy())


    joint_angles_list.append(joint_angles)
    position_list.append(spike_position.numpy())
    velocity_list.append(spike_velocity.numpy())
    sensory_list.append(spike_sensory.numpy())


'''
pickle_save(joint_angles_list, 'Data/joint_angles_list')
pickle_save(sensory_list, 'Data/sensory_list')
pickle_save(position_list, 'Data/position_list')
pickle_save(velocity_list, 'Data/velocity_list')
'''

pickle_save(primitive_list, 'Data/primitive_list')

'''
Position neuron testing
'''

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

handles = ['Dorsal response', 'Ventral response']
for i in range(2):
    firing_rate = get_firing_rate_2(spike_position[:, i].numpy(), parameters.general['dt'], t=0.03)
    ax2.plot(time, firing_rate, color=parameters.general['colors'][i], linestyle=parameters.general['linestyles'][i+1], label = handles[i])



ax1.plot(time, joint_angles[:, 0], color='black', label='Exp. data')
ax1.plot(time, np.full(len(time), np.max(joint_angles[:, 0]) / 2 + np.min(joint_angles[:, 0]) / 2), linestyle='dotted',
         color='black')

plots.plot_position_interneuron(ax1, ax2, fig, 'bi')

fig2, ax3 = plt.subplots()
ax4 = ax3.twinx()

N_hairs, N_half = parameters.hair_field['N_hairs'], parameters.hair_field['N_hairs'] // 2
diff = hair_field.max_list[0] - hair_field.min_list[0]

for i in range(N_half)[2::5]:
    spike_sensory[spike_sensory == 0] = np.nan
    ax3.scatter(time, (-i + N_half) * spike_sensory[:, i + N_half], color=parameters.general['colors'][0], s=1)
    ax3.scatter(time, (1 + i + N_half) * spike_sensory[:, i + N_half + N_hairs], color=parameters.general['colors'][1],
                s=1)

ax4.plot(time, joint_angles[:, 0], color='black')
ax4.plot(time, np.full(time.shape, diff / 2 + hair_field.min_list[0]), linestyle='dotted', color='black')
ax3.set_ylim(1 - N_hairs * .05, N_hairs * 1.05)
ax4.set_ylim(hair_field.min_list[0] - .05 * diff, hair_field.max_list[0] + .05 * diff)

plots.plot_spike_timing(ax3, ax4, fig2, N_hairs)

'''
Velocity neuron testing
'''
spike_velocity = spike_velocity.numpy()

fig, ax = plt.subplots()
ax1 = ax.twinx()

firing_rate_down = get_firing_rate_2(spike_velocity[:, 0], parameters.general['dt'], t=0.05)
firing_rate_up = get_firing_rate_2(spike_velocity[:, 1], parameters.general['dt'], t=0.05)

ax.plot(time[1:], np.diff(joint_angles[:, 0]) / parameters.general['dt'], color='black',
        linestyle=parameters.general['linestyles'][0], label='Exp. data')
ax1.plot(time, firing_rate_down, color=parameters.general['colors'][0],
         linestyle=parameters.general['linestyles'][1], label='Dorsal direction')
ax1.plot(time, firing_rate_up, color=parameters.general['colors'][1],
         linestyle=parameters.general['linestyles'][2], label='Ventral direction')
ax.plot(time, np.full(time.size, 0), color='black', linestyle='dotted')
ax.set_xlim([0, 1])

plots.plot_movement_binary(ax, ax1, fig)

spike_velocity[spike_velocity == 0] = np.nan

fig, ax = plt.subplots()

plt.plot(time, joint_angles[:, 0], color='black')
plt.scatter(time, joint_angles[:, 0] * spike_velocity[:, 0], color=parameters.general['colors'][0],
            marker=parameters.general['markers'][0], s=10)
plt.scatter(time, joint_angles[:, 0] * spike_velocity[:, 1], color=parameters.general['colors'][1],
            marker=parameters.general['markers'][1], s=10)

plots.plot_movement_interneuron_network(ax, fig)

TP, FP, TN, FN = [], [], [], []
for i in range(len(joint_angles_list)):
    spike_velocity, joint_angles = velocity_list[i], joint_angles_list[i]
    joint_velocity = np.diff(joint_angles, axis=0)

    intersect_up = joint_velocity * spike_velocity[1:, 1::2]
    intersect_down = joint_velocity * spike_velocity[1:, 0::2]
    TP.append(intersect_up[intersect_up > 0].size)
    FP.append(intersect_up[intersect_up < 0].size)
    TN.append(intersect_down[intersect_down < 0].size)
    FN.append(intersect_down[intersect_down > 0].size)

TP, FP, TN, FN = sum(TP), sum(FP), sum(TN), sum(FN)
ACC = np.around((TP + TN) / (TP + TN + FP + FN + 0.000001), 3)
TPR = np.around(TP / (TP + FN + 0.000001), 3)
TNR = np.around(TN / (TN + FP + 0.000001), 3)

print(f'[Velocity interneuron] True positive: {TP}, false Positive: {FP}, true negative: {TN}, false negative: {FN}')
print(f'[Velocity interneuron] Accuracy: {ACC}, true positive rate: {TPR}, true negative rate: {TNR}')

table = {'col1': [TP, FP, TN, FN, ACC, TPR, TNR]}
df = pd.DataFrame(data=table, index=['TP', 'FP', 'TN', 'FN', 'ACC', 'TPR', 'TNR'])
df.to_csv("Images/velocity_table.csv")

print(f'[Velocity interneuron] Data written to "velocity_table.csv"')

'''
Primitive neuron testing (ROC plot)
'''

N_joints = joint_angles.shape[1]
N_legs = 6
true_positive, false_positive, true_negative, false_negative = [np.empty((N_SIMULATIONS, parameters.primitive['n'])) for
                                                                _ in range(4)]
ground_truth_list = []
for k in tqdm(range(N_SIMULATIONS), desc='ROC plot progress'):
    joint_angles, spike_primitive = joint_angles_list[k], primitive_list[k]
    ground_truth, ground_vel, ground_pos = [np.zeros([parameters.general['N_steps'], i]) for i in
                                            [parameters.primitive['n'], 36, 36]]

    for i in range(N_joints):
        mid = np.max(joint_angles[:, i]) / 2 + np.min(joint_angles[:, i]) / 2
        diff = np.diff(joint_angles[:, i])
        ground_vel[np.where(diff < 0), 0 + 2 * i] = 1
        ground_vel[np.where(diff > 0), 1 + 2 * i] = 1
        ground_pos[np.where(joint_angles[:, i] < mid), 0 + 2 * i] = 1
        ground_pos[np.where(joint_angles[:, i] > mid), 1 + 2 * i] = 1

    for j in range(parameters.general['N_steps']):
        ground_truth_2 = prepare_spikes_primitive(torch.from_numpy(ground_vel[j, :]),
                                                  torch.from_numpy(ground_pos[j, :]),
                                                  permutations, primitive_filter) + primitive_filter_2
        ground_truth_2 = torch.sum(ground_truth_2, dim=1)
        ground_truth[j, ground_truth_2 > 2.9] = 1
        ground_truth[j, ground_truth_2 < 2.9] = 0

    ground_truth_list.append(ground_truth)

    ground_truth_bins = convert_to_bins(ground_truth, 200)
    spike_primitive_bins = convert_to_bins(spike_primitive, 200)

    for i in range(parameters.primitive['n']):
        intersect = spike_primitive_bins[:, i] + ground_truth_bins[:, i]
        difference = spike_primitive_bins[:, i] - ground_truth_bins[:, i]

        true_positive[k, i] = intersect[intersect > 1.5].size
        false_positive[k, i] = difference[difference > 0.5].size
        true_negative[k, i] = intersect[intersect < 0.5].size
        false_negative[k, i] = difference[difference < -0.5].size

pickle_save(ground_truth_list, 'Data/ground_truth')

true_pos_sum = np.sum(true_positive, axis=0)
false_pos_sum = np.sum(false_positive, axis=0)
true_neg_sum = np.sum(true_negative, axis=0)
false_neg_sum = np.sum(false_negative, axis=0)
N_types = np.bincount(synapse_type)
accuracy, accuracy_types = np.array([]), np.zeros([N_types.size])

fig, ax = plt.subplots()

LEGEND_LABELS = ['pos-pos', 'vel-vel', 'pos-vel', 'pos-pos-vel', 'vel-vel-pos', 'pos-pos-pos', 'vel-vel-vel']

for i in range(parameters.primitive['n']):
    TPR = true_pos_sum[i] / (true_pos_sum[i] + false_neg_sum[i] + 0.0000001)
    TNR = true_neg_sum[i] / (true_neg_sum[i] + false_pos_sum[i] + 0.0000001)
    plt.scatter(false_pos_sum[i] / (false_pos_sum[i] + true_neg_sum[i] + 0.0000001),
                true_pos_sum[i] / (true_pos_sum[i] + false_neg_sum[i] + 0.0000001),
                color=parameters.general['colors'][synapse_type[i]], s=8,
                marker=parameters.general['markers'][synapse_type[i]], zorder=2)
    ACC_balanced = (TPR + TNR) / 2
    accuracy = np.append(accuracy, ACC_balanced)
    accuracy_types[synapse_type[i]] += ACC_balanced / N_types[synapse_type[i]]

for i in range(len(LEGEND_LABELS)):
    plt.scatter(100, 100, color=parameters.general['colors'][i], s=13,
                marker=parameters.general['markers'][i], label=LEGEND_LABELS[i])

accuracy_mean = np.mean(accuracy)

print(f'[Primitive neuron] Mean accuracy of primitive neurons: {accuracy_mean}')
print(
    f'[Primitive neuron] and of type pos-pos: {np.around(accuracy_types[0], 3)}, vel-vel: {np.around(accuracy_types[1], 3)}, pos-vel: '
    f'{np.around(accuracy_types[2], 3)}, pos-pos-vel: {np.around(accuracy_types[3], 3)}, vel-vel-pos: {np.around(accuracy_types[4], 3)}, '
    f'pos-pos-pos: {np.around(accuracy_types[5], 3)}, vel-vel-vel: {np.around(accuracy_types[6], 3)}')

plt.plot([0, 1], [0, 1], color='black', linestyle='dotted')
plots.plot_primitive_roc(ax, fig)

'''
Primitive neuron testing (PSTH plot)
'''

position_angles = ['alpha-', 'alpha+', 'beta-', 'beta+', 'gamma-', 'gamma+']
stance, swing = np.empty((6, permutations_name.shape[0])), np.empty((6, permutations_name.shape[0]))

for m in tqdm(range(6), desc='PSTH plot progress'):
    swing_bin_rate, stance_bin_rate, swing_bin_likelihood, stance_bin_likelihood = [
        np.empty((N_SIMULATIONS, permutations_name.shape[0], i)) for
        i in [15, 15, 15, 15]]
    swing_bin_likelihood_vel = np.empty((N_SIMULATIONS, 6, 15))
    stance_bin_likelihood_vel = np.empty((N_SIMULATIONS, 6, 15))

    for k in range(N_SIMULATIONS):
        spike_primitive = primitive_list[k]
        spike_position = position_list[k]
        spike_velocity = velocity_list[k]
        gait = np.array(data[f'simulation_{k}'][1])[m, :]
        gait = gait[:parameters.general['N_frames']]
        gait = interpolate(gait, parameters.general['t_total'], parameters.general['N_steps'], True)
        for i in range(permutations_name.shape[0]):
            swing_bin_rate[k, i, :], stance_bin_rate[k, i, :], swing_bin_likelihood[k, i, :], stance_bin_likelihood[k,
                                                                                              i, :] = \
                get_stance_swing_bins(gait, spike_primitive[:, i + permutations_name.shape[0] * m])
        for i in range(N_LEGS):
            _, _, swing_bin_likelihood_vel[k, i, :], stance_bin_likelihood_vel[k, i, :] = \
                get_stance_swing_bins(gait, spike_velocity[:, i + 6 * m])

    swing_bin_likelihood, stance_bin_likelihood = np.mean(swing_bin_likelihood, axis=0), np.mean(stance_bin_likelihood,
                                                                                                 axis=0)
    swing_bin_likelihood_vel, stance_bin_likelihood_vel = np.mean(swing_bin_likelihood_vel, axis=0), np.mean(
        stance_bin_likelihood_vel, axis=0)

    swing_sum, stance_sum = np.mean(swing_bin_likelihood, axis=1), np.mean(stance_bin_likelihood, axis=1)

    swing[m, :] = swing_sum / (stance_sum + swing_sum)
    stance[m, :] = stance_sum / (stance_sum + swing_sum)

    fig, ax = plt.subplots()

    for i in range(permutations_name.shape[0]):
        ax.scatter(np.linspace(0, .725, num=15), swing_bin_likelihood[i, :]*100, color=parameters.general['colors'][0],
                   marker='^')
        ax.scatter(np.linspace(.775, 1.5, num=15), stance_bin_likelihood[i, :]*100, color=parameters.general['colors'][1],
                   marker='^')

        plots.plot_psth(ax, fig, i, m, permutations_name[i], 'primitive')

    for i in range(N_LEGS):
        ax.scatter(np.linspace(0, .725, num=15), swing_bin_likelihood_vel[i, :]*100, color=parameters.general['colors'][0],
                   marker='^')
        ax.scatter(np.linspace(.775, 1.5, num=15), stance_bin_likelihood_vel[i, :]*100,
                   color=parameters.general['colors'][1], marker='^')

        plots.plot_psth(ax, fig, i, m, position_angles[i], 'velocity')

'''
Swing Stance
'''


n_primitive = permutations_name.shape[0]
n_primitive_2 = n_primitive // 2

legs = ['R1', 'R2', 'R3', 'L1', 'L2', 'L3']
x = np.linspace(n_primitive_2, n_primitive_2 + (len(legs) - 1) * n_primitive, num=len(legs))
swing_flat = np.ndarray.flatten(swing)

plt.close('all')

fig, ax = plt.subplots(3)

for j in range(3):
    for i in range(parameters.primitive['n']):
        ax[j].scatter(i, swing_flat[i], color=parameters.general['colors'][int(base_perm[i % base_perm.shape[0], j])], s=4)
        if i % permutations_name.shape[0] == 0:
            ax[j].plot([i, i], [0, 1], color='black', linestyle='dotted')
    ax[j].plot([parameters.primitive['n'], parameters.primitive['n']], [0, 1], color='black', linestyle='dotted')
    #ax[j].plot([0, parameters.primitive['n']], [0.9, 0.9], color='red', linestyle='dotted')
    #ax[j].plot([0, parameters.primitive['n']], [0.1, 0.1], color='red', linestyle='dotted')

plots.plot_swing_stance(ax, fig, x, legs)

indexes_swing = np.where(swing_flat > 0.9)
indexes_stance = np.where(swing_flat < 0.1)

indexes_swing, leg_swing = get_indexes_legs(indexes_swing[0])
indexes_stance, leg_stance = get_indexes_legs(indexes_stance[0])

print('swing')
for i in range(len(leg_swing)):
    print(indexes_swing[i], leg_swing[i], legs[leg_swing[i]], permutations_name[indexes_swing[i]])

print('stance')
for i in range(len(leg_stance)):
    print(indexes_stance[i], leg_stance[i], legs[leg_stance[i]], permutations_name[indexes_stance[i]])

'''
Swing Stance Comparison
'''

swings = np.zeros((3, 4, 6, 24))

for i, j, k in np.ndindex((6, 3, 112)):
    perm_type = base_perm[k, j] - 1
    if perm_type >= 0:
        index_to_update = np.argmax(swings[j, perm_type, i, :] == 0)
        swings[j, perm_type, i, index_to_update] = swing[i, k]

swings_average = np.mean(swings, axis=3)
swings_max = np.max(swings, axis=3) - swings_average
swings_min = swings_average - np.min(swings, axis=3)
swings_std = np.std(swings, axis=3)

x = [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15]
LEGS = ['R1', 'R2', 'R3', 'L1', 'L2', 'L3']
Y_AXIS_LIMITS = [-0.1, 1.1]
X_AXIS_LIMITS = [-0.3, 2.9]
Y_TICKS = [0, 0.5, 1]
X_TICKS = [.3, 1.3, 2.3]
X_LABELS = ['α', 'β', 'γ']
LEGEND_LABELS = ['v-', 'v+', 'p-', 'p+']

fig, axes = plt.subplots(2, 3)

for k in range(6):
    ax = axes[k // 3, k % 3]
    ax.set_ylim(Y_AXIS_LIMITS)
    ax.set_xlim(X_AXIS_LIMITS)

    for y_val in [0, 0.5, 1]:
        ax.plot([-2, 5], [y_val, y_val], linestyle='dotted', color='black', zorder=0)

    for j in range(4):
        ax.errorbar(np.array([0, 1, 2]) + 0.2 * j, swings_average[:, j, k],
                    yerr=(swings_min[:, j, k], swings_max[:, j, k]), capsize=3, fmt='None', color=parameters.general['colors'][j + 1])

        for i in range(3):
            ax.add_patch(Rectangle((i + 0.2 * j - 0.08, swings_average[i, j, k] - swings_std[i, j, k]), 0.16,
                                   2 * swings_std[i, j, k], facecolor=parameters.general['colors'][j + 1]))
            ax.add_patch(
                Rectangle((i + 0.2 * j - 0.08, swings_average[i, j, k] - 0.0075), 0.16, 0.015, facecolor='black',
                          zorder=10))

    if k % 3 != 0:
        ax.set_yticks([])
    else:
        ax.set_yticks(Y_TICKS)

    if k // 3 == 1:
        ax.set_xticks(X_TICKS)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[:] = X_LABELS
        ax.set_xticklabels(labels)
    else:
        ax.set_xticks([])
    ax.set_title(LEGS[k])

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=LEGEND_LABELS[i], markerfacecolor=parameters.general['colors'][i + 1], markersize=7)
    for i in range(len(LEGEND_LABELS))]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.09))

fig.tight_layout(pad=0.5)
fig.savefig('Images/swing_stance_comparison.png', bbox_inches='tight')


