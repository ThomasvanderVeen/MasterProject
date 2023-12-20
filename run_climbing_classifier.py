from class_primitive_neuron import LIF_primitive
from dictionaries import Parameters
from plots import *

parameters = Parameters(t_total=5, dt=0.001)
N_simulations = 2
N_simulations_test = 2
N_plot = 1
n_bins = 50
w_1_list = np.linspace(0.25E-4, 1E-4, num=3)
w_2_list = np.linspace(0E-3, -1E-4, num=2)

data = pickle_open('Data/simulation_data')
primitive_list = pickle_open('Data/primitive_list')
pitch, pitch_max, pitch_min, pitch_middle = get_pitch(data, N_simulations + N_simulations_test, parameters)
accuracy_list = np.zeros((w_2_list.size, w_1_list.size))
time = np.linspace(0, parameters.general['t_total'], num=parameters.general['N_steps'])
spike_posture_list, spike_posture_binned_list = [], []

for l, m in itertools.product(range(w_2_list.size), range(w_1_list.size)):
    weights = np.zeros(672)

    for j in range(672):
        incidence_binned = np.zeros(2)

        for i in range(N_simulations):
            spike_primitive = primitive_list[i]
            spike_train = spike_primitive[:, j]
            incidence = spike_train * pitch[:, i]
            incidence[incidence == 0] = np.nan
            incidence_binned += np.histogram(incidence, bins=2, range=(pitch_min, pitch_max))[0]

        ratio = incidence_binned[1] / incidence_binned[0]

        if ratio > 2:
            weights[j] = w_1_list[m]

        if ratio < 1 / 2:
            weights[j] = w_2_list[l]

    parameters.posture['w'] = weights.T
    parameters.posture['n'] = 1
    posture_neuron = define_and_initialize(LIF_primitive, parameters.posture)
    spike_posture = np.empty((parameters.general['N_steps'], N_simulations_test))
    pitch_binary = np.zeros(pitch.shape)

    for j in range(N_simulations_test):
        spike_primitive = primitive_list[j + N_simulations]
        for i in range(parameters.general['N_steps']):
            _, spike_posture[i, j] = posture_neuron.forward(torch.from_numpy(spike_primitive[i, :]))

    pitch_binary[pitch > pitch_middle] = 1
    pitch_binned = convert_to_bins(pitch_binary, n_bins)
    spike_posture_binned = convert_to_bins(spike_posture, n_bins)
    spike_posture_binned_list.append(spike_posture_binned), spike_posture_list.append(spike_posture)

    intersection = pitch_binned[:, N_simulations:] + spike_posture_binned
    difference = pitch_binned[:, N_simulations:] - spike_posture_binned

    true_positive = intersection[intersection > 1.5].size
    true_negative = intersection[intersection < 0.5].size
    false_positive = difference[difference < -0.5].size
    false_negative = difference[difference > 0.5].size

    MCC = matthews_correlation(true_positive, true_negative, false_positive, false_negative)

    TPR = true_positive / (true_positive + false_negative)
    FPR = false_positive / (false_positive + true_negative)
    TNR = true_negative / (true_negative + false_positive)

    ACC = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    ACC_balanced = (TPR + TNR) / 2

    accuracy_list[l, m] = MCC

max_index = np.where(np.ndarray.flatten(accuracy_list) == np.max(accuracy_list))
spike_posture, spike_posture_binned = spike_posture_list[max_index[0][0]], spike_posture_binned_list[max_index[0][0]]

fig, ax = plt.subplots(figsize=(1.5 * 3.54, 3.54), dpi=600)

for i in range(w_2_list.size):
    ax.plot(w_1_list * 1000, accuracy_list[i, :], label=f'w_inh = {np.round(w_2_list[i] * 1000, 2)} mV',
            color=parameters.general['colors'][i],
            marker=parameters.general['markers'][i],
            linestyle=parameters.general['linestyles'][i])

plot_climbing_accuracy(fig, ax, 'climbing')

fig, ax = plt.subplots(figsize=(1.5 * 3.54, 3.54), dpi=600)

ax.plot(time, pitch[:, N_simulations + N_plot], color='black', label='body pitch')
ax.plot([time[0], time[-1]], [pitch_middle, pitch_middle], linestyle='dotted', color='red')

spike_posture[spike_posture == 0] = np.nan
x = np.linspace(0, 20, num=n_bins)
x_dist = x[1] - x[0]

ax.scatter(time, pitch[:, N_simulations + N_plot] * spike_posture[:, N_plot], color='blue', marker='^')
for i in range(n_bins):
    if spike_posture_binned[i, N_plot] == 1:
        ax.fill_between([x[i] - x_dist / 2, x[i] + x_dist / 2], -1000, 1000, alpha=0.3, color='grey')

plot_climbing_classifier(fig, ax)
