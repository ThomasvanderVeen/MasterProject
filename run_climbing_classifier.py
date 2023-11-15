from functions import *
from dictionaries import Parameters
from class_primitive_neuron import LIF_primitive
from scipy.signal import find_peaks

data = pickle_open('Data/simulation_data')
primitive_list = pickle_open('Data/primitive_list')
parameters = Parameters(t_total=20, dt=0.001)
N_simulations = 8
N_simulations_test = 2
N_plot = 1

pitch = np.empty((parameters.general['N_steps'], N_simulations + N_simulations_test))
for i in range(N_simulations + N_simulations_test):
    pitch_single = np.array(data[f'simulation_{i}'][2][1, :])
    pitch_single = pitch_single[:parameters.general['N_frames']]
    pitch[:, i] = interpolate(pitch_single, parameters.general['t_total'], parameters.general['N_steps'])

pitch_max, pitch_min = np.max(pitch), np.min(pitch)
pitch_middle = pitch_max/2 + pitch_min/2

N_bins = 2
N_spikes = np.zeros(360)
weights = np.zeros(360)

w_1_list = np.linspace(0.9E-3, 1.4E-3, num=12)
w_2_list = np.linspace(0E-3, -1E-3, num=3)
accuracy_list = np.zeros((w_2_list.size, w_1_list.size))

for l in tqdm(range(w_2_list.size)):
    for m in range(w_1_list.size):
        weights = np.zeros(360)
        for j in range(360):
            bin, bin_2 = np.zeros(N_bins), np.zeros(60)
            bin_pitch = np.zeros(N_bins)
            for i in range(N_simulations):
                spike_primitive = primitive_list[i]
                spike_train = spike_primitive[:, j]
                incidence = spike_train*pitch[:, i]
                incidence[incidence == 0] = np.nan
                bin += np.histogram(incidence, bins=N_bins, range=(pitch_min, pitch_max))[0]
                bin_2 += np.histogram(incidence, bins=60, range=(pitch_min, pitch_max))[0]
                bin_pitch += np.histogram(pitch[:, i], bins=N_bins, range=(pitch_min, pitch_max))[0]

            ratio = bin[1]/bin[0]

            if ratio > 1.5:
                weights[j] = w_1_list[m]

            if ratio < 1/1.5:
                weights[j] = w_2_list[l]

        parameters.posture['w'] = weights.T
        posture_neuron = define_and_initialize(LIF_primitive, parameters.posture)
        spike_posture, time = np.empty((parameters.general['N_steps'], N_simulations_test)), np.zeros(parameters.general['N_steps'])

        for j in range(N_simulations_test):
            spike_primitive = primitive_list[j+N_simulations]
            for i in range(parameters.general['N_steps']):
                time[i] = i * parameters.general['dt']
                _, spike_posture[i, j] = posture_neuron.forward(torch.from_numpy(spike_primitive[i, :]))

        n_bins = 50

        pitch_binary = np.zeros(pitch.shape)
        pitch_binary[pitch > pitch_middle] = 1

        pitch_binned = convert_to_bins(pitch_binary, n_bins)

        spike_posture_binned = convert_to_bins(spike_posture, n_bins)

        intersection = pitch_binned[:, N_simulations:] + spike_posture_binned
        difference = pitch_binned[:, N_simulations:] - spike_posture_binned

        true_positive = intersection[intersection > 1.5].size
        true_negative = intersection[intersection < 0.5].size
        false_positive = difference[difference < -0.5].size
        false_negative = difference[difference > 0.5].size

        TPR = true_positive/(true_positive+false_negative)
        FPR = false_positive/(false_positive+true_negative)
        ACC = (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)

        accuracy_list[l, m] = ACC

print(accuracy_list)

colors = ['#c1272d', '#0000a7', '#eecc16', '#008176', '#b3b3b3']

for i in range(w_2_list.size):
    plt.plot(w_1_list*1000, accuracy_list[i, :], label=f'w_2 = {w_2_list[i]*1000}', color=colors[i])

plt.xlabel('w_1 (mV)')
plt.ylabel('Accuracy')
plt.plot()
plt.legend()

print(f'TPR: {TPR}, FPR: {FPR}, ACC: {ACC}')

fig, ax = plt.subplots()

ax.plot(time, pitch[:, N_simulations+N_plot], color='black', label='body pitch')
ax.plot([time[0], time[-1]], [pitch_middle, pitch_middle], linestyle='dotted', color='red')

spike_posture[spike_posture == 0] = np.nan
x = np.linspace(0, 20, num=n_bins)
x_dist = x[1]-x[0]

ax.scatter(time, pitch[:, N_simulations+N_plot]*spike_posture[:, N_plot], color='blue', marker='^')
for i in range(n_bins):
    if spike_posture_binned[i, N_plot] == 1:
        ax.fill_between([x[i]-x_dist/2, x[i]+x_dist/2], -1000, 1000, alpha=0.3, color='grey')
ax.set_ylim([-20, 75])
ax.set_xlim([0, 20])
ax.legend(['body pitch', 'divide', 'spikes', 'climbing'])
ax.set_xlabel("Time [s]")
ax.set_ylabel("Body Pitch [°]")

plt.show()

#spike_primitive[spike_primitive == 0] = np.nan



