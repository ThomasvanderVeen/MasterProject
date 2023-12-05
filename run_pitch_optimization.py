from functions import *
from dictionaries import Parameters
from class_primitive_neuron import LIF_primitive
import plots
import scipy.stats.mstats as zscore
from dtw import dtw

N_SIMULATIONS = 9
N_SIMULATIONS_TEST = 2
N_PLOT = 1
N_SKIP = 100

parameters = Parameters(t_total=25, dt=0.001)
w_1_list = np.linspace(2E-3, 15E-3, num=3)
w_2_list = np.linspace(2E-3, 15E-3, num=2)

data = pickle_open('Data/simulation_data')
primitive_list = pickle_open('Data/primitive_list')
pitch, pitch_max, pitch_min, pitch_middle = get_pitch(data, N_SIMULATIONS+N_SIMULATIONS_TEST, parameters)
accuracy_list = np.zeros((w_2_list.size, w_1_list.size))
time = np.linspace(0, parameters.general['t_total'], num=parameters.general['N_steps'])
spike_posture_list, spike_posture_binned_list = [], []
euclidean_norm = lambda x, y: np.abs(x - y)

for l, m in itertools.product(range(w_2_list.size), range(w_1_list.size)):
    weights = np.zeros((672, 2))

    for j in range(672):
        incidence_binned = np.zeros(2)

        for i in range(N_SIMULATIONS):
            spike_primitive = primitive_list[i]
            spike_train = spike_primitive[:, j]
            incidence = spike_train*pitch[:, i]
            incidence[incidence == 0] = np.nan
            incidence_binned += np.histogram(incidence, bins=2, range=(pitch_min, pitch_max))[0]

        ratio = incidence_binned[1]/incidence_binned[0]

        if ratio > 1.3:
            weights[j, 0] = w_1_list[m]

        if ratio < 1/1.3:
            weights[j, 1] = w_2_list[l]

    parameters.posture['w'] = weights.T
    posture_neuron = define_and_initialize(LIF_primitive, parameters.posture)
    spike_posture = np.empty((parameters.general['N_steps'], N_SIMULATIONS_TEST, 2))
    pitch_binary = np.zeros(pitch.shape)

    for j in range(N_SIMULATIONS_TEST):
        spike_primitive = primitive_list[j+N_SIMULATIONS]
        for i in range(parameters.general['N_steps']):
            _, spike_posture[i, j, :] = posture_neuron.forward(torch.from_numpy(spike_primitive[i, :]))

        firing_rate_1 = get_firing_rate_2(spike_posture[:, j, 0], parameters.general['dt'], t=1, nan_bool=False, sigma=3)
        firing_rate_2 = get_firing_rate_2(spike_posture[:, j, 1], parameters.general['dt'], t=1, nan_bool=False, sigma=3)
        firing_rate = firing_rate_1 - firing_rate_2
        error = np.mean(np.abs(zscore.zscore(firing_rate) - zscore.zscore(pitch[:, j+N_SIMULATIONS])))

        d, cost_matrix, acc_cost_matrix, path = dtw(zscore.zscore(firing_rate)[::N_SKIP], zscore.zscore(pitch[:, j+N_SIMULATIONS])[::N_SKIP],
                                                                dist=euclidean_norm)
        accuracy_list[l, m] += d


    spike_posture_list.append(spike_posture)

accuracy_list[accuracy_list == np.nan] = 0
max_index = np.where(np.ndarray.flatten(accuracy_list) == np.max(accuracy_list))
spike_posture = spike_posture_list[max_index[0][0]]

firing_rate_1 = get_firing_rate_2(spike_posture[:, N_PLOT, 0], parameters.general['dt'], t=1, nan_bool=False, sigma=3)
firing_rate_2 = get_firing_rate_2(spike_posture[:, N_PLOT, 1], parameters.general['dt'], t=1, nan_bool=False, sigma=3)
firing_rate = firing_rate_1 - firing_rate_2

fig, ax = plt.subplots()

ax.plot(time, zscore.zscore(firing_rate))
ax.plot(time, zscore.zscore(pitch[:, N_PLOT + N_SIMULATIONS]))

plots.plot_pitch_estimation(ax, fig)

fig, ax = plt.subplots()

for i in range(w_2_list.size):
    ax.scatter(w_1_list * 1000, accuracy_list[i, :], color=parameters.general['colors'][i], s=13)
    ax.plot(w_1_list*1000, accuracy_list[i, :], label=f'w_inh = {np.round(w_2_list[i]*1000, 2)} mV',
            color=parameters.general['colors'][i])

plots.plot_climbing_accuracy(fig, ax, 'pitch')

#fig, ax = plt.subplots(figsize=(1.5*3.54, 3.54), dpi=600)