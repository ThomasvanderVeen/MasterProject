import scipy.stats.mstats as zscore
from dtw import dtw

import plots
from class_primitive_neuron import LIF_primitive
from dictionaries import Parameters
from functions import *

N_SIMULATIONS = 9
N_SIMULATIONS_TEST = 2
N_PLOT = 2
N_SKIP = 100 #Try with N_SKIP = 1

w_1_list = np.linspace(7.2E-3, 7.2E-3, num=1)
ratios = np.linspace(3.06, 3.06, num=1)
parameters = Parameters(t_total=25, dt=0.001)

print(w_1_list)

data = pickle_open('Data/simulation_data')
primitive_list = pickle_open('Data/primitive_list')
pitch, pitch_max, pitch_min, pitch_middle = get_pitch(data, N_SIMULATIONS + N_SIMULATIONS_TEST, parameters)
accuracy_list = np.zeros((ratios.size, w_1_list.size))
time = np.linspace(0, parameters.general['t_total'], num=parameters.general['N_steps'])
spike_posture_list, spike_posture_binned_list = [], []
euclidean_norm = lambda x, y: np.abs(x - y)
perm, synapse_type, _, _, _, _, _ = get_encoding()

d_noises = np.zeros(5)
N = ratios.size * w_1_list.size * N_SIMULATIONS_TEST

for l, m in itertools.product(range(ratios.size), range(w_1_list.size)):
    weights = np.zeros((672, 2))

    for j in range(672):
        incidence_binned = np.zeros(2)

        for i in range(N_SIMULATIONS):
            spike_primitive = primitive_list[i]
            spike_train = spike_primitive[:, j]
            incidence = spike_train * pitch[:, i]
            incidence[incidence == 0] = np.nan
            incidence_binned += np.histogram(incidence, bins=2, range=(-180, 200))[0]

        ratio = incidence_binned[1] / incidence_binned[0]

        if ratio > ratios[l]:
            weights[j, 0] = w_1_list[m]

        if ratio < 1 / ratios[l]:
            weights[j, 1] = w_1_list[m]

    parameters.posture['w'] = weights.T
    posture_neuron = define_and_initialize(LIF_primitive, parameters.posture)
    spike_posture = np.empty((parameters.general['N_steps'], N_SIMULATIONS_TEST, 2))
    pitch_binary = np.zeros(pitch.shape)

    for j in range(N_SIMULATIONS_TEST):
        spike_primitive = primitive_list[j + N_SIMULATIONS]
        for i in range(parameters.general['N_steps']):
            _, spike_posture[i, j, :] = posture_neuron.forward(torch.from_numpy(spike_primitive[i, :]))

        firing_rate_1 = get_firing_rate_2(spike_posture[:, j, 0], parameters.general['dt'], t=0.2, nan_bool=False,
                                          sigma=3)
        firing_rate_2 = get_firing_rate_2(spike_posture[:, j, 1], parameters.general['dt'], t=0.2, nan_bool=False,
                                          sigma=3)
        firing_rate = firing_rate_1 - firing_rate_2
        error = np.mean(np.abs(
            np.convolve(zscore.zscore(firing_rate), np.ones(3000) / 3000, mode='same') - zscore.zscore(
                pitch[:, j + N_SIMULATIONS])))

        d, _, _, _ = dtw(np.convolve(zscore.zscore(firing_rate), np.ones(3000) / 3000, mode='same')[::N_SKIP], zscore.zscore(pitch[:, j + N_SIMULATIONS])[::N_SKIP],
                         dist=euclidean_norm)
        accuracy_list[l, m] += d / N_SIMULATIONS_TEST




        d, _, _, _ = dtw(zscore.zscore(pitch[:, j + N_SIMULATIONS])[::N_SKIP]+np.random.normal(0, 0.27, size=250), zscore.zscore(pitch[:, j + N_SIMULATIONS])[::N_SKIP],
                         dist=euclidean_norm)
        d_noises[0] += d

        d, _, _, _ = dtw(zscore.zscore(pitch[:, j + N_SIMULATIONS])[::N_SKIP]+np.random.normal(0, 0.29, size=250), zscore.zscore(pitch[:, j + N_SIMULATIONS])[::N_SKIP],
                         dist=euclidean_norm)
        d_noises[1] += d
        d, _, _, _ = dtw(zscore.zscore(pitch[:, j + N_SIMULATIONS])[::N_SKIP]+np.random.normal(0, 0.31, size=250), zscore.zscore(pitch[:, j + N_SIMULATIONS])[::N_SKIP],
                         dist=euclidean_norm)
        d_noises[2] += d
        d, _, _, _ = dtw(zscore.zscore(pitch[:, j + N_SIMULATIONS])[::N_SKIP]+np.random.normal(0, 0.33, size=250), zscore.zscore(pitch[:, j + N_SIMULATIONS])[::N_SKIP],
                         dist=euclidean_norm)
        d_noises[3] += d
        d, _, _, _ = dtw(zscore.zscore(pitch[:, j + N_SIMULATIONS])[::N_SKIP]+np.random.normal(0, 0.35, size=250), zscore.zscore(pitch[:, j + N_SIMULATIONS])[::N_SKIP],
                         dist=euclidean_norm)
        d_noises[4] += d



        # plt.plot(time[::N_SKIP], zscore.zscore(pitch[:, j + N_SIMULATIONS])[::N_SKIP]+np.random.normal(0, 1, size=pitch[:, j + N_SIMULATIONS][::N_SKIP].shape))
        # plt.plot(time[::N_SKIP], zscore.zscore(pitch[:, j + N_SIMULATIONS])[::N_SKIP])
        # plt.show()

    spike_posture_list.append(spike_posture)

indices_up = np.where(parameters.posture['w'][0, :] > 0)
indices_down = np.where(parameters.posture['w'][1, :] > 0)

neuron_indices_up, legs_up = get_indexes_legs(indices_up[0])
neuron_indices_down, legs_down = get_indexes_legs(indices_down[0])

print(perm[neuron_indices_up])

print(np.bincount(legs_up))
print(np.bincount(legs_down))

print(d_noises/N)

#print(np.around(100 * np.bincount(np.array(synapse_type)[neuron_indices_up]) / np.bincount(np.array(synapse_type)), 3))
#print(
#    np.around(100 * np.bincount(np.array(synapse_type)[neuron_indices_down]) / np.bincount(np.array(synapse_type)), 3))

"""
print('neurons up')
for i in range(neuron_indices_up.size):
    print(neuron_indices_up[i], legs_up[i], perm[neuron_indices_up[i]])

print('neurons down')
for i in range(neuron_indices_down.size):
    print(neuron_indices_down[i], legs_down[i], perm[neuron_indices_down[i]])
"""

accuracy_list[accuracy_list == np.nan] = 0
min_index = np.where(np.ndarray.flatten(accuracy_list) == np.min(accuracy_list))

spike_posture = spike_posture_list[min_index[0][0]]


fig, ax = plt.subplots()
for i in range(N_PLOT):
    firing_rate_1 = get_firing_rate_2(spike_posture[:, i, 0], parameters.general['dt'], t=0.2, nan_bool=False, sigma=3)
    firing_rate_2 = get_firing_rate_2(spike_posture[:, i, 1], parameters.general['dt'], t=0.2, nan_bool=False, sigma=3)
    firing_rate = firing_rate_1 - firing_rate_2

    ax.plot(time + i * time[-1], zscore.zscore(pitch[:, i + N_SIMULATIONS]), color=parameters.general['colors'][0],
            linewidth=2)
    ax.plot(time + i * time[-1], zscore.zscore(firing_rate), color='black', linewidth=1)
    ax.plot(time + i * time[-1], np.convolve(zscore.zscore(firing_rate), np.ones(3000) / 3000, mode='same'),
            color=parameters.general['colors'][1], linewidth=2)

plots.plot_pitch_estimation(ax, fig)

fig, ax = plt.subplots()

for i in range(ratios.size):
    # ax.scatter(w_1_list * 1000, accuracy_list[i, :], color=parameters.general['colors'][i], s=13)
    ax.plot(w_1_list * 1000, accuracy_list[i, :], label=f'ratio = {ratios[i]}',
            color=parameters.general['colors'][i],
            marker=parameters.general['markers'][i],
            linestyle=parameters.general['linestyles'][i])

plots.plot_climbing_accuracy(fig, ax, 'pitch')

# fig, ax = plt.subplots(figsize=(1.5*3.54, 3.54), dpi=600)
