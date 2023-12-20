import scipy.stats.mstats as zscore
from dtw import dtw

from class_position_neuron import LIF
from dictionaries import Parameters
from plots import *

position_list = pickle_open('Data/position_list')
sensory_list = pickle_open('Data/sensory_list')
joint_angles_list = pickle_open('Data/joint_angles_list')


N_GRID = 4
NOISES = [0.01, 0.05, 0.10]
N_PLOT = 5
N_ANGLES = 18
N_SIMULATIONS = 2

parameters = Parameters(t_total=5, dt=0.001, n_hairs=200)
x = np.linspace(0, parameters.general['t_total'], num=parameters.general['N_steps'])
d, d_noise = np.empty((N_ANGLES, N_SIMULATIONS, N_GRID, N_GRID)), np.empty((N_ANGLES, N_SIMULATIONS, 3))
b_list = np.linspace(1e-3, 16e-3, num=N_GRID)
tau_w_list = np.linspace(3e-3, 18e-3, num=N_GRID)
euclidean_norm = lambda x, y: np.abs(x - y)

for j, l, m in itertools.product(range(N_SIMULATIONS), range(b_list.size), range(tau_w_list.size)):
    joint_angles = joint_angles_list[j][:parameters.general['N_steps']]
    spike_sensory = sensory_list[j]
    spike_position = np.zeros((parameters.general['N_steps'], 36))
    position_neuron = define_and_initialize(LIF, parameters.position)

    parameters.position['b'] = b_list[l]
    parameters.position['tau_W'] = tau_w_list[m]
    position_neuron = define_and_initialize(LIF, parameters.position)

    for i in range(parameters.general['N_steps']):
        reshaped_spikes = torch.reshape(torch.from_numpy(spike_sensory[i, :]),
                                        (parameters.velocity['n'], (parameters.hair_field['N_hairs'])))

        _, spike_position[i, :] = position_neuron.forward(
            reshaped_spikes[:, int(reshaped_spikes.shape[1] / 2) - 20:])

    for i in range(N_ANGLES):
        firing_rate_1 = get_firing_rate_2(spike_position[:, 2 * i], parameters.general['dt'], 0.05, nan_bool=False)
        firing_rate_2 = get_firing_rate_2(spike_position[:, 2 * i + 1], parameters.general['dt'], 0.05, nan_bool=False)

        combined_firing_rate = firing_rate_2 - firing_rate_1
        combined_firing_rate = zscore.zscore(combined_firing_rate)

        joint_angle = zscore.zscore(joint_angles[:, i])

        d[i, j, l, m], cost_matrix, acc_cost_matrix, path = dtw(combined_firing_rate[::N_PLOT], joint_angle[::N_PLOT], dist=euclidean_norm)

        #plt.plot(x, combined_firing_rate)
        #plt.plot(x, joint_angle)
        #plt.show()

for i, j, k in itertools.product(range(N_ANGLES), range(N_SIMULATIONS), range(len(NOISES))):
    joint_angle = zscore.zscore(joint_angles[:parameters.general['N_steps'], i])
    joint_angle_noise = joint_angle[::N_PLOT] + np.random.normal(0, np.std(joint_angle[::N_PLOT]) * NOISES[k],
                                                            joint_angle[::N_PLOT].size)
    d_noise[i, j, k], _, _, _ = dtw(joint_angle[::N_PLOT], joint_angle_noise, dist=euclidean_norm)

d_average, d_average_noise = np.mean(d, axis=(0, 1)), np.mean(d_noise, axis=(0, 1))
d_std, d_std_noise = np.std(d, axis=(0, 1)), np.std(d_noise, axis=(0, 1))
d_25, d_25_noise = d_average - np.percentile(d, 25, axis=(0, 1)), d_average_noise - np.percentile(d_noise, 75, axis=(0, 1))
d_75, d_75_noise = np.percentile(d, 75, axis=(0, 1)) - d_average, np.percentile(d_noise, 75, axis=(0, 1)) - d_average_noise

d_max, d_max_noise = (np.max(d, axis=(0, 1)) - d_average), (np.max(d_noise, axis=(0, 1)) - d_average_noise)
d_min, d_min_noise = np.abs((np.min(d, axis=(0, 1)) - d_average)), np.abs((np.min(d_noise, axis=(0, 1)) - d_average_noise))

yticks = []
fig, ax = plt.subplots()

for i in range(d_average_noise.size):
    ax.errorbar(d_average_noise[i], i, xerr=[[d_min_noise[i]], [d_max_noise[i]]], fmt='o', color='grey', capsize=2)
    ax.errorbar(d_average_noise[i], i, xerr=[[d_25_noise[i]], [d_75_noise[i]]], fmt='o', color='black', capsize=3)
    yticks.append(f'noise = {NOISES[i]*100}%')

i = 0
for m, l in np.ndindex((b_list.size, tau_w_list.size)):
    ax.errorbar(d_average[l, m], i + 3, xerr=([d_min[l, m]], [d_max[l, m]]), fmt='o', color='grey', capsize=2)
    ax.errorbar(d_average[l, m], i + 3, xerr=[[d_25[l, m]], [d_75[l, m]]], fmt='o', color='black', capsize=3)
    yticks.append(r'$\tau_w$=' + str(np.around(tau_w_list[m] * 1000, 3)) + 'ms, b=' + str(np.around(b_list[l] * 1000, 3)) + 'mV')
    i += 1

ax.set_yticks(range(i + 3))
ax.set_yticklabels(yticks)
ax.set_xlabel('DTW Score', fontsize=15)
fig.tight_layout(pad=0.5)

fig.savefig('Images/DTW_plot.png', bbox_inches='tight')

# for i in np.arange(len(path[0]))[::skip]:
#    plt.plot([x[::n][path[0][i]], x[::n][path[1][i]]], [combined_firing_rate[::n][path[0][i]], joint_angle[::n][path[1][i]]], color='grey', alpha=0.7)
#plt.cla()
#plt.plot(x, combined_firing_rate, color='red')
#plt.plot(x, joint_angle, color='blue')
#plt.show()
