import scipy.stats.mstats as zscore
from dtw import dtw

from class_position_neuron import LIF
from dictionaries import Parameters
from plots import *

position_list = pickle_open('Data/position_list')
sensory_list = pickle_open('Data/sensory_list')
joint_angles_list = pickle_open('Data/joint_angles_list')

n_grid = 4
noises = [0.01, 0.05, 0.10]
n = 100
skip = 10

parameters = Parameters(t_total=5, dt=0.0001, n_hairs=200)
x = np.linspace(0, parameters.general['t_total'], num=parameters.general['N_steps'])
d, d_noise = np.empty((18, len(position_list), n_grid, n_grid)), np.empty((18, len(position_list), 3))
b_list = np.linspace(5e-4, 5e-3, num=n_grid)
tau_w_list = np.linspace(1e-3, 5e-3, num=n_grid)
euclidean_norm = lambda x, y: np.abs(x - y)

for j, l, m in itertools.product(range(len(position_list)), range(b_list.size), range(tau_w_list.size)):
    joint_angles = joint_angles_list[j]
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

    for i in range(18):
        firing_rate_1 = get_firing_rate_2(spike_position[:, 2 * i], parameters.general['dt'], 0.1, nan_bool=False)
        firing_rate_2 = get_firing_rate_2(spike_position[:, 2 * i + 1], parameters.general['dt'], 0.1, nan_bool=False)

        combined_firing_rate = firing_rate_2 - firing_rate_1
        combined_firing_rate = zscore.zscore(combined_firing_rate)

        joint_angle = zscore.zscore(joint_angles[:, i])

        d[i, j, l, m], cost_matrix, acc_cost_matrix, path = dtw(combined_firing_rate[::n], joint_angle[::n],
                                                                dist=euclidean_norm)

for j in range(len(position_list)):
    for i in range(18):
        for k in range(len(noises)):
            joint_angle = zscore.zscore(joint_angles[:, i])
            joint_angle_noise = joint_angle[::n] + np.random.normal(0, np.std(joint_angle[::n]) * noises[k],
                                                                    joint_angle[::n].size)
            d_noise[i, j, k], _, _, _ = dtw(joint_angle[::n], joint_angle_noise, dist=euclidean_norm)

d_average, d_average_noise = np.mean(d, axis=(0, 1)), np.mean(d_noise, axis=(0, 1))
d_std, d_std_noise = np.std(d, axis=(0, 1)), np.std(d_noise, axis=(0, 1))
d_max, d_max_noise = (np.max(d, axis=(0, 1)) - d_average), (np.max(d_noise, axis=(0, 1)) - d_average_noise)

# pickle_save([d_average, d_std, d_average_noise, d_std_noise], 'Data/temp_d')
# d_average, d_std, d_average_noise, d_std_noise = pickle_open('Data/temp_d')

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=6.5)
yticks = []
fig, ax = plt.subplots(figsize=(1.5 * 3.54, 3.54), dpi=600)

for i in range(d_average_noise.size):
    ax.errorbar(d_average_noise[i], i, xerr=d_std_noise[i], fmt='o', color='black', capsize=3)
    # ax.errorbar(d_average_noise[i], i, xerr=d_max_noise[i], fmt='o', color='black', capsize=2)
    yticks.append(f'noise = {noises[i]}%')
    print(f'noise = {noises[i]}')

i = 0
for m, l in np.ndindex((b_list.size, tau_w_list.size)):
    ax.errorbar(d_average[l, m], i + 3, xerr=d_std[l, m], fmt='o', color='black', capsize=3)
    # ax.errorbar(d_average[l, m], i + 3, xerr=d_max[l, m], fmt='o', color='black', capsize=2)
    yticks.append(f'tau={np.around(tau_w_list[m] * 1000, 3)}ms, b={np.around(b_list[l] * 1000, 3)}mV')
    i += 1

ax.set_yticks(range(i + 3))
ax.set_yticklabels(yticks)
ax.set_xlabel('DTW Score', fontsize=15)
fig.tight_layout(pad=0.5)

fig.savefig('Images/DTW_plot.pdf', bbox_inches='tight')

# for i in np.arange(len(path[0]))[::skip]:
#    plt.plot([x[::n][path[0][i]], x[::n][path[1][i]]], [combined_firing_rate[::n][path[0][i]], joint_angle[::n][path[1][i]]], color='grey', alpha=0.7)
plt.cla()
plt.plot(x, combined_firing_rate, color='red')
plt.plot(x, joint_angle, color='blue')
plt.show()
