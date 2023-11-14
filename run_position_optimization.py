from dtw import dtw
from functions import *
from dictionaries import Parameters
from plots import *
from scipy import interpolate
import scipy.stats.mstats as zscore
from class_position_neuron import LIF

position_list = pickle_open('Data/position_list')
sensory_list = pickle_open('Data/sensory_list')
joint_angles_list = pickle_open('Data/joint_angles_list')

n_grid = 4
noises = [0.01, 0.05, 0.10]
n = 100
skip = 10

parameters = Parameters(t_total=6, dt=0.0001, n_hairs=20)
x = np.linspace(0, parameters.general['t_total'], num=parameters.general['N_steps'])
d, d_noise = np.empty((18, len(position_list), n_grid, n_grid)), np.empty((18, len(position_list), 3))
b_list = np.linspace(10e-3, 40e-3, num=n_grid)
tau_w_list = np.linspace(10e-3, 25e-3, num=n_grid)
euclidean_norm = lambda x, y: np.abs(x - y)

for j in tqdm(range(len(position_list))):
    joint_angles = joint_angles_list[j]
    spike_sensory = sensory_list[j]
    spike_position = np.zeros((parameters.general['N_steps'], 36))
    position_neuron = define_and_initialize(LIF, parameters.position)

    for l in range(b_list.size):
        for m in range(tau_w_list.size):

            parameters.position['b'] = b_list[l]
            parameters.position['tau_W'] = tau_w_list[m]
            position_neuron = define_and_initialize(LIF, parameters.position)

            for i in range(parameters.general['N_steps']):
                reshaped_spikes = torch.reshape(torch.from_numpy(spike_sensory[i, :]),
                                                (parameters.velocity['n'], (parameters.hair_field['N_hairs'])))

                _, spike_position[i, :] = position_neuron.forward(
                    reshaped_spikes[:, int(reshaped_spikes.shape[1]/2)-1:])

            for i in range(18):
                firing_rate_1, spikes_index_1 = get_firing_rate(spike_position[:, 2*i], parameters.general['dt'])
                firing_rate_2, spikes_index_2 = get_firing_rate(spike_position[:, 2*i+1], parameters.general['dt'])

                func1 = interpolate.interp1d(x[spikes_index_1], firing_rate_1)
                func2 = interpolate.interp1d(x[spikes_index_2], firing_rate_2)

                combined_firing_rate = np.zeros(x.shape)
                combined_firing_rate[spikes_index_1[0]:spikes_index_1[-1]] -= func1(x[spikes_index_1[0]:spikes_index_1[-1]])
                combined_firing_rate[spikes_index_2[0]:spikes_index_2[-1]] += func2(x[spikes_index_2[0]:spikes_index_2[-1]])

                minimum = np.min([spikes_index_1[0], spikes_index_2[0]])
                maximum = np.max([spikes_index_1[-1], spikes_index_2[-1]])

                combined_firing_rate = zscore.zscore(combined_firing_rate[minimum:maximum])
                joint_angle = zscore.zscore(joint_angles[minimum:maximum, i])

                x_plot = x[minimum:maximum]

                d[i, j, l, m], cost_matrix, acc_cost_matrix, path = dtw(combined_firing_rate[::n], joint_angle[::n], dist=euclidean_norm)

    for i in range(18):
        for k in range(len(noises)):

            joint_angle = zscore.zscore(joint_angles[:, i])
            joint_angle_noise = joint_angle[::n]+np.random.normal(0, np.std(joint_angle[::n])*noises[k], joint_angle[::n].size)
            d_noise[i, j, k], _, _, _ = dtw(joint_angle[::n], joint_angle_noise, dist=euclidean_norm)


d_average, d_average_noise = np.mean(d, axis=0), np.mean(d_noise, axis=0)
d_average, d_average_noise = np.mean(d_average, axis=0), np.mean(d_average_noise, axis=0)

d_std, d_std_noise = np.std(d, axis=0), np.std(d_noise, axis=0)
d_std, d_std_noise = np.std(d_std, axis=0), np.std(d_std_noise, axis=0)

pickle_save([d_average, d_std, d_average_noise, d_std_noise], 'Data/temp_d')


d_average, d_std, d_average_noise, d_std_noise = pickle_open('Data/temp_d')
print(d_average, d_std)
print(d_average_noise, d_std_noise)
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
yticks = ['start']
fig, ax = plt.subplots(figsize=(1.5*3.54, 3.54), dpi=600)

for i in range(d_average_noise.size):
    ax.errorbar(d_average_noise[i]/1000, i, xerr=d_std_noise[i]/1000, fmt='o', color='black', capsize=3)
    yticks.append(f'noise = {noises[i]}%')
    print(f'noise = {noises[i]}')

i = 0
for l in range(b_list.size):
    for m in range(tau_w_list.size):
        ax.errorbar(d_average[l, m]/1000, i+3, xerr=d_std[l, m]/1000, fmt='o', color='black', capsize=3)
        yticks.append(f'tau={tau_w_list[m]*1000}ms, b={b_list[l]*1000}mV')
        i += 1
ax.set_yticklabels(yticks)
ax.set_xlabel('DTW Score', fontsize=15)
fig.tight_layout(pad=0.5)

fig.savefig('Images/DTW_plot', bbox_inches='tight')



'''
for i in np.arange(len(path[0]))[::skip]:
    plt.plot([x_plot[::n][path[0][i]], x_plot[::n][path[1][i]]], [combined_firing_rate[::n][path[0][i]], joint_angle[::n][path[1][i]]], color='grey', alpha=0.7)

plt.plot(x_plot, combined_firing_rate, color='red')
plt.plot(x_plot, joint_angle, color='blue')
plt.show()
'''