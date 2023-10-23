from dictionaries import Parameters
from class_velocity_neuron import LIF_simple
from class_primitive_neuron import LIF_primitive
from class_position_neuron import LIF
from class_sensory_neuron import AdEx
from class_hair_field import HairField
from plots import *
from functions import *

N_simulations = 25
w_pos = [19e-3, 0, 19e-3, 12e-3, 12e-3]
w_vel = [19e-3, 19e-3, 0, 12e-3, 12e-3]
_, synapse_type, weights_primitive, primitive_filter_2, primitive_filter = get_encoding(w_pos, w_vel)
permutations = get_primitive_indexes(6)
data = pickle_open('simulation_data')

joint_angles_list, primitive_list = [], []

for k in tqdm(range(N_simulations), desc='Network progress'):
    joint_angles = np.array(data[f'simulation_{k}'][0]).T

    parameters = Parameters(max_joint_angle=np.amax(joint_angles, axis=0), min_joint_angle=np.amin(joint_angles, axis=0),
                            N_hairs=20, t_total=8, dt=0.0003, N_sims=18)
    parameters.position['N_input'] = int(parameters.position['N_input']/2)
    parameters.primitive['n'] = 360
    parameters.primitive['w'] = weights_primitive

    joint_angles = joint_angles[:parameters.general['N_frames']]
    joint_angles = interpolate(joint_angles, parameters.general['t_total'], parameters.general['N_steps'])

    hair_angles = np.zeros((joint_angles.shape[0], 2*joint_angles.shape[1]*parameters.hair_field['N_hairs']))

    for i in range(parameters.general['N_sims']):
        hair_field = HairField(parameters.hair_field)
        hair_field.reset_max_min(i)
        hair_field.get_double_receptive_field()
        hair_angles[:, i * 2 * parameters.hair_field['N_hairs']: 2 * parameters.hair_field['N_hairs']
                    + i * 2 * parameters.hair_field['N_hairs']] = hair_field.get_hair_angle(joint_angles[:, i])/37e9

    neurons = [AdEx, LIF, LIF_simple, LIF_primitive]
    parameters_list = [parameters.sensory, parameters.position, parameters.velocity, parameters.primitive]
    sensory_neuron, position_neuron, velocity_neuron, primitive_neuron = \
        [define_and_initialize(neurons[i], parameters_list[i]) for i in range(len(neurons))]

    time, spike_sensory = np.array([]), torch.empty(hair_angles.shape)
    spike_position, spike_velocity, spike_primitive = \
        [torch.empty((parameters.general['N_steps'], par['n'])) for par in parameters_list[1:]]

    for i in range(parameters.general['N_steps']):
        time = np.append(time, i)
        _, spike_sensory[i, :] = sensory_neuron.forward(torch.from_numpy(hair_angles[i, :]))

        reshaped_spikes = torch.reshape(spike_sensory[i, :], (parameters.velocity['n'], (parameters.hair_field['N_hairs'])))

        _, spike_velocity[i, :] = velocity_neuron.forward(reshaped_spikes)
        _, spike_position[i, :] = position_neuron.forward(reshaped_spikes[:, int(parameters.hair_field['N_hairs']/2):])

        pos_vel_spikes = prepare_spikes_primitive(spike_velocity[i, :], spike_position[i, :], permutations, primitive_filter)

        _, spike_primitive[i, :] = primitive_neuron.forward(pos_vel_spikes)

    primitive_list.append(spike_primitive), joint_angles_list.append(joint_angles)

#if False:
'''
Position neuron testing
'''

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

for i in range(2):
    firing_rate, spikes_index = get_firing_rate(spike_position[:, i], parameters.general['dt'])

    ax1.plot(time[spikes_index], firing_rate)

ax2.plot(time, joint_angles[:, 0], color='black')
ax2.plot(time, np.full(len(time), np.max(joint_angles[:, 0])/2 + np.min(joint_angles[:, 0])/2), linestyle='dotted',
         color='black')
plot_position_interneuron(ax1, ax2, fig, 'bi')

fig2, ax3 = plt.subplots()
ax4 = ax3.twinx()

N_hairs, N_half = parameters.hair_field['N_hairs'], int(parameters.hair_field['N_hairs']/2)

for i in range(N_half):
    spike_sensory[spike_sensory == 0] = np.nan
    ax3.scatter(time, (-i+N_half)*spike_sensory[:, i+N_half], color='dodgerblue', s=1)
    ax3.scatter(time, (1+i+N_half)*spike_sensory[:, i+N_half+N_hairs], color='red', s=1)
ax4.plot(time, joint_angles[:, 0], color='black')
ax4.plot(time, np.full(time.shape, np.max(joint_angles[:, 0])/2 + np.min(joint_angles[:, 0])/2), linestyle='dotted', color='red')
plot_spike_timing(ax3, ax4, fig2, N_hairs)

'''
Velocity neuron testing
'''

spike_velocity = spike_velocity.numpy()
spike_velocity[spike_velocity == 0] = np.nan

fig, ax = plt.subplots()

plt.plot(time, joint_angles[:, 0], color='black')
plt.scatter(time, joint_angles[:, 0]*spike_velocity[:, 0], color='blue')
plt.scatter(time, joint_angles[:, 0]*spike_velocity[:, 1], color='red')

plot_movement_interneuron_network(ax, fig)

'''
Primitive neuron testing (ROC plot)
'''
'''
N_joints = joint_angles.shape[1]
N_legs = 6
true_positive, false_positive, true_negative, false_negative = [np.empty((N_simulations, 60)) for _ in range(4)]
for k in tqdm(range(N_simulations), desc='ROC plot progress'):
    joint_angles, spike_primitive = joint_angles_list[k], primitive_list[k]
    ground_truth, ground_vel, ground_pos = [np.zeros([parameters.general['N_steps'], i]) for i in [360, 36, 36]]

    for i in range(N_joints):
        mid = np.max(joint_angles[:, i]) / 2 + np.min(joint_angles[:, i]) / 2
        diff = np.diff(joint_angles[:, i])
        ground_vel[np.where(diff < 0), 0 + 2 * i] = 1
        ground_vel[np.where(diff > 0), 1 + 2 * i] = 1
        ground_pos[np.where(joint_angles[:, i] < mid), 0 + 2 * i] = 1
        ground_pos[np.where(joint_angles[:, i] > mid), 1 + 2 * i] = 1

    for j in range(parameters.general['N_steps']):
        ground_truth_2 = prepare_spikes_primitive(torch.from_numpy(ground_vel[j, :]), torch.from_numpy(ground_pos[j, :]),
                                                  permutations, primitive_filter) + primitive_filter_2
        ground_truth_2 = torch.sum(ground_truth_2, dim=1)
        ground_truth[j, ground_truth_2 > 2.9] = 1
        ground_truth[j, ground_truth_2 < 2.9] = 0

    ground_truth_bins = convert_to_bins(ground_truth, 100)
    spike_primitive_bins = convert_to_bins(spike_primitive.numpy(), 100)

    for i in range(60):
        intersect = spike_primitive_bins[:, i] + ground_truth_bins[:, i]
        difference = spike_primitive_bins[:, i] - ground_truth_bins[:, i]

        true_positive[k, i] = intersect[intersect > 1.5].size
        false_positive[k, i] = difference[difference > 0.5].size
        true_negative[k, i] = intersect[intersect < 0.5].size
        false_negative[k, i] = difference[difference < -0.5].size

fig, ax = plt.subplots()
colors = ['blue', 'black', 'green', 'yellow', 'orange']

true_pos_sum = np.sum(true_positive, axis=0)
false_pos_sum = np.sum(false_positive, axis=0)
true_neg_sum = np.sum(true_negative, axis=0)
false_neg_sum = np.sum(false_negative, axis=0)

for i in range(60):
    plt.scatter(false_pos_sum[i]/(false_pos_sum[i] + true_neg_sum[i] + 0.0000001),
                true_pos_sum[i]/(true_pos_sum[i] + false_neg_sum[i] + 0.0000001), color=colors[synapse_type[i]])

plt.plot([0, 1], [0, 1], color='red', linestyle='dotted')
plot_primitive_interneuron(ax, fig)
'''
'''
Primitive neuron testing (PSTH plot)
'''

for m in tqdm(range(6), desc='PSTH plot progress'):
    swing_bin_rate, stance_bin_rate, swing_bin_likelihood, stance_bin_likelihood = [np.empty((N_simulations, 60, i)) for
                                                                                    i in [10, 20, 10, 20]]
    for k in range(N_simulations):
        gait = np.array(data[f'simulation_{k}'][1])[m, :]
        gait = gait[:parameters.general['N_frames']]
        gait = interpolate(gait, parameters.general['t_total'], parameters.general['N_steps'], True)
        for i in range(60):
            swing_bin_rate[k, i, :], stance_bin_rate[k, i, :], swing_bin_likelihood[k, i, :], stance_bin_likelihood[k, i, :] = \
                get_stance_swing_bins(gait, spike_primitive[:, i + 60*m].numpy())

    swing_bin_likelihood, stance_bin_likelihood = np.mean(swing_bin_likelihood, axis=0), np.mean(stance_bin_likelihood, axis=0)
    fig, ax = plt.subplots()
    for i in range(60):
        ax.scatter(np.linspace(0, .475, num=10), swing_bin_likelihood[i, :], color='red', marker='^')
        ax.scatter(np.linspace(.525, 1.5, num=20), stance_bin_likelihood[i, :], color='blue', marker='^')
        ax.set_ylabel('Likelihood of Spiking')
        fig.text(0.33, 0.04, 'Swing', ha='center')
        fig.text(0.66, 0.04, 'Stance')
        ax.set_xticks([])
        fig.savefig(f'Images_PSTH/neuron_{i}_leg_{m}')
        plt.cla()

'''
Body pitch estimation
'''
'''
def highpass(x, dt, tau):
    y = np.zeros(len(x))
    alpha = dt/(dt+tau)
    y[0] = x[0]
    for i in np.arange(1,len(x),1):
        y[i] = alpha*(y[i-1] + x[i] - x[i-1])
    return y

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.preprocessing import scale

pitch_list = []
for i in range(N_simulations):
    pitch = np.array(data[f'simulation_{i}'][2])
    pitch = pitch[:parameters.general['N_frames']]
    pitch = interpolate(pitch, parameters.general['t_total'], parameters.general['N_steps'])
    pitch_list.append(pitch)

model = SGD(alpha=0.001, max_iter=10000, verbose=100)

x = np.stack(primitive_list)
y = np.stack(pitch_list)

x = highpass(primitive_list[0][0, :], parameters.general['dt'], 0.005)
plt.plot(range(360), x)
plt.show()


y = scale(y)
#plt.plot(range(1600), y[0, :])
plt.show()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(x_train, y_train)
score_train = model.score(x_train, y_train)
score_test = model.score(x_test, y_test)

print(score_train, score_test)
'''