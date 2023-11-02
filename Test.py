import numpy as np

from functions import *
from dictionaries import Parameters
import matplotlib.pyplot as plt
from class_primitive_neuron import LIF_primitive
from scipy import signal

N_simulations = 12
parameters = Parameters(t_total=20, dt=0.001)
sim = 4
primitive_list = pickle_open('Data/primitive_list')
joint_angles_list = pickle_open('Data/joint_angles_list')
data = pickle_open('Data/simulation_data')
w_pos = [14e-3, 0, 12e-3, 10e-3, 7.5e-3]
w_vel = [12e-3, 14.5e-3, 0, 11e-3, 12e-3]
_, synapse_type, weights_primitive, primitive_filter_2, primitive_filter = get_encoding(w_pos, w_vel)
correlation_pos, correlation_neg = np.zeros((360, N_simulations)), np.zeros((360, N_simulations))
permutations = get_primitive_indexes(6)

N_legs = 6
true_positive, false_positive, true_negative, false_negative = [np.empty((N_simulations, 360)) for _ in range(4)]

for k in range(N_simulations):
    joint_angles, spike_primitive = joint_angles_list[k], primitive_list[k]
    ground_truth, ground_vel, ground_pos = [np.zeros([parameters.general['N_steps'], i]) for i in [360, 36, 36]]
    N_joints = joint_angles.shape[1]
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

    pitch = np.array(data[f'simulation_{k}'][2][1, :])
    spike_primitive = ground_truth
    pitch = pitch[:parameters.general['N_frames']]
    pitch = interpolate(pitch, parameters.general['t_total'], parameters.general['N_steps'])


    for j in range(360):
        spikes = spike_primitive[:, j]
        correlation = spikes*pitch/(np.sum(spikes)+1)
        #correlation = spikes * pitch

        correlation_pos[j, k] = np.mean(correlation)


#weights = np.ndarray.flatten((correlation_pos-correlation_neg)/(correlation_pos+correlation_neg))

weights = np.ndarray.flatten(np.mean(correlation_pos, axis=1))
weights = normalize(weights)
colors = ['blue', 'black', 'green', 'yellow', 'orange']


for j in range(360):
    #plt.scatter(np.full(2, j), [np.max(weights[j]), np.min(weights[j])], color=colors[synapse_type[j]], s=2)
    #plt.plot(np.full(2, j), [np.max(weights[j]), np.min(weights[j])], color=colors[synapse_type[j]], linewidth=1)
    plt.scatter(j, np.mean(weights[j]), color=colors[synapse_type[j]], s=10)
    #plt.scatter(np.full(20, j), weights[j], color=colors[synapse_type[j]], s=2)

for i in [0, 60, 120, 180, 240, 300, 360]:
    plt.plot([i, i], [-0.5, 0.5], linestyle='dotted', color='black')
plt.show()


weights = np.nan_to_num(weights)
pickle_save(weights, 'Data/weights_pitch')
pickle_save(ground_truth, 'Data/ground_truth')

weights = pickle_open('Data/weights_pitch')
ground_truth = pickle_open('Data/ground_truth')
weights = np.nan_to_num(weights)

parameters.posture['w'] = weights/200
#parameters.posture['w'] = np.ones(360)/300
parameters.posture['tau'] = 50e-3
posture_neuron = define_and_initialize(LIF_primitive, parameters.posture)

spike_primitive = primitive_list[sim]
spike_posture, voltage, time = np.empty(parameters.general['N_steps']), np.empty(parameters.general['N_steps']), np.array([])

fig, ax3 = plt.subplots()
ax4 = ax3.twinx()
spike_primitive = primitive_list[sim]
for i in tqdm(range(parameters.general['N_steps'])):
    time = np.append(time, i * parameters.general['dt'])
    voltage[i], spike_posture[i] = posture_neuron.forward(torch.from_numpy(spike_primitive[i, :]))
    #print(torch.from_numpy(ground_truth[i, :]))

#plt.plot(time, voltage)
#plt.show()


#firing_rate, spikes_index = get_firing_rate(spike_posture, parameters.general['dt'])
firing_rate = get_firing_rate_2(spike_posture, parameters.general['dt'], 0.25)
firing_rate = gaussian_filter(firing_rate)
difference = np.mean(np.absolute(normalize(firing_rate) - normalize(pitch)))
fig.text(0.65, 0.85, difference)
ax3.plot(time, firing_rate, color='b')
#plt.plot(time, spike_posture)
#ax3.plot(time, voltage)
ax4.plot(time, pitch, color='r')
fig.legend(['model', 'ground_truth'])
plt.show()

