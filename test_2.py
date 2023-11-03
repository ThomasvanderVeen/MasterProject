from functions import *
from dictionaries import Parameters
from class_primitive_neuron import LIF_primitive
from scipy.signal import find_peaks

data = pickle_open('Data/simulation_data')
primitive_list = pickle_open('Data/primitive_list')
parameters = Parameters(t_total=20, dt=0.001)

N_simulations = 12
sim = 1

pitch = np.empty((parameters.general['N_steps'], N_simulations))
for i in range(N_simulations):
    pitch_single = np.array(data[f'simulation_{i}'][2][1, :])
    pitch_single = pitch_single[:parameters.general['N_frames']]
    pitch[:, i] = interpolate(pitch_single, parameters.general['t_total'], parameters.general['N_steps'])

pitch_max, pitch_min = np.max(pitch), np.min(pitch)

#for i in range(10):
#    plt.plot(range(pitch.shape[0]), pitch[:, i])
#    plt.show()

N_bins = 2
x = np.linspace(pitch_min, pitch_max, num=N_bins)
N_spikes = np.zeros(360)
weights = np.zeros((360, N_bins))

for j in range(360):
    bin = np.zeros(N_bins)
    bin_pitch = np.zeros(N_bins)
    for i in range(N_simulations):
        spike_primitive = primitive_list[i]
        spike_train = spike_primitive[:, j]
        incidence = spike_train*pitch[:, i]
        incidence[incidence == 0] = np.nan
        bin += np.histogram(incidence, bins=N_bins, range=(pitch_min, pitch_max))[0]
        bin_pitch += np.histogram(pitch[:, i], bins=N_bins, range=(pitch_min, pitch_max))[0]

    ratio = bin[0]/bin[1]
    if ratio > 2.3:
        weights[j, 0] = 1.5E-3
    if ratio < 1/3:
        weights[j, 1] = 1.5E-3



    #weight = bin/bin_pitch
    #weight = weight/np.sum(weight)


    #weights[j, :] = weight/N_spikes[j]


parameters.posture['n'] = N_bins
parameters.posture['w'] = weights.T
posture_neuron = define_and_initialize(LIF_primitive, parameters.posture)
spike_posture, voltage, time = np.empty((parameters.general['N_steps'], N_bins)), np.empty((parameters.general['N_steps'], N_bins)), np.array([])

spike_primitive = primitive_list[sim]
for i in tqdm(range(parameters.general['N_steps'])):
    time = np.append(time, i * parameters.general['dt'])
    voltage[i, :], spike_posture[i, :] = posture_neuron.forward(torch.from_numpy(spike_primitive[i, :]))

spike_posture[spike_posture == 0] = np.nan
'''
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax2.plot(time, pitch[:, sim], color='black')
ax2.plot([time[0], time[-1]], [pitch_max/2 + pitch_min/2, pitch_max/2 + pitch_min/2], linestyle='dotted', color='red')
ax.scatter(time, pitch[:, sim]*spike_posture[:, 0], color='red', s=20)
ax.scatter(time, pitch[:, sim]*spike_posture[:, 1], color='blue', s=20)
plt.show()
'''
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax2.plot(time, pitch[:, sim])
ax2.plot([time[0], time[-1]], [pitch_max/2 + pitch_min/2, pitch_max/2 + pitch_min/2])
ax.scatter(time, pitch[:, sim]*spike_posture[:, 0])
ax.scatter(time, pitch[:, sim]*spike_posture[:, 1])
plt.show()

