from functions import *
from dictionaries import Parameters
import matplotlib.pyplot as plt
N_simulations = 20
parameters = Parameters(t_total=5.5, dt=0.001)
i = 0
primitive_list = pickle_open('Data/primitive_list')
data = pickle_open('Data/simulation_data')
w_pos = [14e-3, 0, 12e-3, 10e-3, 7.5e-3]
w_vel = [12e-3, 14.5e-3, 0, 11e-3, 12e-3]
_, synapse_type, weights_primitive, primitive_filter_2, primitive_filter = get_encoding(w_pos, w_vel)

correlation_pos, correlation_neg = np.zeros((360, N_simulations)), np.zeros((360, N_simulations))
for i in range(N_simulations):
    pitch = np.array(data[f'simulation_{i}'][2][0, :])
    spike_primitive = primitive_list[i]
    pitch = pitch[:parameters.general['N_frames']]
    pitch = interpolate(pitch, parameters.general['t_total'], parameters.general['N_steps'])

    pitch_diff = np.diff(pitch)

    for j in range(360):
        correlation = spike_primitive[:-1, j]*pitch_diff

        correlation_pos[j, i] = correlation[correlation > 0].size
        correlation_neg[j, i] = correlation[correlation < 0].size

weights = (correlation_pos-correlation_neg)/(correlation_pos+correlation_neg)
colors = ['blue', 'black', 'green', 'yellow', 'orange']
for j in range(360):
    #plt.scatter(np.full(2, j), [np.max(weights[j]), np.min(weights[j])], color=colors[synapse_type[j]], s=2)
    #plt.plot(np.full(2, j), [np.max(weights[j]), np.min(weights[j])], color=colors[synapse_type[j]], linewidth=1)
    plt.scatter(j, np.mean(weights[j]), color=colors[synapse_type[j]], s=10)
    #plt.scatter(np.full(20, j), weights[j], color=colors[synapse_type[j]], s=2)

for i in [0, 60, 120, 180, 240, 300, 360]:
    plt.plot([i, i], [-0.5, 0.5], linestyle='dotted', color='black')
plt.show()

pickle_save(weights, 'Data/weights')
print(weights.shape)