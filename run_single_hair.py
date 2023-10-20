from dictionaries import Parameters
from class_sensory_neuron import AdEx
from ramp_generator import RampGenerator
from plots import *
from functions import *

v_var, v_stat = np.array([24, 47, 88, 151, 245]), 47
max_angle_var, max_angle_stat = np.array([15, 23, 34, 46, 60]), 37
parameters = Parameters(1, 1, 1, 10, 0.0001, 1)
parameters.sensory['n'] = 5

t_total = 10
colors = ['green', 'yellow', 'blue', 'black', 'red']

for [v, max_angle] in [[v_var, max_angle_stat], [v_stat, max_angle_var]]:

    N_steps = round(t_total/parameters.sensory['dt'])
    N_ramp = np.around(max_angle/(v*parameters.sensory['dt'])).astype(int)
    height = np.empty([parameters.sensory['n']])
    height[:] = max_angle/37e9

    parameters_ramp = {'n_ramp': N_ramp, 'n_steps': N_steps, 'height': height, 'low': np.full(height.shape, 0)}
    ramp_generator = RampGenerator(parameters_ramp)
    input_ramp = ramp_generator.ramp()

    adex = AdEx(parameters.sensory)
    adex.initialize_state()

    voltage, time, spike_list = np.empty(input_ramp.numpy().shape), np.array([]), np.empty(input_ramp.numpy().shape)

    for i in tqdm(range(N_steps)):
        voltage[i, :], spike_list[i, :] = adex.forward(input_ramp[i])
        time = np.append(time, i*parameters.sensory['dt'])

    for j in range(parameters.sensory['n']):
        spikes = np.where(spike_list[:, j] == 1)

        ISI = np.diff(spikes)

        t_spikes = np.append(0, time[spikes])
        spike_rate = np.append(0, 1/(ISI*parameters.sensory['dt']))
        spike_rate = np.append(spike_rate, 0)
        spike_rate = np.append(spike_rate, 0)
        t_spikes = np.append(t_spikes, time[-1])

        plt.plot(t_spikes, spike_rate)

    plot_single_hair(plt.gca(), v)
