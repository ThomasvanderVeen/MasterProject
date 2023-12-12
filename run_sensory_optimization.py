import pandas as pd

from class_sensory_neuron import AdEx
from dictionaries import Parameters
from functions import *
from plots import *


def get_tau(t_1, t_2, firing_rate_max, firing_rate_plateau):
    tau = (t_2 - t_1) / np.log(firing_rate_plateau / firing_rate_max)
    return tau


class RampGenerator:
    def __init__(self, parameters_ramp, device=None):
        self.n_ramp = parameters_ramp['n_ramp']
        self.n_steps = parameters_ramp['n_steps']
        self.height = parameters_ramp['height']
        self.low = parameters_ramp['low']
        self.device = device

    def ramp(self):
        n_dims = self.n_ramp.size

        input_ramp = torch.empty((self.n_steps, n_dims))

        for i in range(n_dims):
            t1 = torch.linspace(self.low[i], self.height[i], steps=self.n_ramp[i], dtype=torch.float64,
                                device=self.device)
            t2 = torch.linspace(self.height[i], self.height[i], steps=self.n_steps - 2 * self.n_ramp[i],
                                dtype=torch.float64, device=self.device)
            t3 = torch.linspace(self.height[i], self.low[i], steps=self.n_ramp[i], dtype=torch.float64,
                                device=self.device)

            input_ramp[:, i] = torch.cat((t1, t2, t3))

        return input_ramp


real_values = np.array(
    [[25, 105, 155, 200, 255], [0.1, 60, 95, 130, 150], [165, 150, 140, 130, 120], [90, 90, 90, 90, 90],
     [1.5, 0.8, 0.4, 0.2, 0.1]])
v_var, v_stat = np.array([24, 47, 88, 151, 245]), 47
max_angle_var, max_angle_stat = np.array([15, 23, 34, 46, 60]), 37
parameters = Parameters(max_joint_angle=1, min_joint_angle=1, n_hairs=1, t_total=10, dt=0.0001, n_angles=1)
parameters.sensory['n'] = 5
t_total = 10
colors = ['green', 'yellow', 'blue', 'black', 'red']
spike_rates, spike_rates_list = [], []
times, times_list = [], []
tau_list = np.linspace(400, 800, num=4, dtype=int) * 1E-3
b_list = np.linspace(1, 22, num=4, dtype=int) * 1e-12
MSE = np.zeros((tau_list.size, b_list.size))

for l in tqdm(range(tau_list.size)):
    for m in range(b_list.size):
        for [v, max_angle] in [[v_var, max_angle_stat], [v_stat, max_angle_var]]:

            N_steps = round(t_total / parameters.sensory['dt'])
            N_ramp = np.around(max_angle / (v * parameters.sensory['dt'])).astype(int)
            height = np.empty([parameters.sensory['n']])
            height[:] = max_angle / 37e9

            parameters_ramp = {'n_ramp': N_ramp, 'n_steps': N_steps, 'height': height, 'low': np.full(height.shape, 0)}
            ramp_generator = RampGenerator(parameters_ramp)
            input_ramp = ramp_generator.ramp()

            parameters.sensory['tau_W'] = tau_list[l]
            parameters.sensory['b'] = b_list[m]

            adex = AdEx(parameters.sensory)
            adex.initialize_state()

            voltage, time, spike_list = np.empty(input_ramp.numpy().shape), np.array([]), np.empty(
                input_ramp.numpy().shape)

            for i in range(N_steps):
                voltage[i, :], spike_list[i, :] = adex.forward(input_ramp[i])
                time = np.append(time, i * parameters.sensory['dt'])

            for j in range(parameters.sensory['n']):
                spikes = np.where(spike_list[:, j] == 1)

                ISI = np.diff(spikes)

                t_spikes = np.append(0, time[spikes])
                spike_rate = np.append(0, 1 / (ISI * parameters.sensory['dt']))
                spike_rate = np.append(spike_rate, 0)
                spike_rate = np.append(spike_rate, 0)
                t_spikes = np.append(t_spikes, time[-1])

                spike_rate_max = np.max(spike_rate)
                spike_rate_plateau = spike_rate[int(spike_rate.size / 2)]

                index_t_1 = np.where(spike_rate == spike_rate_max)
                index_t_2 = np.where(spike_rate == spike_rate_plateau)
                t_1 = t_spikes[index_t_1[0][0]]
                t_2 = t_spikes[index_t_2[0][5]]

                tau_predict = get_tau(t_1, t_2, spike_rate_max, spike_rate_plateau)

                if isinstance(v, int):
                    tau_real = get_tau(0, 1.7, real_values[0, j], real_values[1, j])

                    MSE[l, m] += abs(real_values[0, j] - spike_rate_max)
                    MSE[l, m] += abs(real_values[1, j] - spike_rate_plateau)
                    MSE[l, m] += abs(tau_real - tau_predict)
                else:
                    tau_real = get_tau(real_values[4, j], 2.5, real_values[2, j], real_values[3, j])

                    MSE[l, m] += abs(real_values[2, j] - spike_rate_max)
                    MSE[l, m] += abs(real_values[3, j] - spike_rate_plateau)
                    MSE[l, m] += abs(tau_real - tau_predict)

                spike_rates.append(spike_rate)
                times.append(t_spikes)

            spike_rates_list.append(spike_rates)
            times_list.append(times)
            spike_rates, times = [], []

MSE = MSE / real_values.size
MSE_flat = np.ndarray.flatten(MSE)
indexes, index_flat = np.where(MSE == np.min(MSE)), np.where(MSE_flat == np.min(MSE_flat))
tau_opt, b_opt = tau_list[indexes[0]], b_list[indexes[1]]
spike_rates = spike_rates_list[index_flat[0][0] * 2 + 1]
times = times_list[index_flat[0][0] * 2 + 1]

for i in range(len(spike_rates)):
    plt.plot(times[i], spike_rates[i])
plot_single_hair(plt.gca(), v_stat)

spike_rates = spike_rates_list[index_flat[0][0] * 2]
times = times_list[index_flat[0][0] * 2]

for i in range(len(spike_rates)):
    plt.plot(times[i], spike_rates[i])
plot_single_hair(plt.gca(), v_var)

df = pd.DataFrame(MSE.astype(int), columns=np.around(b_list, 15), index=np.around(tau_list, 5))
plot_heat_map(df)

print(f'[Single hair] Optimum tau {tau_opt}, Optimum b: {b_opt}')
