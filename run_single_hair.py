from dictionaries import Parameters
from class_sensory_neuron import AdEx
from plots import *
from functions import *


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


v_var, v_stat = np.array([24, 47, 88, 151, 245]), 47
max_angle_var, max_angle_stat = np.array([15, 23, 34, 46, 60]), 37
parameters = Parameters(max_joint_angle=1, min_joint_angle=1, n_hairs=1, t_total=10, dt=0.0001, n_angles=1)
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
