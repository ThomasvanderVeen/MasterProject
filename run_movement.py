from class_hair_field import HairField
from class_sensory_neuron import AdEx
from class_velocity_neuron import LIF_simple
from dictionaries import Parameters
from functions import *
from plots import *

N_SPEEDS = 2

tau_list = [1e-3, 10e-3]
n_hairs_list = [15, 60]

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for l, k in itertools.product(range(len(tau_list)), range(len(n_hairs_list))):
    parameters = Parameters(max_joint_angle=180, min_joint_angle=0, n_hairs=n_hairs_list[k], t_total=5, dt=0.001, n_angles=1)

    speeds = np.linspace(10, 200, num=N_SPEEDS)
    ramp = (45 / (speeds * parameters.general['dt'])).astype(int)

    parameters.sensory['n'] = int(parameters.sensory['n'] / 2)
    parameters.velocity['n'] = 1
    parameters.velocity['tau_min'] = tau_list[l]

    spikes_rates = []
    hair_field = HairField(parameters.hair_field)
    hair_field.get_receptive_field()

    neurons = [AdEx, LIF_simple]
    parameters_list = [parameters.sensory, parameters.velocity]

    for j in range(N_SPEEDS):
        sensory_neuron, velocity_neuron = [define_and_initialize(neurons[i], parameters_list[i]) for i in
                                           range(len(neurons))]
        #t1 = torch.linspace(0, 0, steps=100)
        t11 = torch.linspace(0, 0, 100)
        t12 = torch.linspace(0, 0, 100)
        t1 = torch.linspace(0, 45, steps=ramp[j])
        t2 = torch.linspace(45, 45, steps=parameters.general['N_steps'] - ramp[j]-200)
        ramp_angles = torch.cat((t11, t12, t1, t2))
        hair_angles = hair_field.get_hair_angle(torch.Tensor.numpy(ramp_angles)) / 37e9

        time, spike_list, spike_inter = np.array([]), torch.empty(hair_angles.shape), np.empty(
            [parameters.general['N_steps']])

        for i in range(parameters.general['N_steps']):
            _, spike_list[i, :] = sensory_neuron.forward(torch.from_numpy(hair_angles[i, :]))
            _, spike_inter[i] = velocity_neuron.forward(spike_list[i, :], fill_with_ones(spike_list[i, :]))
            print(_)
            time = np.append(time, i * parameters.sensory['dt'])

        firing_rate = get_firing_rate_2(spike_inter[200:ramp[j]+200], parameters.general['dt'], t=0.001, nan_bool=0)



        spikes_rates.append(np.mean(firing_rate))

    if l == 0:
        ax1.scatter(speeds, spikes_rates, color=parameters.general['colors'][k], marker=parameters.general['markers'][k],
                label='$N_{h}$ = ' + str(n_hairs_list[k]), zorder=2)

    if k == len(n_hairs_list)-1:
        ax2.scatter(speeds, spikes_rates, color=parameters.general['colors'][l], marker=parameters.general['markers'][l],
                label=r'$\tau_I$ = ' + str(tau_list[l] * 1000) + 'ms', zorder=2)



plot_movement_interneuron(ax1, fig1, 'hairs')
plot_movement_interneuron(ax2, fig2, 'tau')
