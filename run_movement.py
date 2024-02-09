from class_hair_field import HairField
from class_sensory_neuron import AdEx
from class_velocity_neuron import LIF_simple
from class_primitive_neuron import LIF_primitive
from dictionaries import Parameters
from functions import *
from plots import *

N_SPEEDS = 5
tau_list = [1e-3]
n_hairs_list = [20, 40, 60]

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for l, k in itertools.product(range(len(tau_list)), range(len(n_hairs_list))):
    parameters = Parameters(max_joint_angle=180, min_joint_angle=0, n_hairs=n_hairs_list[k], t_total=5, dt=0.001, n_angles=1)

    speeds = np.linspace(40, 300, num=N_SPEEDS)
    ramp = (180 / (speeds * parameters.general['dt'])).astype(int)

    parameters.sensory['n'] = int(parameters.sensory['n'] / 2)
    parameters.velocity['n'] = 1
    parameters.LIF_NEW['n'] = int(parameters.sensory['n'])
    parameters.LIF_NEW['N_input'] = int(parameters.sensory['n'])
    parameters.velocity['tau_min'] = tau_list[l]

    spikes_rates = [0]
    hair_field = HairField(parameters.hair_field)
    hair_field.get_receptive_field()

    neurons = [AdEx, LIF_simple, LIF_primitive]
    parameters_list = [parameters.sensory, parameters.velocity, parameters.LIF_NEW]

    for j in range(N_SPEEDS):
        sensory_neuron, velocity_neuron, extra_neuron = [define_and_initialize(neurons[i], parameters_list[i]) for i in
                                           range(len(neurons))]



        #t11 = torch.linspace(0, 0, 100)
        #t22 = torch.linspace(0, 0, 100)
        t1 = torch.linspace(0, 180, steps=ramp[j])
        t2 = torch.linspace(180, 180, steps=parameters.general['N_steps'] - ramp[j])
        ramp_angles = torch.cat((t1, t2))
        hair_angles = hair_field.get_hair_angle(torch.Tensor.numpy(ramp_angles)) / 37e9

        time, spike_list, spike_inter, spike_new, voltage = np.array([]), torch.empty(hair_angles.shape), np.empty(
            [parameters.general['N_steps']]), np.empty([parameters.general['N_steps'], int(parameters.sensory['n'])]), np.empty([parameters.general['N_steps'], int(parameters.sensory['n'])])

        for i in range(parameters.general['N_steps']):
            _, spike_list[i, :] = sensory_neuron.forward(torch.from_numpy(hair_angles[i, :]))
            _, spike_inter[i] = velocity_neuron.forward(spike_list[i, :], fill_with_ones(spike_list[i, :]))
            voltage[i, :], spike_new[i, :] = extra_neuron.forward(spike_list[i, :])
            time = np.append(time, i * parameters.sensory['dt'])

        #plt.plot(time, voltage[:, 0])
        #plt.show()

        #plt.scatter(time, spike_new[:, 0])
        #plt.show()

        #plt.plot(range(5000), hair_angles*1e9)
        #plt.plot(range(5000), spike_inter)
        #plt.show()
        firing_rate = get_firing_rate_2(np.sum(spike_new, axis=1)[:ramp[j]], parameters.general['dt'], t=0.001, nan_bool=0)
        #firing_rate = get_firing_rate_2(spike_inter[:ramp[j]], parameters.general['dt'], t=0.001, nan_bool=0)
        print(np.mean(firing_rate))



        spikes_rates.append(np.mean(firing_rate))

    if l == 0:
        ax1.scatter(np.append(0, speeds), spikes_rates, color=parameters.general['colors'][k], marker=parameters.general['markers'][k],
                label='$N_{h}$ = ' + str(n_hairs_list[k]), zorder=2)

    if k == len(n_hairs_list)-1:
        ax2.scatter(np.append(0, speeds), spikes_rates, color=parameters.general['colors'][l], marker=parameters.general['markers'][l],
                label=r'$\tau_I$ = ' + str(tau_list[l] * 1000) + 'ms', zorder=2)



plot_movement_interneuron(ax1, fig1, 'hairs')
plot_movement_interneuron(ax2, fig2, 'tau')
