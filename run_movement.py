from class_hair_field import HairField
from class_sensory_neuron import AdEx
from class_velocity_neuron import LIF_simple
from dictionaries import Parameters
from functions import *
from plots import *

N_SPEEDS = 4

n_hairs_list = [200]
p_list = [0.1, 0.5, 0.9]

fig, ax = plt.subplots()


for l, k in itertools.product(range(len(p_list)), range(len(n_hairs_list))):
    parameters = Parameters(max_joint_angle=180, min_joint_angle=0, n_hairs=n_hairs_list[k], t_total=5, dt=0.001, n_angles=1)

    speeds = np.linspace(10, 600, num=N_SPEEDS)
    ramp = (45 / (speeds * parameters.general['dt'])).astype(int)

    parameters.sensory['n'] = int(parameters.sensory['n'] / 2)
    parameters.velocity['n'] = 1

    spikes_rates = []
    hair_field = HairField(parameters.hair_field)
    hair_field.get_receptive_field()

    parameters.velocity['p'] = p_list[l]

    neurons = [AdEx, LIF_simple]
    parameters_list = [parameters.sensory, parameters.velocity]

    for j in range(N_SPEEDS):
        sensory_neuron, velocity_neuron = [define_and_initialize(neurons[i], parameters_list[i]) for i in
                                           range(len(neurons))]

        t1 = torch.linspace(0, 45, steps=ramp[j])
        t2 = torch.linspace(45, 45, steps=parameters.general['N_steps'] - ramp[j])
        ramp_angles = torch.cat((t1, t2))
        hair_angles = hair_field.get_hair_angle(torch.Tensor.numpy(ramp_angles)) / 37e9

        time, spike_list, spike_inter = np.array([]), torch.empty(hair_angles.shape), np.empty(
            [parameters.general['N_steps']])

        for i in range(parameters.general['N_steps']):
            _, spike_list[i, :] = sensory_neuron.forward(torch.from_numpy(hair_angles[i, :]))
            volt, spike_inter[i] = velocity_neuron.forward(spike_list[i, :], fill_with_ones(spike_list[i, :]))
            #print(spike_list[i, :])
            #print(fill_with_ones(spike_list[i, :]))
            time = np.append(time, i * parameters.sensory['dt'])

        firing_rate = get_firing_rate_2(spike_inter[:ramp[j]], parameters.general['dt'], t=0.001, nan_bool=0)
        spikes_rates.append(np.mean(firing_rate))
        print(np.mean(firing_rate))
        #plt.plot(range(firing_rate.size), firing_rate)
        #plt.show()

    ax.plot(speeds, spikes_rates, color=parameters.general['colors'][k], marker=parameters.general['markers'][k],
            linestyle=parameters.general['linestyles'][l], label='$N_{h}$ = ' + str(n_hairs_list[k]) + ', p = ' + str(p_list[l]), zorder=2)

plot_movement_interneuron(ax, fig)
