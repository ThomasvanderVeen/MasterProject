from dictionaries import Parameters
from class_velocity_neuron import LIF_simple
from class_hair_field import HairField
from class_sensory_neuron import AdEx
from plots import *
from functions import *

variables = {'dt': 0.001, 't_total': 10, 'N_steps': None, 'N_sims': 19}
variables['N_steps'] = round(variables['t_total']/variables['dt'])

speeds = np.linspace(10, 190, num=variables['N_sims'])
ramp = (90/(speeds*variables['dt'])).astype(int)

parameters = Parameters(180, 0, variables['dt'], N_hairs=20)
parameters.sensory['n'] = int(parameters.sensory['n']/2)
parameters.velocity['n'] = 1

spikes_rates = []
hair_field = HairField(parameters.hair_field)
hair_field.get_receptive_field()

neurons = [AdEx, LIF_simple]
parameters_list = [parameters.sensory, parameters.velocity]

for j in tqdm(range(variables['N_sims'])):

    sensory_neuron, velocity_neuron = [define_and_initialize(neurons[i], parameters_list[i]) for i in
                                       range(len(neurons))]

    t1 = torch.linspace(90, 180, steps=ramp[j])
    t2 = torch.linspace(180, 180, steps=variables['N_steps']-ramp[j])
    ramp_angles = torch.cat((t1, t2))
    hair_angles = hair_field.get_hair_angle(torch.Tensor.numpy(ramp_angles))/37e9

    time, spike_list, spike_inter = np.array([]), torch.empty(hair_angles.shape), np.empty([variables['N_steps']])

    for i in range(variables['N_steps']):
        _, spike_list[i, :] = sensory_neuron.forward(hair_angles[i, :])
        _, spike_inter[i] = velocity_neuron.forward(spike_list[i, :])
        time = np.append(time, i * parameters.sensory['dt'])

    firing_rate, pik = get_firing_rate(spike_inter, variables['dt'])
    spikes_rates.append(np.mean(firing_rate[2:]))

fig, ax = plt.subplots()

ax.plot(speeds, spikes_rates, color='black')
ax.scatter(speeds, spikes_rates, color='black')

plot_movement_interneuron(ax, fig)
