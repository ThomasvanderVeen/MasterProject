from LIF_simple_class import LIF_simple
from HairField_class import HairField
from AdEx_class import AdEx
from plots import *
from functions import *

variables = {'dt': 0.001, 't_total': 10, 'N_steps': None, 'N_sims': 19}
variables['N_steps'] = round(variables['t_total']/variables['dt'])

speeds = np.linspace(10, 190, num=variables['N_sims'])
ramp = (90/(speeds*variables['dt'])).astype(int)

parameters_hair_field = {'N_hairs': 20, 'min_joint_angle': 0, 'max_joint_angle': 180, 'max_angle': 90, 'overlap': 2,
                         'overlap_bi': 18}

parameters_AdEx = {'C': 200e-12, 'g_L': 12e-9, 'E_L': -70e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'V_T': -50e-3,
                   'tau_W': 600e-3, 'b': 8e-12, 'V_R': -70e-3, 'V_cut': -40e-3, 'refrac': 0.00,
                   'n': parameters_hair_field['N_hairs'], 'dt': variables['dt']}

parameters_LIF_simple = {'tau': 5e-3, 'tau_G': 500e-3, 'G_r': 25e-3, 'p': 0.1, 'V_T': -50e-3,
                         'V_R': -70e-3, 'n': 1, 'N_input': parameters_hair_field['N_hairs'], 'dt': variables['dt']
                         , 'refrac': 0}

spikes_rates = []
hair_field = HairField(parameters_hair_field)
hair_field.get_receptive_field()

for j in tqdm(range(variables['N_sims'])):

    t1 = torch.linspace(90, 180, steps=ramp[j])
    t2 = torch.linspace(180, 180, steps=variables['N_steps']-ramp[j])
    ramp_angles = torch.cat((t1, t2))
    hair_angles = hair_field.get_hair_angle(torch.Tensor.numpy(ramp_angles))/37e9

    adex = AdEx(parameters_AdEx)
    adex.initialize_state()

    lif = LIF_simple(parameters_LIF_simple)
    lif.initialize_state()

    time, spike_list, spike_inter = np.array([]), torch.empty(hair_angles.shape), np.empty([variables['N_steps']])

    for i in range(variables['N_steps']):
        _, spike_list[i, :] = adex.forward(hair_angles[i, :])
        _, spike_inter[i] = lif.forward(spike_list[i, :])
        time = np.append(time, i * parameters_AdEx['dt'])

    firing_rate, _ = get_firing_rate(spike_inter, variables['dt'])
    spikes_rates.append(np.mean(firing_rate[1:]))

fig, ax = plt.subplots()

ax.plot(speeds, spikes_rates, color='black')
ax.scatter(speeds, spikes_rates, color='black')

plot_movement_interneuron(ax, fig)
