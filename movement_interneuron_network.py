from LIF_simple_class import LIF_simple
from AdEx_class import AdEx
from HairField_class import HairField
from plots import *
from functions import *

variables = {'dt': 0.001, 't_total': 10, 'N_steps': None}
variables['N_steps'] = round(variables['t_total']/variables['dt'])

data = pickle_open('simulation_data')

joint_angle = data[f'simulation_0'][0][:1000]
joint_angle = interpolate(joint_angle, variables['t_total'], variables['N_steps'])

parameters_hair_field = {'N_hairs': 20, 'min_joint_angle': np.min(joint_angle), 'max_joint_angle': np.max(joint_angle),
                         'max_angle': 90, 'overlap': 4, 'overlap_bi': 18}

parameters_AdEx = {'C': 200e-12, 'g_L': 12e-9, 'E_L': -70e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'V_T': -50e-3,
                   'tau_W': 600e-3, 'b': 8e-12, 'V_R': -70e-3, 'V_cut': -40e-3, 'refrac': 0.00,
                   'n': 2*parameters_hair_field['N_hairs'], 'dt': variables['dt']}

parameters_LIF_simple = {'tau': 2e-3, 'tau_G': 500e-3, 'G_r': 20e-3, 'p': 0.1, 'V_T': -50e-3,
                         'V_R': -70e-3, 'n': 2, 'N_input': parameters_hair_field['N_hairs'], 'dt': variables['dt']
                         , 'refrac': 0}

hair_field = HairField(parameters_hair_field)
hair_field.get_double_receptive_field()

adex = AdEx(parameters_AdEx)
adex.initialize_state()

lif = LIF_simple(parameters_LIF_simple)
lif.initialize_state()

hair_angles = hair_field.get_hair_angle(joint_angle)

time, spike_list, spike_inter = np.array([]), torch.empty(hair_angles.shape), np.empty([variables['N_steps'],
                                                                                        parameters_LIF_simple['n']])
for i in tqdm(range(variables['N_steps'])):
    _, spike_list[i, :] = adex.forward(hair_angles[i, :])
    reshaped_spikes = torch.reshape(spike_list[i, :], (parameters_LIF_simple['n'], (parameters_hair_field['N_hairs'])))
    _, spike_inter[i, :] = lif.forward(reshaped_spikes)
    time = np.append(time, i * parameters_AdEx['dt'])

plt.plot(time, hair_angles[:, 11].numpy())
plt.plot(time, joint_angle)
plt.show()
spike_inter[spike_inter == 0] = np.nan

fig, ax = plt.subplots()

plt.plot(time, joint_angle, color='black')
plt.scatter(time, joint_angle*spike_inter[:, 0], color='blue')
plt.scatter(time, joint_angle*spike_inter[:, 1], color='red')

plot_movement_interneuron_network(ax, fig)