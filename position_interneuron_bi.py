from HairField_class import HairField
from AdEx_class import AdEx
from LIF_class import LIF
from plots import *
import matplotlib.pyplot as plt
from functions import *

variables = {'dt': 0.0001, 't_total': 10, 'N_steps': None}
variables['N_steps'] = round(variables['t_total']/variables['dt'])

data = pickle_open('simulation_data')

joint_angle = data[f'simulation_0'][0][:1000]
joint_angle = interpolate(joint_angle, variables['t_total'], variables['N_steps'])

parameters_hair_field = {'N_hairs': 12, 'max_joint_angle': np.max(joint_angle), 'min_joint_angle': np.min(joint_angle),
                         'max_angle': 90, 'overlap': 1, 'overlap_bi': 5}

parameters_AdEx = {'C': 200e-12, 'g_L': 12e-9, 'E_L': -70e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'V_T': -50e-3,
                   'tau_W': 600e-3, 'b': 8e-12, 'V_R': -70e-3, 'V_cut': -40e-3, 'refrac': 0.00,
                   'n': parameters_hair_field['N_hairs']*2, 'dt': variables['dt']}

parameters_LIF = {'E_L': -70e-3, 'V_T': -50e-3, 'tau': 50e-3, 'tau_W': 10e-3, 'tau_epsp': 50e-3, 'b': 15e-3,
                  'V_R': -70e-3, 'n': 2, 'N_input': parameters_hair_field['N_hairs'], 'dt': variables['dt'],
                  'refrac': 0}

hair_field = HairField(parameters_hair_field)
hair_field.get_binary_receptive_field()

neuron = AdEx(parameters_AdEx)
neuron.initialize_state()

lif = LIF(parameters_LIF)
lif.initialize_state()

hair_angles = hair_field.get_hair_angle(joint_angle)/37e9

time, spike_list = np.array([]), torch.empty(hair_angles.shape)
spike_inter = np.empty([variables['N_steps'], parameters_LIF['n']])

for i in tqdm(range(variables['N_steps'])):
    _, spike_list[i, :] = neuron.forward(hair_angles[i, :])
    reshaped_spikes = torch.reshape(spike_list[i, :], (parameters_LIF['n'], (parameters_hair_field['N_hairs'])))
    _, spike_inter[i, :] = lif.forward(reshaped_spikes)
    time = np.append(time, i * parameters_AdEx['dt'])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

for j in range(parameters_LIF['n']):
    firing_rate, spikes_index = get_firing_rate(spike_inter[:, j], variables['dt'])

    ax1.plot(time[spikes_index], firing_rate)

ax2.plot(time, joint_angle, color='black')
ax2.plot(time, np.full(time.shape, np.max(joint_angle)/2 + np.min(joint_angle)/2), linestyle='dotted', color='black')
plot_position_interneuron(ax1, ax2, fig, 'bi')

fig2, ax3 = plt.subplots()
ax4 = ax3.twinx()

for i in range(hair_field.N_hairs):
    height = np.mean(hair_field.receptive_field[:, i])
    spike_list[spike_list == 0] = np.nan
    ax3.scatter(time, i*spike_list[:, i], color='dodgerblue', s=1)
ax4.plot(time, joint_angle, color='black')
ax4.plot(time, np.full(time.shape, np.max(joint_angle)/2 + np.min(joint_angle)/2), linestyle='dotted', color='red')
plot_spike_timing(ax3, ax4, fig2, hair_field.N_hairs)

