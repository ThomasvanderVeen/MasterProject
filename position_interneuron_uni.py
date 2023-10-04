from HairField_class import HairField
from AdEx_class import AdEx
from LIF_class import LIF
from plots import *
from functions import *
import matplotlib.pyplot as plt

variables = {'dt': 0.0001, 't_total': 10, 'N_steps': None}
variables['N_steps'] = round(variables['t_total']/variables['dt'])

data = pickle_open('simulation_data')

joint_angle = data[f'simulation_0'][0][:1000]
joint_angle = interpolate(joint_angle, variables['t_total'], variables['N_steps'])

parameters_hair_field = {'N_hairs': 5, 'max_joint_angle': np.max(joint_angle), 'min_joint_angle': np.min(joint_angle),
                         'max_angle': 90, 'overlap': 4, 'overlap_bi': 18}

hair_field = HairField(parameters_hair_field)
hair_field.get_receptive_field()
hair_angles = hair_field.get_hair_angle(joint_angle)/37e9

parameters_AdEx = {'C': 200e-12, 'g_L': 12e-9, 'E_L': -70e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'V_T': -50e-3,
                   'tau_W': 600e-3, 'b': 8e-12, 'V_R': -58e-3, 'V_cut': 50e-3, 'refrac': 0.00, 'n': 5,
                   'dt': variables['dt']}

parameters_LIF = {'E_L': -70e-3, 'V_T': -50e-3, 'tau': 50e-3, 'tau_W': 10e-3, 'tau_epsp': 50e-3, 'b': 15e-3,
                  'V_R': -70e-3, 'n': 1, 'N_input': hair_field.N_hairs, 'dt': variables['dt'], 'refrac': 0}

neuron = AdEx(parameters_AdEx)
neuron.initialize_state()

lif = LIF(parameters_LIF)
lif.initialize_state()

voltage, time, spike_list = np.empty(hair_angles.numpy().shape), np.array([]), torch.empty(hair_angles.numpy().shape)
voltage_inter, spike_inter = np.empty([variables['N_steps']]), np.empty([variables['N_steps']])

for i in tqdm(range(variables['N_steps'])):
    voltage[i, :], spike_list[i, :] = neuron.forward(hair_angles[i, :])
    voltage_inter[i], spike_inter[i] = lif.forward(spike_list[i, :])
    time = np.append(time, i * variables['dt'])

print(np.sum(spike_inter))

firing_rate, spike_index = get_firing_rate(spike_inter, variables['dt'])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(time[spike_index], firing_rate, color='red')
ax2.plot(time, joint_angle, color='blue')
plot_position_interneuron(ax1, ax2, fig, 'uni')
