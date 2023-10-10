from dictionaries import Parameters
from class_velocity_neuron import LIF_simple
from class_primitive_neuron import LIF_primitive
from class_position_neuron import LIF
from class_sensory_neuron import AdEx
from class_hair_field import HairField
from plots import *
from functions import *

N_sims = 3

w_pos = [12e-3, 0, 13e-3, 11e-3, 8e-3]
w_vel = [13.5e-3, 13e-3, 0, 12.5e-3, 13.5e-3]

permutations, synapse_type, weights_primitive, primitive_filter, primitive_filter_2 = get_encoding(w_pos, w_vel)

data = pickle_open('simulation_data')

joint_angle = np.array(data[f'simulation_0'][:N_sims]).T

parameters = Parameters(max_joint_angle=np.amax(joint_angle, axis=0), min_joint_angle=np.amin(joint_angle, axis=0),
                        N_hairs=20, t_total=7.5, dt=0.001, N_sims=N_sims)
parameters.position['N_input'] = int(parameters.position['N_input']/2)
parameters.primitive['w'] = weights_primitive

joint_angle = joint_angle[:parameters.general['N_frames']]
joint_angles = np.empty((parameters.general['N_steps'], N_sims))
hair_angles = np.empty((parameters.general['N_steps'], 2*N_sims*parameters.hair_field['N_hairs']))

ground_truth = np.zeros([12, 7500])

for i in range(N_sims):
    joint_angles[:, i] = interpolate(joint_angle[:, i], parameters.general['t_total'], parameters.general['N_steps'])
    mid = np.max(joint_angles[:, i])/2 + np.min(joint_angles[:, i])/2
    diff = np.diff(joint_angles[:, i])
    ground_truth[0 + 4 * i, np.where(joint_angles[:, i] > mid)] = 1
    ground_truth[1 + 4 * i, np.where(joint_angles[:, i] < mid)] = 1
    ground_truth[2 + 4 * i, np.where(diff < 0)] = 1
    ground_truth[3 + 4 * i, np.where(diff > 0)] = 1

    #plt.plot(np.linspace(0, 10, num=7500), mid*ground_truth[0 + 4 * i, :])
    #plt.plot(np.linspace(0, 10, num=7500), mid*ground_truth[1 + 4 * i, :])
    #plt.plot(np.linspace(0, 10, num=7500), mid*ground_truth[2 + 4 * i, :])
    plt.plot(np.linspace(0, 10, num=7500), mid*ground_truth[3 + 4 * i, :])
    plt.plot(np.linspace(0, 10, num=7500), joint_angles[:, i])
    plt.show()

