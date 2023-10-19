import pickle
import torch
import numpy as np
from tqdm import tqdm
import itertools
import scipy.ndimage as img
import matplotlib.pyplot as plt

def pickle_open(file):
    file = open(file, 'rb')
    data = pickle.load(file)
    file.close()

    return data


def get_firing_rate(spike_train, dt):

    spike_index = np.where(spike_train == 1)[0]
    inter_spike_interval = np.diff(spike_index)*dt
    firing_rate = 1/inter_spike_interval
    try:
        firing_rate = np.append(firing_rate, firing_rate[-1])
    except:
        firing_rate = np.append(firing_rate, 0)

    return firing_rate, spike_index


def interpolate(joint_angle, t, n, gait_boolean=False):
    x = np.linspace(0, t, num=joint_angle.shape[0])
    xvals = np.linspace(0, t, num=n)
    joint_angle = np.interp(xvals, x, joint_angle)
    if not gait_boolean:
        joint_angle = smooth_function(joint_angle, sig=25)
    if gait_boolean:
        joint_angle[joint_angle > 0.5] = 1
        joint_angle[joint_angle < 0.51] = 0

    return joint_angle


def define_and_initialize(class_handle, parameters):
    neuron = class_handle(parameters)
    neuron.initialize_state()
    return neuron


def get_primitive_indexes(N):
    permutations = np.array(list(itertools.permutations([-np.inf, 0, 1, 2, 3], 3)))
    extra = np.array([0, 4, 8])
    extra = np.tile(extra, (permutations.shape[0], 1))
    permutations = permutations.astype(int) + extra
    permutations[permutations < 0] = 0

    return np.ndarray.flatten(permutations)


def get_encoding(w_pos, w_vel):
    encoding = ['none', 'Vel+', 'Vel-', 'Pos+', 'Pos-']
    permutations = np.array(list(itertools.permutations(encoding, 3)))
    synapse_type = []
    w = np.zeros(permutations.shape)

    for permutation in permutations:
        if 'none' in permutation:
            if len([s for s in permutation if 'Pos' in s]) == 2:
                synapse_type.append(2)
            elif len([s for s in permutation if 'Vel' in s]) == 2:
                synapse_type.append(1)
            else:
                synapse_type.append(0)
        else:
            if len([s for s in permutation if 'Pos' in s]) == 2:
                synapse_type.append(4)
            else:
                synapse_type.append(3)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if 'Pos' in permutations[i, j]:
                w[i, j] = w_pos[synapse_type[i]]
            elif 'Vel' in permutations[i, j]:
                w[i, j] = w_vel[synapse_type[i]]

    encoding_filter = [1, 0, 0, 0, 0]
    primitive_filter = np.array(list(itertools.permutations(encoding_filter, 3)))
    encoding_filter_2 = [0, 1, 1, 1, 1]
    primitive_filter_2 = np.array(list(itertools.permutations(encoding_filter_2, 3)))

    return permutations, synapse_type, w, primitive_filter, primitive_filter_2


def smooth_function(inp, sig = 5):
    minOrig = min(inp)
    maxOrig = max(inp)
    smoothed = img.gaussian_filter1d(inp, sig)
    smoothStretch = minOrig + (((smoothed-min(smoothed))*(maxOrig-minOrig))/(max(smoothed)-min(smoothed)))
    return smoothStretch


def convert_to_bins(old_array, n_bins, sum_bool=False):
    n_steps = old_array.shape[0]
    n_steps_bin = int(n_steps / n_bins)
    new_array = np.empty((n_bins, old_array.shape[1]), dtype=int)

    for j in range(new_array.shape[1]):
        for i in range(n_bins):
            elements = old_array[n_steps_bin*i:n_steps_bin*(i+1), j]
            elements[elements == 0] = False
            if np.any(elements):
                if sum_bool:
                    new_array[i, j] = np.sum(elements)/n_steps
                else:
                    new_array[i, j] = 1
            else:
                new_array[i, j] = 0

    return new_array


def get_stance_swing_bins(gait, spike_train):
    change_index = np.where(gait[:-1] != gait[1:])[0]

    N = int(change_index.size/2)-1
    swing_bin_rate, swing_bin_likelihood, stance_bin_rate, stance_bin_likelihood = np.zeros((N, 10)), np.zeros((N, 10)),\
                                                                                    np.zeros((N, 20)), np.zeros((N, 20))

    for i in range(N):
        k = 0
        if gait[0] == 0:
            k = 1

        spikes_swing = np.array_split(spike_train[change_index[2*i+k]:change_index[1+2*i+k]], 10)
        spikes_stance = np.array_split(spike_train[change_index[1+2*i+k]:change_index[2+2*i+k]], 20)

        for j in range(10):
            swing_bin_rate[i, j] = np.sum(spikes_swing[j])

        for j in range(20):
            stance_bin_rate[i, j] = np.sum(spikes_stance[j])

        stance_bin_likelihood[stance_bin_rate > 0.5] = 1
        swing_bin_likelihood[swing_bin_rate > 0.5] = 1

    swing_bin_rate = np.mean(swing_bin_rate, axis=0)
    stance_bin_rate = np.mean(stance_bin_rate, axis=0)
    swing_bin_likelihood = np.mean(swing_bin_likelihood, axis=0)
    stance_bin_likelihood = np.mean(stance_bin_likelihood, axis=0)

    return swing_bin_rate, stance_bin_rate, swing_bin_likelihood, stance_bin_likelihood