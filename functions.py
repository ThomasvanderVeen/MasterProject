import pickle
import torch
import numpy as np
from tqdm import tqdm
import itertools
import scipy.ndimage as img
import matplotlib.pyplot as plt
import os
import pandas as pd


def pickle_save(file, name):
    with open(name, 'wb') as f:
        pickle.dump(file, f)

    return


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


def interpolate(old_array, t_total, n_steps, boolean=False):
    if len(old_array.shape) == 1:
        old_array = np.reshape(old_array, (old_array.size, 1))

    new_array = np.zeros((n_steps, old_array.shape[1]))
    x_old = np.linspace(0, t_total, num=old_array.shape[0])
    x_new = np.linspace(0, t_total, num=n_steps)

    for i in range(old_array.shape[1]):
        new_array[:, i] = np.interp(x_new, x_old, old_array[:, i])
        if not boolean:
            new_array[:, i] = gaussian_filter(new_array[:, i], sigma=5)
        if boolean:
            new_array[:, i][new_array[:, i] > 0.5] = 1
            new_array[:, i][new_array[:, i] < 0.51] = 0

    if new_array.shape[1] == 1:
        return np.ndarray.flatten(new_array)
    else:
        return new_array


def define_and_initialize(class_handle, parameters):
    neuron = class_handle(parameters)
    neuron.initialize_state()

    return neuron


def get_primitive_indexes(n):
    permutations = np.array(list(itertools.permutations([-np.inf, 0, 1, 2, 3], 3)))
    extra = np.array([0, 4, 8])
    extra = np.tile(extra, (permutations.shape[0], 1))
    permutations = permutations.astype(int) + extra
    permutations[permutations < 0] = 0
    extra_2 = np.linspace(0, 12*(n-1), num=n)
    extra_2 = np.repeat(extra_2, 180)
    permutations = np.tile(np.ndarray.flatten(permutations), n) + extra_2

    return permutations


def get_encoding(w_pos, w_vel):
    encoding = ['none', 'Vel-', 'Vel+', 'Pos-', 'Pos+']
    permutations = np.array(list(itertools.permutations(encoding, 3)))

    synapse_type = []
    weights = np.zeros(permutations.shape)

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
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if 'Pos' in permutations[i, j]:
                weights[i, j] = w_pos[synapse_type[i]]
            elif 'Vel' in permutations[i, j]:
                weights[i, j] = w_vel[synapse_type[i]]

    encoding_filter = [1, 0, 0, 0, 0]
    negative_mask = np.array(list(itertools.permutations(encoding_filter, 3)))

    encoding_filter = [0, 1, 1, 1, 1]
    positive_mask = np.array(list(itertools.permutations(encoding_filter, 3)))

    negative_mask = np.tile(negative_mask, (6, 1))
    positive_mask = np.tile(positive_mask, (6, 1))
    weights = np.tile(weights, (6, 1))
    synapse_type = synapse_type*6

    return permutations, synapse_type, weights, negative_mask, positive_mask


def gaussian_filter(x, sigma=5):
    input_min = min(x)
    input_max = max(x)
    x = img.gaussian_filter1d(x, sigma)
    y = input_min + (((x-min(x))*(input_max-input_min))/(max(x)-min(x)))
    return y


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
    n_phases = int(change_index.size/2)-1
    swing_bin_rate, swing_bin_likelihood, stance_bin_rate, stance_bin_likelihood = \
        [np.zeros((n_phases, i)) for i in [10, 10, 20, 20]]

    for i in range(n_phases):
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


def prepare_spikes_primitive(spike_velocity, spike_position, permutations, mask):
    toepel = ()
    for i in range(18):
        toepel += (spike_velocity[[0 + 2*i, 1 + 2*i]], spike_position[[0 + 2*i, 1 + 2*i]])

    pos_vel_spikes = torch.concat(toepel)
    pos_vel_spikes = pos_vel_spikes[permutations].reshape((360, 3))*mask

    return pos_vel_spikes


def low_pass_filter(x, dt, tau):
    y = np.zeros(len(x))
    alpha = dt/(dt+tau)
    y[0] = alpha * x[0]
    for i in np.arange(1, len(x), 1):
        y[i] = alpha * x[i] + (1-alpha) * y[i-1]

    return y


def high_pass_filter(x, dt, tau):
    y = np.zeros(len(x))
    alpha = dt/(dt+tau)
    y[0] = x[0]
    for i in np.arange(1, len(x), 1):
        y[i] = alpha*(y[i-1] + x[i] - x[i-1])

    return y


def normalize(x, maximum=False, minimum=False, t_0=0, t_1=1):
    if not maximum:
        y = (x-np.min(x))/(np.max(x) - np.min(x))*(t_1-t_0)+t_0
    else:
        y = (x - maximum) / (maximum - minimum)*(t_1-t_0)+t_0

    return y


def get_firing_rate_2(spike_train, dt, t=0.5, sigma=5):
    n = int(t/dt)
    firing_rate = np.empty(spike_train.shape)
    for i in range(n, spike_train.size-n):
        firing_rate[n + i] = np.sum(spike_train[i:i+n])
    firing_rate[firing_rate < 2] = np.mean(firing_rate)
    firing_rate = gaussian_filter(firing_rate, sigma)
    return firing_rate
