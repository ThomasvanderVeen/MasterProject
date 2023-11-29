import pickle
import torch
import numpy as np
from tqdm import tqdm
from itertools import permutations
import scipy.ndimage as img
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os
import pandas as pd


def pickle_save(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def pickle_open(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    return data


def get_firing_rate(spike_train, dt):
    spike_index = np.where(spike_train == 1)[0]

    if len(spike_index) > 1:
        inter_spike_interval = np.diff(spike_index) * dt
        firing_rate = 1 / inter_spike_interval
        firing_rate = np.append(firing_rate, firing_rate[-1])
    else:
        firing_rate = np.array([0])

    return firing_rate, spike_index


def get_firing_rate_2(spike_train, dt, t=0.5, sigma=3):
    n = int(t / dt)

    firing_rate = np.convolve(spike_train, np.ones(n) / t, mode='same')
    firing_rate = gaussian_filter1d(firing_rate, sigma=sigma)

    firing_rate[firing_rate < 0.000001] = np.nan

    return firing_rate


def interpolate(old_array, t_total, n_steps, boolean=False):
    if old_array.ndim == 1:
        old_array = old_array.reshape((old_array.size, 1))

    new_array = np.zeros((n_steps, old_array.shape[1]))

    x_old = np.linspace(0, t_total, num=old_array.shape[0])
    x_new = np.linspace(0, t_total, num=n_steps)

    for i in range(old_array.shape[1]):
        new_array[:, i] = np.interp(x_new, x_old, old_array[:, i])
        if not boolean:
            new_array[:, i] = gaussian_filter(new_array[:, i], sigma=5)
        if boolean:
            new_array[:, i][new_array[:, i] > 0.5] = 1
            new_array[:, i][new_array[:, i] <= 0.50] = 0

    if new_array.shape[1] == 1:
        return np.ndarray.flatten(new_array)
    else:
        return new_array


def define_and_initialize(class_handle, parameters):
    instance = class_handle(parameters)
    instance.initialize_state()

    return instance


def get_primitive_indexes(n):
    base_perm = np.array(list(permutations([-np.inf, 0, 1, 2, 3], 3)))
    extra = np.array([0, 4, 8])
    extra = np.tile(extra, (base_perm.shape[0], 1))
    final_perm = (base_perm + extra).clip(min=0)

    extra_2 = np.linspace(0, 12 * (n - 1), num=n).repeat(180)
    final_perm = np.tile(final_perm.flatten(), n) + extra_2

    return final_perm.astype(int)


def get_encoding(w_pos, w_vel):
    encoding = ['none', 'Vel-', 'Vel+', 'Pos-', 'Pos+']
    perm = np.array(list(permutations(encoding, 3)))

    synapse_type = []
    weights = np.zeros_like(perm, dtype=float)

    for permutation in perm:

        if 'none' in permutation:
            synapse_type.append(2 if (permutation == 'Pos-').sum() + (permutation == 'Pos+').sum() == 2
                                else
                                (1 if (permutation == 'Vel+').sum() + (permutation == 'Vel-').sum() == 2
                                 else 0))
        else:
            synapse_type.append(4 if (permutation == 'Pos-').sum() + (permutation == 'Pos-').sum() == 2 else 3)

    for i, j in np.ndindex(weights.shape):
        if 'Pos' in perm[i, j]:
            weights[i, j] = w_pos[synapse_type[i]]
        elif 'Vel' in perm[i, j]:
            weights[i, j] = w_vel[synapse_type[i]]

    encoding_filter = [1, 0, 0, 0, 0]
    negative_mask = np.array(list(permutations(encoding_filter, 3)))

    encoding_filter = [0, 1, 1, 1, 1]
    positive_mask = np.array(list(permutations(encoding_filter, 3)))

    negative_mask = np.tile(negative_mask, (6, 1))
    positive_mask = np.tile(positive_mask, (6, 1))
    weights = np.tile(weights, (6, 1))
    synapse_type = synapse_type*6

    return perm, synapse_type, weights, negative_mask, positive_mask


def gaussian_filter(x, sigma=5):
    input_min = min(x)
    input_max = max(x)
    x = img.gaussian_filter1d(x, sigma)
    y = input_min + (((x-min(x))*(input_max-input_min))/(max(x)-min(x)))
    return y

def convert_to_bins(old_array, n_bins, sum_bool=False):
    old_array = old_array.T
    while old_array.shape[1] != n_bins:
        try:
            old_array = np.sum(old_array.reshape(old_array.shape[0], n_bins, int(old_array.shape[1] / n_bins)), axis=2)
        except:
            old_array = np.vstack((old_array.T, old_array[:, -1].T)).T
    if not sum_bool:
        old_array[old_array > 0] = 1

    return old_array.T


def get_stance_swing_bins(gait, spike_train):
    change_index = np.where(gait[:-1] != gait[1:])[0]
    n_phases = int(change_index.size/2)-1
    swing_bin_rate, swing_bin_likelihood, stance_bin_rate, stance_bin_likelihood = \
        [np.zeros((n_phases, i)) for i in [15, 15, 15, 15]]

    for i in range(n_phases):
        k = 0
        if gait[0] == 0:
            k = 1

        spikes_swing = np.array_split(spike_train[change_index[2*i+k]:change_index[1+2*i+k]], 15)
        spikes_stance = np.array_split(spike_train[change_index[1+2*i+k]:change_index[2+2*i+k]], 15)

        for j in range(15):
            swing_bin_rate[i, j] = np.sum(spikes_swing[j])

        for j in range(15):
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


def get_pitch(data, n_simulations, parameters):
    pitch = np.empty((parameters.general['N_steps'], n_simulations))
    for i in range(n_simulations):
        pitch_single = np.array(data[f'simulation_{i}'][2][1, :])
        pitch_single = pitch_single[:parameters.general['N_frames']]
        pitch[:, i] = interpolate(pitch_single, parameters.general['t_total'], parameters.general['N_steps'])

    pitch_max, pitch_min = np.max(pitch), np.min(pitch)
    pitch_middle = pitch_max / 2 + pitch_min / 2

    return pitch, pitch_max, pitch_min, pitch_middle


def get_indexes_legs(indexes_old):
    leg, indexes_new = [], []
    for i in range(indexes_old[0].size):
        index = indexes_old[0][i]
        j = 0
        while index > 59:
            j += 1
            index -= 60
        else:
            leg.append(j)
            indexes_new.append(index)
    return indexes_new, leg