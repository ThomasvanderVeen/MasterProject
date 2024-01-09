import itertools
import pickle
import torch
import numpy as np
from tqdm import tqdm
from tqdm.contrib import itertools
from itertools import product
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


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_firing_rate(spike_train, dt):
    spike_index = np.where(spike_train == 1)[0]

    if len(spike_index) > 1:
        inter_spike_interval = np.diff(spike_index) * dt
        firing_rate = 1 / inter_spike_interval
        firing_rate = np.append(firing_rate, firing_rate[-1])
    else:
        firing_rate = np.array([0])

    return firing_rate, spike_index


def get_firing_rate_2(spike_train, dt, t=0.5, sigma=3, nan_bool=True):
    n = int(t / dt)

    firing_rate = np.convolve(spike_train.astype(int), np.ones(n), mode='same') / t
    firing_rate = gaussian_filter1d(firing_rate, sigma=sigma)

    if nan_bool:
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
            new_array[:, i] = gaussian_filter1d(new_array[:, i], sigma=1)
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


def get_encoding(w_pos=[0, 0, 0, 0, 0, 0, 0], w_vel=[0, 0, 0, 0, 0, 0, 0], n=6):
    encoding = ['none', 'Vel-', 'Vel+', 'Pos-', 'Pos+']
    perm = np.array(list(product(encoding, repeat=3)))
    base_perm = np.array(list(product([-np.inf, 0, 1, 2, 3], repeat=3)))

    synapse_type = []

    for permutation in perm:

        if 'none' in permutation:
            synapse_type.append(0 if (permutation == 'Pos+').sum() + (permutation == 'Pos-').sum() == 2
                                else
                                (1 if (permutation == 'Vel+').sum() + (permutation == 'Vel-').sum() == 2
                                 else
                                 (2 if (permutation == 'Pos+').sum() + (permutation == 'Pos-').sum() == 1
                                       and (permutation == 'Vel+').sum() + (permutation == 'Vel-').sum() == 1
                                  else -1)))
        else:
            synapse_type.append(3 if (permutation == 'Pos+').sum() + (permutation == 'Pos-').sum() == 2
                                else
                                (4 if (permutation == 'Vel+').sum() + (permutation == 'Vel-').sum() == 2
                                 else
                                 (5 if (permutation == 'Pos+').sum() + (permutation == 'Pos-').sum() == 3
                                  else 6)))

    zero_index = np.where(np.array(synapse_type) == -1)[0]

    synapse_type = list(np.delete(synapse_type, zero_index))
    perm = np.delete(perm, zero_index, axis=0)
    base_perm = np.delete(base_perm, zero_index, axis=0)
    weights = np.zeros_like(perm, dtype=float)

    for i, j in np.ndindex(weights.shape):
        if 'Pos' in perm[i, j]:
            weights[i, j] = w_pos[synapse_type[i]]
        elif 'Vel' in perm[i, j]:
            weights[i, j] = w_vel[synapse_type[i]]

    negative_mask = np.zeros_like(perm, dtype=float)
    negative_mask[perm == 'none'] = 1
    negative_mask = np.tile(negative_mask, (6, 1))
    positive_mask = 1 - negative_mask

    weights = np.tile(weights, (6, 1))
    synapse_type = synapse_type * 6

    extra = np.array([0, 4, 8])
    extra = np.tile(extra, (base_perm.shape[0], 1))
    final_perm = (base_perm + extra).clip(min=0)

    extra_2 = np.linspace(0, 12 * (n - 1), num=n).repeat(3 * final_perm.shape[0])
    final_perm = (np.tile(final_perm.flatten(), n) + extra_2).astype(int)

    base_perm = base_perm + 1
    base_perm[base_perm == -np.inf] = 0

    return perm, synapse_type, weights, negative_mask, positive_mask, final_perm, base_perm.astype(int)


def convert_to_bins(old_array, n_bins, sum_bool=False):
    old_array = np.transpose(old_array)

    while old_array.shape[1] != n_bins:
        try:
            old_array = np.sum(old_array.reshape(old_array.shape[0], n_bins, -1), axis=2)
        except:
            old_array = np.column_stack((old_array, old_array[:, -1]))

    if not sum_bool:
        old_array = (old_array > 0).astype(int)

    return np.transpose(old_array)


def get_stance_swing_bins(gait, spike_train):
    change_index = np.where(gait[:-1] != gait[1:])[0]
    n_phases = (change_index.size // 2) - 1

    swing_bin_rate = np.zeros((n_phases, 15))
    stance_bin_rate = np.zeros((n_phases, 15))
    swing_bin_likelihood = np.zeros((n_phases, 15))
    stance_bin_likelihood = np.zeros((n_phases, 15))

    for i in range(n_phases):
        start_idx = change_index[2 * i + (1 if gait[0] == 0 else 0)]
        end_idx_swing = change_index[1 + 2 * i + (1 if gait[0] == 0 else 0)]
        end_idx_stance = change_index[2 + 2 * i + (1 if gait[0] == 0 else 0)]

        spikes_swing = np.array_split(spike_train[start_idx:end_idx_swing], 15)
        spikes_stance = np.array_split(spike_train[end_idx_swing:end_idx_stance], 15)

        for j in range(15):
            swing_bin_rate[i, j] = np.sum(spikes_swing[j])
            stance_bin_rate[i, j] = np.sum(spikes_stance[j])

        stance_bin_likelihood[i, stance_bin_rate[i, :] > 0.5] = 1
        swing_bin_likelihood[i, swing_bin_rate[i, :] > 0.5] = 1

    swing_bin_rate = np.mean(swing_bin_rate, axis=0)
    stance_bin_rate = np.mean(stance_bin_rate, axis=0)
    swing_bin_likelihood = np.mean(swing_bin_likelihood, axis=0)
    stance_bin_likelihood = np.mean(stance_bin_likelihood, axis=0)

    return swing_bin_rate, stance_bin_rate, swing_bin_likelihood, stance_bin_likelihood


def prepare_spikes_primitive(spike_velocity, spike_position, permutations, mask):
    toepel = ()
    for i in range(18):
        toepel += (spike_velocity[[0 + 2 * i, 1 + 2 * i]], spike_position[[0 + 2 * i, 1 + 2 * i]])

    pos_vel_spikes = torch.concat(toepel)
    pos_vel_spikes = pos_vel_spikes[permutations].reshape(mask.shape) * mask

    return pos_vel_spikes


def low_pass_filter(x, dt, tau):
    alpha = dt / (dt + tau)
    y = np.zeros_like(x)
    y[0] = alpha * x[0]

    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]

    return y


def get_pitch(data, n_simulations, parameters):
    pitch = np.empty((parameters.general['N_steps'], n_simulations))

    for i in range(n_simulations):
        pitch_single = data[f'simulation_{i}'][2][1, :parameters.general['N_frames']]
        pitch[:, i] = interpolate(pitch_single, parameters.general['t_total'], parameters.general['N_steps'])

    pitch_max, pitch_min = np.max(pitch), np.min(pitch)
    pitch_middle = pitch_max / 2 + pitch_min / 2

    return pitch, pitch_max, pitch_min, pitch_middle


def get_indexes_legs(indexes_old):
    leg = np.floor_divide(indexes_old, 112)
    indexes_new = np.mod(indexes_old, 112)
    return indexes_new, leg


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def matthews_correlation(tp, tn, fp, fn):
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    if denominator == 0:
        return 0  # Handle division by zero

    mcc = numerator / denominator
    return mcc


def fill_with_ones(tensor):
    # Find the maximum index containing a one for each column
    if tensor.sum() < 0.5:
        result_tensor = torch.zeros(tensor.shape)

    elif tensor.ndim == 1:
        x = torch.linspace(0, 0.1, steps=tensor.shape[0])
        max_index = torch.argmax(tensor + x)

        # Create a mask to set all values above the maximum index to one
        mask = torch.arange(tensor.shape[0]) < max_index

        # Use the mask to update the tensor
        result_tensor = torch.where(mask, 1, 0)
    else:
        x = torch.linspace(0, 0.1, steps=tensor.shape[1]).repeat(tensor.shape[0], 1)
        max_indices = torch.argmax(tensor, dim=1)

        # Create a mask to set all values above the maximum index to one
        mask = torch.arange(tensor.shape[1]).unsqueeze(0) < max_indices.unsqueeze(1)

        # Use the mask to update the tensor
        result_tensor = torch.where(mask, 1, 0)

    return result_tensor

