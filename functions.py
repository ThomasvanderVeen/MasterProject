import pickle
import torch
import numpy as np
from tqdm import tqdm

def pickle_open(file):

    file = open(file, 'rb')
    data = pickle.load(file)
    file.close()

    return data


def get_firing_rate(spike_train, dt):

    spike_index = np.where(spike_train == 1)[0]
    inter_spike_interval = np.diff(spike_index)*dt
    firing_rate = 1/inter_spike_interval
    firing_rate = np.append(firing_rate, firing_rate[-1])

    return firing_rate, spike_index


def interpolate(joint_angle, t, n):
    x = np.linspace(0, t, num=joint_angle.shape[0])
    xvals = np.linspace(0, t, num=n)
    joint_angle = np.interp(xvals, x, joint_angle)

    return joint_angle


def define_and_initialize(class_handle, parameters):
    neuron = class_handle(parameters)
    neuron.initialize_state()
    return neuron



