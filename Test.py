from class_velocity_neuron import LIF_simple
from dictionaries import Parameters
from functions import *
import matplotlib.pyplot as plt


A = np.zeros((100, 5))

A[[58, 59], 0] = 10

B = convert_to_bins(A, 5, 2, False)

print(B)

