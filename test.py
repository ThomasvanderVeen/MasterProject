from class_velocity_neuron import LIF_simple
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools


a = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None])
b = ([0, 0])


encoding = ['none', 'upvel', 'downvel', 'dorsalLoc', 'ventralLoc']
encoding_2 = [-100, 0, 1, 2, 3]
allCombo = np.array(list(itertools.permutations(encoding, 3))) ##lists all movement primitive neurons combinations
synpType = []
for combo in allCombo:  ## determine synapse type -  number of inputs, and which ones
    if 'none' in combo:
        if len([s for s in combo if 'Loc' in s]) == 2:
            synpType.append(3)
        elif len([s for s in combo if 'vel' in s]) == 2:
            synpType.append(2)
        else:
            synpType.append(1)
    else:
        if len([s for s in combo if 'Loc' in s]) == 2:
            synpType.append(5)
        else:
            synpType.append(4)

allCombo = np.array(list(itertools.permutations(encoding_2, 3)))
extra = np.array([0, 4, 8])
extra = np.tile(extra, (allCombo.shape[0], 1))
#print(extra)
allCombo = allCombo.astype(int) + extra
allCombo[allCombo < 0] = 12
c = np.reshape(a[np.ndarray.flatten(allCombo)], (60, 3))
print(allCombo, c)

#print(a[np.ndarray.flatten(allCombo)])
