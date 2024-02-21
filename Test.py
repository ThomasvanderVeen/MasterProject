from class_velocity_neuron import LIF_simple
from dictionaries import Parameters
from functions import *
import matplotlib.pyplot as plt
import scipy.stats as stats

swings = pickle_open('Data/swings')

print(swings.shape)

t_test_score = np.zeros((6, 6))
t_test_p = np.zeros((6, 6))

for i in range(6):
    for j in range(3):
        print(stats.ttest_rel(swings[j, 0, i, :], swings[j, 1, i, :]))
        vel = stats.ttest_rel(swings[j, 0, i, :], swings[j, 1, i, :])
        pos = stats.ttest_rel(swings[j, 2, i, :], swings[j, 3, i, :])
        t_test_score[j, i] = vel[0]
        t_test_score[j + 3, i] = pos[0]
        t_test_p[j, i] = vel[1]
        t_test_p[j + 3, i] = pos[1]

np.savetxt('Data/t_test_score.csv', t_test_score)
np.savetxt('Data/t_test_p.csv', t_test_p)

print(t_test_score)
print(t_test_p)