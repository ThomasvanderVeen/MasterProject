import numpy as np
import pickle
from functions import *

velocity_list = pickle_open('Data/velocity_list')
position_list = pickle_open('Data/position_list')
spike_velocity = velocity_list[0]
spike_position = position_list[0]


firing_rate, spike_index = get_firing_rate(spike_velocity[:, 0], 0.001)
firing_rate_2, spike_index_2 = get_firing_rate(spike_velocity[:, 1], 0.001)
x = np.linspace(0, 5, num=5000)

plt.scatter(x[spike_index], firing_rate)
plt.scatter(x[spike_index_2], firing_rate_2)
plt.show()

print(spike_velocity.shape)
print(np.sum(spike_velocity))
print(np.sum(spike_position))

'''

p_1 = np.random.choice([0, 1], size=10000, p=[.5, .5])
p_2 = np.random.choice([0, 1], size=10000, p=[.5, .5])
p_3 = np.random.choice([0, 1], size=10000, p=[.5, .5])

p_combined_list = []

p_combined = p_1 + p_2 + p_3
p_combined[p_combined < 2.5] = 0
p_combined[p_combined > 2.5] = 1

p_combined_list.append(p_combined)

p_combined_2 = np.zeros(10000)

p_combined_list.append(p_combined_2)

for p_combined in p_combined_list:
    p_truth = p_1 + p_2 + p_3

    p_truth[p_truth > 0.5] = 1

    intersect = p_truth + p_combined
    difference = p_truth - p_combined

    true_positive = intersect[intersect > 1.5].size
    false_positive = difference[difference > 0.5].size
    true_negative = intersect[intersect < 0.5].size
    false_negative = difference[difference < -0.5].size

    TPR = true_positive / (true_positive + false_negative + 0.0000001)
    TNR = true_negative / (true_negative + false_positive + 0.0000001)
    PPV = true_positive / (true_positive + false_positive + 0.0000001)
    NPV = true_negative / (true_negative + false_negative + 0.0000001)
    FNR = 1 - TPR
    FPR = 1 - TNR
    FDR = 1 - PPV
    FOR = 1 - NPV

    ACC_balanced = (TPR + TNR) / 2
    F1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative + 0.0000001)
    MCC = np.sqrt(TPR * TNR * PPV * NPV) - np.sqrt(FNR * FPR * FOR * FDR)


    print(ACC_balanced, F1, MCC)
    
'''
