from functions import *
from dictionaries import Parameters
import pandas as pd




data = pickle_open('Data/simulation_data')
velocity_list = pickle_open('Data/velocity_list')
joint_angles_list = pickle_open('Data/joint_angles_list')
parameters = Parameters(t_total=20, dt=0.001)


TP, FP, TN, FN = [], [], [], []
for i in range(len(joint_angles_list)):
    spike_velocity, joint_angles = velocity_list[i], joint_angles_list[i]
    joint_velocity = np.diff(joint_angles, axis=0)

    intersect_up = joint_velocity * spike_velocity[1:, 1::2]
    intersect_down = joint_velocity * spike_velocity[1:, 0::2]
    TP.append(intersect_up[intersect_up > 0].size)
    FP.append(intersect_up[intersect_up < 0].size)
    TN.append(intersect_down[intersect_down < 0].size)
    FN.append(intersect_down[intersect_down > 0].size)

TP, FP, TN, FN = sum(TP), sum(FP), sum(TN), sum(FN)
ACC = np.around((TP + TN) / (TP + TN + FP + FN), 3)
TPR = np.around(TP/(TP + FN), 3)
TNR = np.around(TN/(TN + FP), 3)

print(f'[Velocity Interneuron] true positive: {TP}, false Positive: {FP}, true negative: {TN}, false negative: {FN}')
print(f'[Velocity Interneuron] Accuracy: {ACC}, true positive rate: {TPR}, true negative rate: {TNR}')

table = {'col1': [TP, FP, TN, FN, ACC, TPR, TNR]}
df = pd.DataFrame(data=table, index=['TP', 'FP', 'TN', 'FN', 'ACC', 'TPR', 'TNR'])
df.to_csv("Images/velocity_table.csv")

print(f'[Velocity Interneuron] Data written to "velocity_table.csv"')