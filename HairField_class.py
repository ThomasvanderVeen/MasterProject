import numpy as np
import torch
from plots import *


class HairField:
    def __init__(self, parameters):
        self.N_hairs = parameters['N_hairs']
        self.max_joint_angle = parameters['max_joint_angle']
        self.min_joint_angle = parameters['min_joint_angle']
        self.max_angle = parameters['max_angle']
        self.overlap = parameters['overlap']
        self.receptive_field = None

    def get_receptive_field(self):
        rf = (self.max_joint_angle - self.min_joint_angle + (self.N_hairs + 2)*self.overlap) / self.N_hairs

        receptive_min = np.linspace(self.min_joint_angle + rf - 2 * self.overlap, self.max_joint_angle - 2 * rf + 2 * self.overlap,
                                    num=self.N_hairs - 2)
        receptive_max = np.linspace(self.min_joint_angle + 2 * rf - 2 * self.overlap, self.max_joint_angle - rf + 2 * self.overlap,
                                    num=self.N_hairs - 2)

        receptive_min = np.append(self.min_joint_angle, receptive_min)
        receptive_min = np.append(receptive_min, self.max_joint_angle - rf)
        receptive_max = np.append(receptive_max, self.max_joint_angle)
        receptive_max = np.append(self.min_joint_angle + rf, receptive_max)

        self.receptive_field = np.stack((receptive_min, receptive_max))

    def get_hair_angle(self, x):
        min_rf = self.receptive_field[0, :]
        slope = self.max_angle / (self.receptive_field[1, :] - self.receptive_field[0, :])

        slope, min_rf, x = np.tile(slope, (x.size, 1)), np.tile(min_rf, (x.size, 1)), np.tile(x, (self.N_hairs, 1)).T

        out = torch.relu(torch.from_numpy(slope * (x - self.receptive_field[0, :])))
        out[out > 90] = 90

        return out


parameters_hair_field = {'N_hairs': 10, 'max_joint_angle': 180, 'min_joint_angle': 0, 'max_angle': 90, 'overlap': 4}

joint_angle = np.linspace(0, 180, num=1200)
hair_field = HairField(parameters_hair_field)
hair_field.get_receptive_field()
hair_angles = hair_field.get_hair_angle(joint_angle)

plt.plot(joint_angle, hair_angles, color='blue')
for i in range(hair_field.N_hairs - 1):
    plt.fill_between([hair_field.receptive_field[0, 1+i], hair_field.receptive_field[1, i]], [90, 90], color='grey', alpha=0.35)

plot_hair_field(plt.gca(), 'uni')

parameters_hair_field_1 = {'N_hairs': 5, 'max_joint_angle': 180, 'min_joint_angle': 70, 'max_angle': 90, 'overlap': 8}
hair_field = HairField(parameters_hair_field_1)
hair_field.get_receptive_field()
hair_field_2 = HairField(parameters_hair_field_1)
hair_field_2.receptive_field = parameters_hair_field_1['max_joint_angle']-hair_field.receptive_field
hair_angles_1 = hair_field.get_hair_angle(joint_angle)
hair_angles_2 = hair_field_2.get_hair_angle(joint_angle)

hair_angles_bi = np.hstack((hair_angles_1, hair_angles_2))
plt.plot(joint_angle, hair_angles_1, color='red')
plt.plot(joint_angle, hair_angles_2, color='blue')
for i in range(hair_field.N_hairs - 1):
    plt.fill_between([hair_field.receptive_field[0, 1+i], hair_field.receptive_field[1, i]], [90, 90], color='red', alpha=0.25)
    plt.fill_between([hair_field_2.receptive_field[0, 1+i], hair_field_2.receptive_field[1, i]], [90, 90], color='blue', alpha=0.25)

plot_hair_field(plt.gca(), 'bi')
