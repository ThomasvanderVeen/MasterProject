import numpy as np


class Parameters:
    def __init__(self, max_joint_angle, min_joint_angle, dt, N_hairs):
        self.hair_field = {'N_hairs': N_hairs, 'min_joint_angle': min_joint_angle,
                           'max_joint_angle': max_joint_angle, 'max_angle': 90, 'overlap': 2, 'overlap_bi': 18}

        self.sensory = {'C': 200e-12, 'g_L': 12e-9, 'E_L': -70e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'V_T': -50e-3,
                                'tau_W': 600e-3, 'b': 8e-12, 'V_R': -70e-3, 'V_cut': -40e-3, 'refrac': 0.00,
                                'n': 2 * N_hairs, 'dt': dt}

        self.position = {'E_L': -70e-3, 'V_T': -50e-3, 'tau': 50e-3, 'tau_W': 10e-3, 'tau_epsp': 50e-3, 'b': 15e-3,
                               'V_R': -70e-3, 'n': 2, 'N_input': N_hairs, 'dt': dt, 'refrac': 0}

        self.velocity = {'tau': 5e-3, 'tau_G': 200e-3, 'G_r': 25e-3, 'p': 0.1, 'V_T': -50e-3,
                                      'V_R': -70e-3, 'n': 2, 'N_input': N_hairs,
                                      'dt': dt, 'refrac': 0}

        self.primitive = {'tau': 1e-3, 'V_T': -50e-3, 'V_R': -70e-3, 'n': 1, 'w': 15e-3,
                          'N_input': 2, 'dt': dt, 'refrac': 0}


