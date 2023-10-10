import numpy as np


class Parameters:
    def __init__(self, max_joint_angle, min_joint_angle, N_hairs, t_total, dt, N_sims):

        dt_camera = 1/200

        self.general = {'dt': dt, 't_total': t_total, 'N_steps': int(t_total/dt), 'dt_camera': dt_camera,
                                  'N_frames': int(t_total/dt_camera), 'N_sims':N_sims}

        self.hair_field = {'N_hairs': N_hairs, 'min_joint_angle': min_joint_angle,
                           'max_joint_angle': max_joint_angle, 'max_angle': 90, 'overlap': 2, 'overlap_bi': 4}

        self.sensory = {'C': 200e-12, 'g_L': 12e-9, 'E_L': -70e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'V_T': -50e-3,
                                'tau_W': 600e-3, 'b': 8e-12, 'V_R': -70e-3, 'V_cut': -40e-3, 'refrac': 0.00,
                                'n': 2 * N_hairs * N_sims, 'dt': dt}

        self.position = {'E_L': -70e-3, 'V_T': -50e-3, 'tau': 50e-3, 'tau_W': 10e-3, 'tau_epsp': 50e-3, 'b': 10e-3,
                               'V_R': -70e-3, 'n': 2 * N_sims, 'N_input': N_hairs, 'dt': dt, 'refrac': 0}

        self.velocity = {'tau': 5e-3, 'tau_G': 200e-3, 'G_r': 25e-3, 'p': 0.1, 'V_T': -50e-3,
                                      'V_R': -70e-3, 'n': 2 * N_sims, 'N_input': N_hairs,
                                      'dt': dt, 'refrac': 0}

        self.primitive = {'tau': 1e-4, 'V_T': -50e-3, 'V_R': -70e-3, 'n': 60, 'w': [[8e-3, 8e-3, 8-3]],
                          'N_input': 3, 'dt': dt, 'refrac': 0}



