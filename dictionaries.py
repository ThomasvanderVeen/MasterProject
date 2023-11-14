class Parameters:
    def __init__(self, t_total, dt, n_angles=18, n_hairs=10, max_joint_angle=180, min_joint_angle=0):

        dt_camera = 1/200

        self.general = {'dt': dt, 't_total': t_total, 'N_steps': int(t_total/dt), 'dt_camera': dt_camera,
                        'N_frames': int(t_total/dt_camera), 'N_sims':n_angles,
                        'colors': ['#c1272d', '#0000a7', '#eecc16', '#008176', '#b3b3b3']}

        self.hair_field = {'N_hairs': n_hairs, 'min_joint_angle': min_joint_angle,
                           'max_joint_angle': max_joint_angle, 'max_angle': 90, 'overlap': 0, 'overlap_bi': 0}

        self.sensory = {'C': 200e-12, 'g_L': 2e-9, 'E_L': -70e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'V_T': -50e-3,
                        'tau_W': 600e-3, 'b': 7e-12, 'V_R': -70e-3, 'V_cut': -40e-3, 'refrac': 0.00,
                        'n': 2 * n_hairs * n_angles, 'dt': dt}

        self.position = {'E_L': -70e-3, 'V_T': -50e-3, 'tau': 80e-3, 'tau_W': 5e-3, 'tau_epsp': 6e-3, 'b': 20e-3,
                         'V_R': -70e-3, 'n': 2 * n_angles, 'N_input': int(n_hairs/2) + 1, 'dt': dt, 'refrac': 0}

        self.velocity = {'tau': 2e-3, 'tau_G': 100e-3, 'G_r': 25e-3, 'p': 0.1, 'V_T': -50e-3,
                                      'V_R': -70e-3, 'n': 2 * n_angles, 'N_input': n_hairs,
                                      'dt': dt, 'refrac': 0}

        self.primitive = {'tau': 50e-3, 'V_T': -50e-3, 'V_R': -70e-3, 'n': 360, 'w': 0,
                          'N_input': 3, 'dt': dt, 'refrac': 0}

        self.posture = {'tau': 50e-3, 'V_T': -50e-3, 'V_R': -70e-3, 'n': 1, 'w': 0,
                          'N_input': 360, 'dt': dt, 'refrac': 0}
