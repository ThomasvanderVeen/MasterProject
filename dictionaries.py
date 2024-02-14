class Parameters:
    def __init__(self, t_total, dt, n_angles=18, n_hairs=10, max_joint_angle=180, min_joint_angle=0):
        dt_camera = 1 / 200

        self.general = {'dt': dt, 't_total': t_total, 'N_steps': int(t_total / dt), 'dt_camera': dt_camera,
                        'N_frames': int(t_total / dt_camera), 'n_angles': n_angles,
                        'colors': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#FFDB58', '#a65628'],
                        'linestyles': ['-', '--', '-.', ':', '-', '--', ':', '-.'],
                        'markers': ['o', "^", "v", "*", "+", "x", "s", "p"]}

        self.hair_field = {'N_hairs': n_hairs, 'min_joint_angle': min_joint_angle,
                           'max_joint_angle': max_joint_angle, 'max_angle': 90, 'overlap': 0.1, 'overlap_bi': 0}

        self.sensory = {'C': 200e-12, 'g_L': 2e-9, 'E_L': -70e-3, 'DeltaT': 2e-3, 'a': 2e-9, 'V_T': -50e-3,
                        'tau_W': 50e-3, 'b': 264e-12, 'V_R': -70e-3, 'V_cut': -40e-3, 'refrac': 0.00,
                        'n': 2 * n_hairs * n_angles, 'dt': dt}

        self.position = {'E_L': -70e-3, 'V_T': -50e-3, 'tau': 25e-3, 'tau_W': 3e-3, 'tau_epsp': 6e-3, 'b': 16e-3,
                         'V_R': -70e-3, 'n': 2 * n_angles, 'N_input': n_hairs // 2, 'dt': dt, 'refrac': 0}

        self.velocity = {'tau': 1e-3, 'tau_G': 5e-3, 'tau_min': 3e-3, 'tau_plus': 1e-3, 'G_r': 17e-3, 'p': 0.01, 'V_T': -50e-3,
                         'V_R': -70e-3, 'V_h': -53e-3, 'n': 2 * n_angles, 'N_input': n_hairs,
                         'dt': dt, 'refrac': 0.006}

        self.primitive = {'tau': 2e-3, 'V_T': -50e-3, 'V_R': -70e-3, 'n': 0, 'w': 0,
                          'N_input': 3, 'dt': dt, 'refrac': 0}

        self.posture = {'tau': 2e-3, 'V_T': -50e-3, 'V_R': -70e-3, 'n': 2, 'w': 0,
                        'N_input': 672, 'dt': dt, 'refrac': 0}

        self.velocity_2 = {'tau': 5e-3, 'V_T': -50e-3, 'V_R': -70e-3, 'n': 36 * n_hairs, 'w': 11.7e-3,
                        'N_input': 1, 'dt': dt, 'refrac': 0}
