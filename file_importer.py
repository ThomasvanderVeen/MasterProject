import os
import pickle
import numpy as np
import scipy.io

DESKTOP = True
DATA_DIR = "Data"
SIMULATION_DATA_FILE = "simulation_data"
DESKTOP_PATH = r"C:\DOCUMENTEN\RUG\Master\Master Research Project\Kinematic_Data"
LAPTOP_PATH = r"C:\Users\thoma\Documents\RUG\Master Project\Kinematic_Data"
LINUX_PATH = r"/home/s3488926/Documents/master_project/drive/Kinematic_Data"
STEPS = '_00_'
T_TOTAL = 6
N_FRAMES = 200 * T_TOTAL
LEGS = ['R1', 'R2', 'R3', 'L1', 'L2', 'L3']


def process_simulation_file(file_path):
    joint_angles, gaits, pitch = [], [], []

    try:
        simulation_file = scipy.io.loadmat(file_path)

        N = np.zeros(len(LEGS))
        for j in range(len(LEGS)):
            gait = np.ndarray.flatten(np.array([simulation_file['gait'][0][0][0][:, j]]))[:N_FRAMES]
            change_index = np.where(gait[:-1] != gait[1:])[0]
            N[j] = int(change_index.size / 2) - 1
        gait = np.ndarray.flatten(np.array([simulation_file['gait'][0][0][0][:, 0]]))
        if gait.size > N_FRAMES and np.min(N) > 1 and STEPS in file_path:
            for leg in range(len(LEGS)):
                for joint in range(3):
                    if joint == 1:
                        angle_list = np.array(simulation_file[LEGS[leg]][0][0][joint + 2][0][0][2][:, 0] +
                                              simulation_file[LEGS[leg]][0][0][joint + 1][0][0][2][:, 2])
                        joint_angles.append(angle_list)
                    else:
                        angle_list = np.array(simulation_file[LEGS[leg]][0][0][joint + 2][0][0][2][:, 0])
                        joint_angles.append(angle_list)
                gait_list = np.ndarray.flatten(np.array([simulation_file['gait'][0][0][0][:, leg]]))
                pitch = np.array([simulation_file['T3'][0][0][2][:, :3]])[0, :, :].T
                gaits.append(gait_list)
            return joint_angles, gaits, pitch
        else:
            return None, None, None

    except:
        return None, None, None


def main():
    if os.name == 'nt':
        if DESKTOP:
            path = DESKTOP_PATH
            print("Windows desktop path used")
        else:
            path = LAPTOP_PATH
            print("Windows laptop path used")
    else:
        path = LINUX_PATH
        print("Linux path used")

    data = {}
    i = 0

    for animal in os.listdir(path):
        animal_path = os.path.join(path, animal)
        for session in os.listdir(animal_path):
            session_path = os.path.join(animal_path, session)
            for filename in os.listdir(session_path):
                file_path = os.path.join(session_path, filename)
                joint_angles, gaits, pitch = process_simulation_file(file_path)

                if joint_angles is not None:
                    data[f"simulation_{i}"] = [joint_angles, gaits, pitch]
                    i += 1

    print(f"{i} simulations saved")

    with open(os.path.join(DATA_DIR, SIMULATION_DATA_FILE), 'wb') as file:
        pickle.dump(data, file)


if __name__ == "__main__":
    main()
