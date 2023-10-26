import scipy.io
from functions import *

if os.name == 'nt':
    path = "C:\DOCUMENTEN\RUG\Master\Master Research Project\Kinematic_Data"
    print("Windows path used")
else:
    path = "/home/s3488926/Documents/master_project/drive/Kinematic_Data"
    print("Linux path used")

t_total = 6
N_frames = 200*t_total
data = {}
i = 0
legs = ['R1', 'R2', 'R3', 'L1', 'L2', 'L3']

animals_list = os.listdir(path)
for animal in animals_list:
    animal_path = f"{path}/{animal}"
    sessions_list = os.listdir(animal_path)
    for session in sessions_list:
        session_path = f"{animal_path}/{session}"
        file_list = os.listdir(session_path)
        for filename in file_list:
            file_path = f"{session_path}/{filename}"
            joint_angles, gaits, pitch = [], [], []
            try:
                simulation_file = scipy.io.loadmat(file_path)
                N = np.zeros(6)
                for j in range(6):
                    gait = np.ndarray.flatten(np.array([simulation_file['gait'][0][0][0][:, j]]))[:N_frames]
                    change_index = np.where(gait[:-1] != gait[1:])[0]
                    N[j] = int(change_index.size / 2) - 1
                gait = np.ndarray.flatten(np.array([simulation_file['gait'][0][0][0][:, 0]]))
                if gait.size > N_frames and np.min(N) > 1 and '_00_' in file_path:
                    for leg in range(6):
                        for joint in range(3):
                            if joint == 1:
                                angle_list = np.array(simulation_file[legs[leg]][0][0][joint + 2][0][0][2][:, 0] +
                                                      simulation_file[legs[leg]][0][0][joint + 1][0][0][2][:, 2])
                                joint_angles.append(angle_list)
                            else:
                                angle_list = np.array(simulation_file[legs[leg]][0][0][joint + 2][0][0][2][:, 0])
                                joint_angles.append(angle_list)
                        gait_list = np.ndarray.flatten(np.array([simulation_file['gait'][0][0][0][:, leg]]))
                        pitch = np.array([simulation_file['T3'][0][0][2][:, :3]])[0, :, :].T
                        gaits.append(gait_list)
                    data[f"simulation_{i}"] = [joint_angles, gaits, pitch]
                    i += 1
                else:
                    print(f'[1. skipped file {np.min(N)}]')
            except:
                print(f'[2. skipped file {file_path}]')

print(f'[{i} simulations saved]')
file = open('Data/simulation_data', 'wb')
pickle.dump(data, file)
file.close()
