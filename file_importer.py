import os
import numpy as np
import scipy.io
import pickle

path = "/home/s3488926/Documents/master_project/drive/Kinematic_Data"
data = {}
i = 0
animals_list = os.listdir(path)
for animal in animals_list:
    animal_path = f"{path}/{animal}"
    sessions_list = os.listdir(animal_path)
    for session in sessions_list:
        session_path = f"{animal_path}/{session}"
        file_list = os.listdir(session_path)
        for filename in file_list:
            file_path = f"{session_path}/{filename}"
            simulation = []
            try:
                simulation_file = scipy.io.loadmat(file_path)
                for leg in ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']:
                    for joint in [0, 1, 2]:
                        angle_list = np.array(simulation_file[leg][0][0][joint + 2][0][0][2][:, 0])
                        simulation.append(angle_list)
                data[f"simulation_{i}"] = simulation
                i += 1
            except:
                print(f"skipped file: {file_path}")


file = open('simulation_data', 'wb')
pickle.dump(data, file)
file.close()

