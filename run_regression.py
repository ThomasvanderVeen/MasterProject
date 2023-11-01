from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.preprocessing import StandardScaler
from functions import *
import numpy as np
from dictionaries import Parameters

primitive_list = pickle_open('Data/primitive_list')
data = pickle_open('Data/simulation_data')
N_simulations = len(primitive_list)
parameters = Parameters(t_total=20, dt=0.001)
tau = [50e-3, 100e-3, 250e-3, 500e-3, 750e-3, 1000e-3, 1500e-3, 2000e-3, 2500e-3]
tau = [750e-3]

test, train = [], []
for t in tqdm(tau):
    primitive_list = pickle_open('Data/primitive_list')
    pitch_list = []
    for i in range(N_simulations):
        pitch = np.array(data[f'simulation_{i}'][2][1, :])
        pitch = pitch[:parameters.general['N_frames']]
        pitch_list.append(pitch)

    for i in range(N_simulations):
        primitive_list[i] = convert_to_bins(primitive_list[i], parameters.general['N_frames']).astype(float)
        for j in range(360):
            primitive_list[i][:, j] = low_pass_filter(primitive_list[i][:, j], 1/200, tau=t)

    model = SGD(alpha=0.000, max_iter=10000, verbose=100)

    x_train, x_test, y_train, y_test = train_test_split(primitive_list, pitch_list, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    scaler.fit(np.vstack(primitive_list))

    x_train = scaler.transform(np.vstack(x_train))
    x_test = scaler.transform(np.vstack(x_test))

    y_train = np.hstack(y_train)
    y_test = np.hstack(y_test)

    model.fit(x_train, y_train)
    score_train = model.score(x_train, y_train)
    score_test = model.score(x_test, y_test)

    train.append(score_train)
    test.append(score_test)

    print(score_train, score_test)

    y_predicted = model.predict(x_test)

plt.scatter(tau, train)
plt.scatter(tau, test)
plt.show()

plt.plot(range(y_predicted.size), gaussian_filter(y_predicted))
plt.plot(range(y_predicted.size), y_test)
plt.show()