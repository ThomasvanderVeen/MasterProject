from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.preprocessing import StandardScaler
from functions import *
import numpy as np
from dictionaries import Parameters
import matplotlib

primitive_list = pickle_open('Data/primitive_list')
data = pickle_open('Data/simulation_data')
N_simulations = len(primitive_list)
parameters = Parameters(t_total=6, dt=0.0001)
tau = np.array([50e-3, 100e-3, 250e-3, 500e-3, 750e-3, 1000e-3, 1500e-3, 2000e-3, 2500e-3])
tau = np.array([750e-3])

interneuron_list, test, train = [], [], []
for t in tqdm(tau):
    primitive_list = pickle_open('Data/primitive_list')
    position_list = pickle_open('Data/position_list')
    velocity_list = pickle_open('Data/velocity_list')
    for k in range(len(position_list)):
        interneuron_list.append(np.hstack((position_list[k], velocity_list[k])))
    pitch_list = []
    for i in range(N_simulations):
        pitch = np.array(data[f'simulation_{i}'][2][1, :])
        pitch = pitch[:parameters.general['N_frames']]
        pitch_list.append(pitch)

    for i in range(N_simulations):
        primitive_list[i] = convert_to_bins(primitive_list[i], parameters.general['N_frames']).astype(float)
        interneuron_list[i] = convert_to_bins(interneuron_list[i], parameters.general['N_frames']).astype(float)
        for j in range(360):
            primitive_list[i][:, j] = low_pass_filter(primitive_list[i][:, j], 1/200, tau=t)
        for j in range(position_list[0].shape[1]):
            interneuron_list[i][:, j] = low_pass_filter(interneuron_list[i][:, j], 1/200, tau=t)

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

    y_predicted_train = model.predict(x_train)
    y_predicted_test = model.predict(x_test)

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

fig, ax = plt.subplots(figsize=(1.5*3.54, 3.54), dpi=600)

ax.plot(tau*1000, train, color='blue')
ax.plot(tau*1000, test, color='red')
ax.scatter(tau*1000, train, color='blue')
ax.scatter(tau*1000, test, color='red')
ax.set_xlabel('tau (ms)', fontsize=15)
ax.set_ylabel('accuracy', fontsize=15)
ax.legend(['train', 'test'])

fig.savefig('Images/regression_tau', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(1.5*3.54, 3.54), dpi=600)

plt.plot(range(y_predicted_test.size), gaussian_filter(y_predicted_test), color='blue')
plt.plot(range(y_predicted_test.size), y_test, color='red')
ax.set_xlabel('tau (ms)', fontsize=15)
ax.set_ylabel('accuracy', fontsize=15)
ax.legend(['prediction', 'ground truth'])

fig.savefig('Images/regression_test', bbox_inches='tight')


