from functions import *

permutations = get_primitive_indexes(3)

data = pickle_open('simulation_data')


for i in range(15):

    pitch = np.array(data[f'simulation_{i}'][2])

    print(pitch)

    plt.plot(range(pitch.size), pitch)
plt.show()