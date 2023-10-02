import pickle
import numpy as np
import matplotlib.pyplot as plt
# open a file, where you stored the pickled data
file = open('simulation_data', 'rb')

# dump information to that file
data = pickle.load(file)

# close the file
file.close()
y = data[f'simulation_0'][0]
x = np.linspace(0, 10, num=y.shape[0])
xvals = np.linspace(0, 10, num=10000)

y = np.interp(xvals, x, y)

plt.plot(xvals, y)
plt.show()