import numpy as np


a = np.full((2, 3, 4), 1)
b = np.sum(a, axis=0)
print(b)