from matplotlib.patches import Rectangle
from scipy import stats
from plots import *

a = np.exp(np.linspace(0, 10, num=100))
print(np.percentile(a, 25), np.percentile(a, 75))
