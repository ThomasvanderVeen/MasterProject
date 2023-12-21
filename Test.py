import numpy as np
import torch
from functions import *

tensor = torch.zeros(2)



new_tensor = fill_with_ones(tensor)

print(new_tensor)