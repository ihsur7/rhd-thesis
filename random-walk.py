import matplotlib
import numpy as np

dims = 3
steps_n = 1000
steps_set = [-1, 0, 1]

steps_shape = (steps_n, dims)
steps = np.random.choice(steps_set, size=steps_shape)