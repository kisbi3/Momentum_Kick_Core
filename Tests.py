import time
time_start = time.time()

import numpy as np


x, y, z = np.meshgrid((0, 1, 2), (2, 3, 4), (4, 5, 6))

print(np.meshgrid((0, 1, 2), (2, 3, 4), (4, 5, 6)))


test1 = np.sum(np.meshgrid((0, 1, 2), (2, 3, 4), (4, 5, 6)), axis = 1)
test2 = np.sum(test1, axis = 0)

print(test1)
print(test2)

