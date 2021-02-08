import math
import numpy as np
# t = 1
# while t > 0:
#     conc_ratio = math.erfc(1/math.sqrt(4*67.6e-5*t))
#     if conc_ratio == 1:
#         print(t)
#         break
#     else:
#         t += 10000000
t = 1e8
conc_ratio = math.erfc((17*0.00714/math.sqrt(4*67.6e-5*2)))
print(conc_ratio)
# N = 4
# print(np.random.uniform(0, 1, size=(N, 3)))