import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

E_array = np.array([
                    [3.11074799, 3.231612906, 3.49612078, 3.395818561, 3.574984294],
                    [3.101399746, 3.593840932, 3.583519062, 3.432509393, 3.493042067],
                    [3.106757791, 3.295204051, 3.417707209, 3.507972937, 3.819293868],
                    [3.157631006, 3.498358303, 3.493396481, 3.505547483, 3.823644558],
                    [3.603378009, 3.629423336, 3.800265289, 3.885838633, 3.479162909]
                    ])

print(E_array)

plt.contourf(E_array, alpha=0.5)
plt.colorbar()
plt.savefig('contour-test-2.png', cmap='RdBu')
plt.show()

implot = 