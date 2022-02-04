import numpy as np
import vedo

x,y,z = np.mgrid[:3, :3, :3]
# print('x = \n', x)
# print('y = \n', y)
# print('z = \n', z)
scalar = (x+y+z)
print(scalar)
vol = vedo.Volume()
lego = vol.legosurface()
vedo.show(lego, axes=True)