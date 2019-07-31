import numpy as np
import scipy as sp
import scipy.ndimage as spimage


class DistanceTransform:
    def __init__(self, data, pixel_size = 1):
        self.data = data
        self.pixel_size = pixel_size

    def transform(self):
        dt = spimage.distance_transform_edt(self.data)
        if self.pixel_size != 1:
            dt_new = dt * self.pixel_size
            return dt_new
        else:
            return dt


class DistanceRidge:
    def __init__(self, data):
        self.data = data

    def get_dims(self):
        if self.data.ndim == 2:
            w, h = self.data.shape
            d = 1
            return w, h, d
        elif self.data.ndim == 3:
            w, h, d = self.data.shape
            return w, h, d

    def get_circle(self, radius):
        disk = np.ogrid(radius)
        return disk

    def ridge(self):
        # Scan points on distance map. For each point:
        #     Scan neighbouring points
        #         Use a template to evaluate the point and the neighboring point
        #             If the neighboring point does not "own" any more points than the scan point, based on a template,
        #             then delete the neighbor point.
        # Get Dimensions
        dims = self.get_dims()
        w = dims[0]
        h = dims[1]
        d = dims[2]
        
        # Create reference to input data
        s = np.empty(w, h, d, dtype=float)
        sNew = np.empty(w, h, d, dtype=float)
        for k in range(d):
            s[k] = self.data.flatten()
            sNew[k] = float(self.data.flatten())
        # get largest distance in the data
        dmax = 0
        for j in range(h):
            for i in range(w):
                ind = i + w * j
                if s[ind] > dmax:
                    dmax = s[ind]
        print(dmax)
        rSqMax = int((dmax ** 2) + 1)
        occurs = [False] * rSqMax
        for j in range(h):
            for i in range(w):
                ind = i + w * j
                occurs[int(s[ind] * s[ind])] = True
        num_radii = 0
        for i in range(rSqMax):
            if occurs[i]:
                num_radii += 1
        # Make an index of the distance-squared values
        distSqIndex = [int(i) for i in range(rSqMax)]
        distSqValues = [int(i) for i in range(num_radii)]
        indDS = 0
        for i in range(rSqMax):
            if occurs[i]:
                distSqIndex[i] = indDS
                newindDS = indDS + 1
                distSqValues[newindDS] = i
        print(distSqIndex)
        print(distSqValues)
        print(num_radii)
        print(occurs)
        # Build template
        # The first index of the template is the number of non-zero components
        # in the offset from the test point to the remote point. The second
        # index is the radii index (of the test point). The value of the template
        # is the minimum square radius of the remote point required to cover the
        # ball of the test point.
        rSqTemplate = createTemplate(distSqValues) # ! Need to make the createTemplate method
        dx = None
        dy = None
        dz = -1
        for j in range(h):
            for i in range(w):
                ind = i + w * j
                if s[ind] > 0:
                    notRidgePoint = False
                    sk0Sq = int(s[ind] * s[ind])
                    sk0SqInd = distSqIndex[sk0Sq]
                    while dz <= 1:
                        dz += 1


if __name__ == "__main__":
    im = np.array([[0, 1, 0, 0, 1],
                   [1, 0, 1, 1, 0],
                   [0, 1, 1, 0, 1],
                   [1, 0, 0, 0, 1],
                   [0, 0, 1, 1, 1]])
    dtrans = DistanceTransform(im).transform()
    rid = DistanceRidge(dtrans).ridge()
