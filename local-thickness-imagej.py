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
        w, h = self.data.shape
        return w, h

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
        val_list = np.nonzero(self.data)
        # get largest distance in the data
        dmax = np.amax(self.data)
        rSqMax = int((dmax ** 2) + 1)
        occurs = [False] * rSqMax
        print(occurs)
        for j in range(dims[1]):
            for i in range(dims[0]):
                ind = i + dims[0] * j
                occurs[int(ind * ind)] = True

        print(occurs)


if __name__ == "__main__":
    im = np.array([[0, 1, 0, 0, 1],
                   [1, 0, 1, 1, 0],
                   [0, 1, 1, 0, 1],
                   [1, 0, 0, 0, 1],
                   [0, 0, 1, 1, 1]])
    dtrans = DistanceTransform(im).transform()
    rid = DistanceRidge(dtrans).ridge()
