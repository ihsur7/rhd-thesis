import numpy as np
import scipy as sp
import scipy.ndimage as spimage
import sys


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
        if d == 1:
            stack = self.data[..., np.newaxis]
            newStack = np.zeros(shape=(w,h), dtype=float)
        else:
            stack = self.data
            newStack = np.zeros(shape=(w,h,d), dtype=float)
        print(newStack)
        # print((i for i in range(d)))
        # print(stack.shape)
        # Create 32bit floating point stack for output, s. Will also use it for g
        # in Transformation 1
        sNew = np.zeros(shape=(d, w*h), dtype=np.float32)
        # print(sNew)
        for k in range(d):
            sNew[k] = np.float32(stack[:,:,k].flatten())
        # print(sNew)
        # Create reference to input data
        s = np.empty(shape=(d, w*h), dtype=np.float32)
        for k in range(d):
            s[k] = np.float32(stack[:,:,k].flatten())
        # print(s)
        # Find the largest distance in the data
        distMax = 0
        # print(i for i in range(d))
        for k in range(d):
            sk = s[k]
            # print('sk', sk)
            for j in range(h):
                for i in range(w):
                    ind = i + w * j
                    if sk[ind] > distMax:
                        distMax = sk[ind]
        # print(distMax)
        rSqMax = int((distMax * distMax + 0.5) + 1)
        occurs = [False] * rSqMax
        # print(occurs)
        for k in range(d):
            sk = s[k]
            for j in range(h):
                for i in range(w):
                    ind = i + w * j
                    # print(int(sk[ind] * sk[ind] + 0.5))
                    occurs[int(sk[ind] * sk[ind] + 0.5)] = True
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
        # print(distSqIndex)
        # print(distSqValues)
        # print(num_radii)
        # print(occurs)
        # Build template
        # The first index of the template is the number of non-zero components
        # in the offset from the test point to the remote point. The second
        # index is the radii index (of the test point). The value of the template
        # is the minimum square radius of the remote point required to cover the
        # ball of the test point.
        rSqTemplate = self.createTemplate(distSqValues) # ! Need to make the createTemplate method
        # dx = None
        # dy = None
        # dz = -1
        k = 0
        while k < d:
            sk = s[k]
            skNew = sNew[k]
            j = 0
            while j < h:
                i = 0
                while i < w:
                    ind = i + w * j
                    if sk[ind] > 0:
                        notRidgePoint = False
                        sk0Sq = int(sk[ind] * sk[ind] + 0.5)
                        sk0SqInd = distSqIndex[sk0Sq]
                        dz = -1
                        while dz <= 1:
                            k1 = k + dz
                            if 0 <= k1 < d: # if k1 >= 0 & k1 < d:
                                # print(k1, d)
                                # print(s, k1)
                                sk1 = s[k1]
                                if dz == 0:
                                    numCompZ = 0
                                else:
                                    numCompZ = 1
                                dy = -1
                                while dy <= 1:
                                    j1 = j + dy
                                    if 0 <= j1 < h:# if j1 >= 0 & j1 < h:
                                        if dy == 0:
                                            numCompY = 0
                                        else:
                                            numCompY = 1
                                        dx = -1
                                        while dx <= 1:
                                            i1 = i + dx
                                            if 0 <= i1 < w: # if i1 >= 0 & i1 < w:
                                                if dx == 0:
                                                    numCompX = 0
                                                else:
                                                    numCompX = 1
                                                numComp = numCompX + numCompY + numCompZ
                                                if numComp > 0:
                                                    sk1Sq = int(sk1[i1 + w * j1] *\
                                                        sk1[i1 + w * j1] + 0.5)
                                                    if sk1Sq >= rSqTemplate[numComp - 1]\
                                                        [sk0SqInd]:
                                                        notRidgePoint = True
                                            dx += 1
                                            if notRidgePoint:
                                                break
                                    dy += 1
                                    if notRidgePoint:
                                        break
                            dz += 1
                            if notRidgePoint:
                                break
                        if not notRidgePoint:
                            skNew[ind] = sk[ind]
                    i += 1
                j += 1
            k += 1
        print(rSqTemplate)
        setMin = 0
        setMax = distMax
        print(setMax)
        return rSqTemplate
        
    
    # For each offset from the origin, (dx, dy, dz), and each radius-squared,
    # rSq, find the smallest radius-squared, r1Squared, such that a ball
    # of radius r1 centered at (dx, dy, dz) include a ball of radius
    # rSq centered at the origin. These balls refer to a 3D integer grid.
    # The set of (dx, dy, dz) points considered is a cube center at the origin.
    # The size of the computed array could be considerably reduced by summetry,
    # but then the time for the calculation using this array would increase
    # (and more code would be needed).
    def createTemplate(self, distSqValues):
        t = np.empty(shape=(3, 3))
        t[0] = self.scanCube(1, 0, 0, distSqValues)
        t[1] = self.scanCube(1, 1, 0, distSqValues)
        t[2] = self.scanCube(1, 1, 1, distSqValues)
        return t
    
    def scanCube(self, dx, dy, dz, distSqValues):
        numRadii = len(distSqValues)
        r1Sq = [0] * numRadii
        if dx == 0 & dy == 0 & dz == 0:
            rSq = 0
            while rSq < numRadii:
                r1Sq[rSq] = sys.maxsize
                rSq += 1
        else:
            dxAbs = -abs(dx)
            dyAbs = -abs(dy)
            dzAbs = -abs(dz)
            rSqInd = 0
            while rSqInd < numRadii:
                rSq = distSqValues[rSqInd]
                mmax = 0
                r = 1 + (rSq ** 0.5)
                k = 0
                while k <= r:
                    scank = k * k
                    dk = (k - dzAbs) * (k - dzAbs)
                    j = 0
                    while j <= r:
                        scankj = scank + j * j
                        if scankj <= rSq:
                            iPlus = (int((rSq - scankj) ** 0.5)) - dxAbs
                            dkji = dk + (j - dyAbs) * (j - dyAbs) + iPlus * iPlus
                            if dkji > mmax:
                                mmax = dkji
                        j += 1
                    k += 1
                # print(r1Sq)
                r1Sq[rSqInd] = mmax
                rSqInd += 1
        # print(r1Sq)
        return r1Sq



if __name__ == "__main__":
    im = np.array([[0, 1, 0, 0, 1],
                   [1, 0, 1, 1, 0],
                   [0, 1, 1, 0, 1],
                   [1, 0, 0, 0, 1],
                   [0, 0, 1, 1, 1]])
    dtrans = DistanceTransform(im).transform()
    rid = DistanceRidge(dtrans).ridge()
