import numpy as np

# from pytransform3d import rotations as pr
from pytransform3d import transformations as pt


class Pose:
    def __init__(self, x, y, z, qx, qy, qz, qw):
        self._T = pt.transform_from_pq(np.array([x, y, z, qw, qx, qy, qz]))
