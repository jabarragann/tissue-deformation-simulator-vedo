from typing import Optional

import numpy as np
from scipy.interpolate import RBFInterpolator


class FullRBF:
    def __init__(self, sigma=0.01):
        self.sigma = sigma
        self.rbf_X: Optional[RBFInterpolator] = None
        self.rbf_Y: Optional[RBFInterpolator] = None
        self.rbf_Z: Optional[RBFInterpolator] = None

    def fit(self, fitting_pts, dX, dY, dZ):
        dX = dX.ravel()
        dY = dY.ravel()
        dZ = dZ.ravel()

        self.rbf_X = RBFInterpolator(
            fitting_pts, dX, kernel="gaussian", epsilon=1.0 / self.sigma
        )
        self.rbf_Y = RBFInterpolator(
            fitting_pts, dY, kernel="gaussian", epsilon=1.0 / self.sigma
        )
        self.rbf_Z = RBFInterpolator(
            fitting_pts, dZ, kernel="gaussian", epsilon=1.0 / self.sigma
        )

        dx_est = self.rbf_X(fitting_pts)
        dy_est = self.rbf_Y(fitting_pts)
        dz_est = self.rbf_Z(fitting_pts)

        # print(f"mse of dx: {np.square(dX - dx_est).sum()}")
        # print(f"mse of dy: {np.square(dY - dy_est).sum()}")
        # print(f"mse of dz: {np.square(dZ - dz_est).sum()}")
        error = np.square(
            (dX - dx_est) ** 2 + (dY - dy_est) ** 2 + (dZ - dz_est) ** 2
        ).sum()

        return error

    def predict(self, P):
        assert (
            self.rbf_X is not None and self.rbf_Y is not None and self.rbf_Z is not None
        )
        DX = self.rbf_X(P)
        DY = self.rbf_Y(P)
        DZ = self.rbf_Z(P)

        displacement = np.stack([DX, DY, DZ], axis=1)

        return displacement
