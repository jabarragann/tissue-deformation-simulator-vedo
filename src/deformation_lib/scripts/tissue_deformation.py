from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator


def generate_tissue_patch(
    size=0.10,  # 10 cm
    resolution=200,  # grid resolution
    height_scale=0.005,  # 5 mm amplitude
):
    """
    Generate a smooth tissue-like surface as a point cloud.

    Returns:
        P: (N,3) array of 3D points
    """
    x = np.linspace(0, size, resolution)
    y = np.linspace(0, size, resolution)
    X, Y = np.meshgrid(x, y)

    ## Patch 1
    # Xn = X / size
    # Yn = Y / size
    # Z = height_scale * np.sin(4 * np.pi * Xn / size) * np.sin(4 * np.pi * Yn / size)

    ## Patch 2
    Xn = X / (0.10 / 3.5)
    Yn = Y / (0.10 / 3.5)
    Z = height_scale * np.sin(2 * Xn * Yn) * np.cos(3 * Yn)

    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return P


def apply_local_deformation(
    P,
    center=(0.05, 0.05),  # center of deformation (meters)
    amplitude=0.004,  # 4 mm max displacement
    sigma=0.015,  # spatial extent (~3 cm)
):
    """
    Apply a smooth local deformation to a tissue surface.

    Returns:
        P_def: (N,3) deformed point cloud
        U_gt:  (N,3) ground-truth displacement field
    """
    U_gt = np.zeros_like(P)

    for i, p in enumerate(P):
        r2 = (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2
        w = np.exp(-r2 / (2 * sigma**2))

        # Deformation along surface normal (approx z-axis)
        U_gt[i, 2] = amplitude * w

    P_def = P + U_gt
    return P_def, U_gt


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

        # self.rbf_X = RBFInterpolator(
        #     fitting_pts, dX, kernel="gaussian", epsilon=self.sigma
        # )
        # self.rbf_Y = RBFInterpolator(
        #     fitting_pts, dY, kernel="gaussian", epsilon=self.sigma
        # )
        self.rbf_Z = RBFInterpolator(
            fitting_pts[:, :2], dZ, kernel="gaussian", epsilon=1.0/self.sigma
        )

        dz_est = self.rbf_Z(fitting_pts[:, :2])
        mse = np.square(dZ - dz_est).sum()
        print(f"sample estima {dz_est[:5]}")
        print(f"sample actual {dZ[:5]}")
        print(f"RBF fit completed. Training MSE (Z): {mse:.6e}")


    def evaluate(self, P):
        assert self.rbf_Z is not None
        # assert (
        #     self.rbf_X is not None and self.rbf_Y is not None and self.rbf_Z is not None
        # )
        # DX = self.rbf_X(P)
        # DY = self.rbf_Y(P)
        DZ = self.rbf_Z(P[:, :2])

        DX = np.zeros_like(DZ)
        DY = np.zeros_like(DZ)
        displacement = np.stack([DX, DY, DZ], axis=1)

        return displacement


if __name__ == "__main__":
    resolution = 100
    P = generate_tissue_patch(resolution=resolution)
    P_def1, U_gt1 = apply_local_deformation(
        P, center=(0.05, 0.05), amplitude=0.006, sigma=0.015
    )
    P_def2, U_gt2 = apply_local_deformation(
        P, center=(0.07, 0.07), amplitude=0.004, sigma=0.018
    )
    deformation_gt = U_gt1 + U_gt2
    P_def = P + deformation_gt

    # Generate observed displacements from smaller patch
    center = (0.05, 0.05)
    patch_size = 0.03
    mask = (
        (P[:, 0] >= center[0] - patch_size / 2)
        & (P[:, 0] <= center[0] + patch_size / 2)
        & (P[:, 1] >= center[1] - patch_size / 2)
        & (P[:, 1] <= center[1] + patch_size / 2)
    )
    P_local = P[mask]

    print(f"Total points: {P.shape}")
    print(f"Using {P_local.shape} points for RBF fitting.")

    ## Fit RBF to observed displacements
    displacement_field = FullRBF(sigma=0.01)
    displacement_field.fit(
        fitting_pts=P_local,
        dX=deformation_gt[mask][:, 0],
        dY=deformation_gt[mask][:, 1],
        dZ=deformation_gt[mask][:, 2],
    )

    deformation_est = displacement_field.evaluate(P)

    P_def_rbf = P + deformation_est

    mse_z = np.square(deformation_gt[:,2] - deformation_est[:,2]).mean()
    print(f"Test MSE (Z): {mse_z:.6e}")

    # from deformation_lib.visualization.matplotlib_visualization import (
    #     visualize_surfaces_with_matplotlib,
    # )
    # visualize_surfaces_with_matplotlib(P, P_def, resolution)

    from deformation_lib.visualization.vedo_visualization import (
        visualize_with_vedo_plot,
    )

    # visualize_with_vedo_plot(P, P_def, resolution)
    visualize_with_vedo_plot(P, deformation_gt, deformation_est, resolution)
