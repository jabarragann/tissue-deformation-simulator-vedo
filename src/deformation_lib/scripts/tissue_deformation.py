import numpy as np
import matplotlib.pyplot as plt


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

    Xn = X / size
    Yn = Y / size
    # Z = height_scale * np.sin(4 * np.pi * Xn / size) * np.sin(4 * np.pi * Yn / size)

    Xn = X / (0.10/3.5)
    Yn = Y / (0.10/3.5)
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

if __name__ == "__main__":
    resolution = 200
    P = generate_tissue_patch(resolution=resolution)
    P_def1, U_gt1 = apply_local_deformation(P, center=(0.05, 0.05), amplitude=0.006, sigma=0.015)
    P_def2, U_gt2 = apply_local_deformation(P, center=(0.07, 0.07), amplitude=0.004, sigma=0.018)
    P_def = P + (U_gt1 + U_gt2) 

    # from deformation_lib.visualization.matplotlib_visualization import (
    #     visualize_surfaces_with_matplotlib,
    # )
    # visualize_surfaces_with_matplotlib(P, P_def, resolution)

    from deformation_lib.visualization.vedo_visualization import (
        visualize_with_vedo_plot,
    )
    visualize_with_vedo_plot(P, P_def, resolution)
