import numpy as np
from scipy.interpolate import RegularGridInterpolator
from deformation_lib.utils import reshape_to_grid


def mesh_from_grid(X, Y, Z):
    """
    Create a vedo Mesh from structured grid data.
    """
    from vedo import Mesh

    ny, nx = X.shape

    # vertices
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

    # faces (two triangles per grid cell)
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            p0 = j * nx + i
            p1 = p0 + 1
            p2 = p0 + nx
            p3 = p2 + 1

            faces.append([p0, p1, p2])
            faces.append([p1, p3, p2])

    return Mesh([pts, faces])

def visualize_with_vedo_mesh(P, P_def, resolution, subsample=2, z_scale=3.0):
    from vedo import dataurl, sin, cos, log, show, Text2D
    from vedo.pyplot import plot

    X, Y, Z = reshape_to_grid(P, resolution)
    _, _, Z_def = reshape_to_grid(P_def, resolution)
    disp = Z_def - Z  # displacement field

    surf_orig = mesh_from_grid(X, Y, Z * z_scale)
    surf_def = mesh_from_grid(X, Y, Z_def * z_scale)
    surf_disp = mesh_from_grid(X, Y, disp * z_scale)

    surf_orig.cmap("viridis", Z).add_scalarbar("Z [m]")
    surf_def.color("lightgray")
    surf_disp.cmap("coolwarm", disp).add_scalarbar("Displacement [m]")

    # Improve appearance
    for s in (surf_orig, surf_def, surf_disp):
        s.lighting("plastic")
        s.compute_normals()

    show(
        [
            (surf_orig, "Original Surface"),
            (surf_def, "Deformed Surface"),
            (surf_disp, "Displacement"),
        ],
        N=3,
        sharecam=False,
    ).close()

def make_surface_function(X, Y, Z):
    """
    Create a callable f(x,y) from grid data for vedo.plot
    """
    x = X[0, :]
    y = Y[:, 0]

    interp = RegularGridInterpolator(
        (x, y),
        Z.T,  # note transpose!
        bounds_error=False,
        fill_value=np.nan,
    )

    def f(xv, yv):
        pts = np.c_[xv, yv]
        return interp(pts)

    return f


def visualize_with_vedo_plot(P, P_def, resolution, z_exaggeration=2.0):
    from vedo import show, Text2D
    from vedo.pyplot import plot

    # Reshape grids
    X, Y, Z = reshape_to_grid(P, resolution)
    _, _, Z_def = reshape_to_grid(P_def, resolution)
    disp = Z_def - Z

    # Optional Z exaggeration (visual only)
    Z_vis = Z * z_exaggeration
    Z_def_vis = Z_def * z_exaggeration

    # Build callable surfaces
    f_orig = make_surface_function(X, Y, Z_vis)
    f_def = make_surface_function(X, Y, Z_def_vis)
    f_disp = make_surface_function(X, Y, disp)

    # Axis limits
    xlim = [X.min(), X.max()]
    ylim = [Y.min(), Y.max()]

    s1 = plot(
        f_orig,
        xlim=xlim,
        ylim=ylim,
        c="viridis",
    )

    s2 = plot(
        f_def,
        xlim=xlim,
        ylim=ylim,
        c="viridis",
    )

    s3 = plot(
        f_disp,
        xlim=xlim,
        ylim=ylim,
        c="coolwarm",
        zlevels=12,
    )

    show(
        [
            (s1, "Original Surface"),
            (s2, "Deformed Surface"),
            (s3, "Displacement Field"),
        ],
        N=3,
        sharecam=True,
        axes=0,
        pos=(1920,0)
    ).close()
