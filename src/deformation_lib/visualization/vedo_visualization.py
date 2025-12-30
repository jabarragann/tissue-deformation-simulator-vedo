import numpy as np
from scipy.interpolate import RegularGridInterpolator
from deformation_lib.utils import reshape_to_grid
from vedo import show
from vedo.pyplot import plot
from vedo import Rectangle


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


def visualize_with_vedo_plot(
    P, deformation_gt, deformation_est, resolution, z_exaggeration=1.0
):
    P_def_gt = P + deformation_gt
    P_def_est = P + deformation_est

    deformation_error = deformation_est - deformation_gt

    # Reshape grids
    X, Y, Z = reshape_to_grid(P, resolution)
    _, _, Z_def = reshape_to_grid(P_def_gt, resolution)
    dX, dY, dZ = reshape_to_grid(deformation_gt, resolution)
    dX_est, dY_est, dZ_est = reshape_to_grid(deformation_est, resolution)
    dX_err, dY_err, dZ_err = reshape_to_grid(deformation_error, resolution)

    # dx = np.zeros_like(Z)
    # dy = np.zeros_like(Z)
    # deformation_field = np.stack([dx.ravel(), dy.ravel(), deformation_gt[:, 2]], axis=1)

    # disp = Z_def - Z
    # disp_gt = deformation_gt[:,2]
    # disp_est = deformation_est[:,2]
    # disp_error = disp_est - disp_gt

    # Optional Z exaggeration (visual only)
    Z_vis = Z * z_exaggeration
    Z_def_vis = Z_def * z_exaggeration

    # Build callable surfaces
    f_orig = make_surface_function(X, Y, Z * z_exaggeration)
    f_def = make_surface_function(X, Y, Z_def * z_exaggeration)
    f_disp_gt = make_surface_function(X, Y, dZ * z_exaggeration)
    f_disp_est = make_surface_function(X, Y, dZ_est * z_exaggeration)
    f_disp_error = make_surface_function(X, Y, dZ_err * z_exaggeration)

    # Axis limits
    xlim = [X.min(), X.max()]
    ylim = [Y.min(), Y.max()]
    Zlim = [min(Z_vis.min(), Z_def_vis.min()), max(Z_vis.max(), Z_def_vis.max())]

    s1 = plot(
        f_orig,
        xlim=xlim,
        ylim=ylim,
        zlim=Zlim,
        c="viridis",
    )

    s2 = plot(
        f_def,
        xlim=xlim,
        ylim=ylim,
        zlim=Zlim,
        c="viridis",
    )

    # Third plot: deformed surface with control area rectangle
    s3 = plot(f_def, xlim=xlim, ylim=ylim, c="lightgray")

    # Define rectangle coordinates centered in the patch
    control_area_size = 0.03
    x_center = (X.min() + X.max()) / 2
    y_center = (Y.min() + Y.max()) / 2
    half_size = control_area_size / 2
    p1 = (x_center - half_size, y_center - half_size)
    p2 = (x_center + half_size, y_center + half_size)
    rect = Rectangle(p1=p1, p2=p2, c="red", alpha=0.3, res=2)

    rect.pos([rect.x(), rect.y(), 0.03])

    s3 += rect

    ## Displacement fields
    s4 = plot(
        f_disp_gt,
        xlim=xlim,
        ylim=ylim,
        c="coolwarm",
        zlevels=12,
    )

    from vedo import Line, ScalarBar3D

    line = Line((1, -1), (1, 1))
    line.cmap("coolwarm", [1000*dZ.min()*z_exaggeration, 1000*dZ.max()*z_exaggeration])
    scbar = ScalarBar3D(
        line,
        title="Displacement [mm]",
        label_rotation=90,
        c="black",
    )
    scbar = scbar.clone2d([-0.95,-0.7], size=0.11, ontop=True)

    s5 = plot(f_disp_est, xlim=xlim, ylim=ylim, c="coolwarm", zlevels=12)

    s6 = plot(f_disp_error, xlim=xlim, ylim=ylim, c="coolwarm", zlevels=12)

    show(
        [
            (s1, "Original Surface"),
            (s2, "Deformed Surface"),
            (s3, "Deformed + Control Area"),
            ((s4,scbar), "Displacement Field"),
            (s5, "Estimated Displacement Field"),
            (s6, "Displacement Error Field"),
        ],
        N=6,
        sharecam=False,
        axes=0,
        pos=(1920, 0),
    ).close()

    # show(
    #     [
    #         (s1, "Original Surface"),
    #         (s2, "Deformed Surface"),
    #         (s3, "Displacement Field"),
    #     ],
    #     N=3,
    #     sharecam=True,
    #     axes=0,
    #     pos=(1920,0)
    # ).close()
