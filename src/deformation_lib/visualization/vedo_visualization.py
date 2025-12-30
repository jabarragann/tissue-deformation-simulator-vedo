from typing import Any
import numpy as np
from deformation_lib.utils import reshape_to_grid
from vedo import show
from vedo.pyplot import plot
from vedo import Rectangle

from vedo import ScalarBar3D, Line
from vedo import Plotter

from deformation_lib.surface import SurfaceData


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
    X, Y, Z = reshape_to_grid(P, resolution)
    _, _, Z_def = reshape_to_grid(P_def, resolution)
    disp = Z_def - Z  # displacement field

    surf_orig = mesh_from_grid(X, Y, Z * z_scale)
    surf_def = mesh_from_grid(X, Y, Z_def * z_scale)
    surf_disp = mesh_from_grid(X, Y, disp * z_scale)

    surf_orig.cmap("viridis", Z).add_scalarbar("Z [m]")
    surf_def.color("lightgray")  # type: ignore
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
    ).close()  # type: ignore


class VedoSurfacePlot:
    def __init__(
        self,
        surface_data: SurfaceData,
        title: str,
        cmap="viridis",
        zlevels=12,
        show_scalarbar=False,
        scalarbar_title=None,
        scalar_range=None,
    ):
        self.surface_data = surface_data
        self.title = title
        self.cmap = cmap
        self.zlevels = zlevels
        self.show_scalarbar = show_scalarbar
        self.scalarbar_title = scalarbar_title
        self.scalar_range = scalar_range
        self.scalarbar_scale = 1000

        self.main_actor = None
        self.scalarbar = None

        self.all_actors = []

    def build(self):
        f = self.surface_data.surface_function()

        self.main_actor = plot(
            f,
            xlim=self.surface_data.xlim,
            ylim=self.surface_data.ylim,
            zlim=self.surface_data.zlim,
            c=self.cmap,
            zlevels=self.zlevels,
        )
        self.all_actors.append(self.main_actor)

        self.scalar_range = self.surface_data.zlim

        if self.show_scalarbar:
            self.scalarbar = self._build_scalarbar()
            self.all_actors.append(self.scalarbar)

        return self

    def _build_scalarbar(self):
        # Dummy line to generate scalarbar (vedo idiom)
        line = Line((1, -1), (1, 1))
        line.cmap(
            self.cmap,
            [
                self.scalar_range[0] * self.scalarbar_scale,  # type: ignore
                self.scalar_range[1] * self.scalarbar_scale,  # type: ignore
            ],
        )

        sb = ScalarBar3D(
            line,
            title=self.scalarbar_title or "",
            label_rotation=90,
            c="black",
        )
        return sb.clone2d([-0.95, -0.7], size=0.11, ontop=True)  # type: ignore

    def actors(self):
        return self.all_actors

    def add_actor(self, actor):
        self.all_actors.append(actor)


class VedoMultiPlotFigure(Plotter):
    def __init__(self, shape, sharecam=True, pos=(1920, 0)):
        kwargs: dict[str, Any] = {"sharecam": sharecam, "pos": pos}
        super().__init__(shape=shape, **kwargs)
        self.idx = 0

    def add_object(self, surface_plot: VedoSurfacePlot):
        surface_plot.build()
        self.at(self.idx).add((surface_plot.actors(), surface_plot.title))
        self.idx += 1


def create_control_rect(X, Y):
    # Define rectangle coordinates centered in the patch
    control_area_size = 0.03
    x_center = (X.min() + X.max()) / 2
    y_center = (Y.min() + Y.max()) / 2
    half_size = control_area_size / 2
    p1 = (x_center - half_size, y_center - half_size)
    p2 = (x_center + half_size, y_center + half_size)
    rect = Rectangle(p1=p1, p2=p2, c="red", alpha=0.3, res=2)
    rect.pos([rect.x(), rect.y(), 0.03])

    return rect


def visualize_with_vedo_plot(P, deformation_gt, deformation_est, resolution):
    deformation_error = deformation_est - deformation_gt

    X, Y, Z = reshape_to_grid(P, resolution)
    _, _, Z_def = reshape_to_grid(P + deformation_gt, resolution)
    _, _, dZ_gt = reshape_to_grid(deformation_gt, resolution)
    _, _, dZ_est = reshape_to_grid(deformation_est, resolution)
    _, _, dZ_err = reshape_to_grid(deformation_error, resolution)

    surf_orig = SurfaceData(X, Y, Z)
    surf_def = SurfaceData(X, Y, Z_def)
    surf_gt = SurfaceData(X, Y, dZ_gt)
    surf_est = SurfaceData(X, Y, dZ_est)
    surf_err = SurfaceData(X, Y, dZ_err)

    fig = VedoMultiPlotFigure(shape=(2, 3), sharecam=True)

    fig.add_object(VedoSurfacePlot(surf_orig, "Original Surface"))
    fig.add_object(VedoSurfacePlot(surf_def, "Deformed Surface"))

    surf_def_control_plot = VedoSurfacePlot(
        surf_def, "Deformed Surface + control", cmap="lightgray"
    )

    rect = create_control_rect(X, Y)
    surf_def_control_plot.add_actor(rect)

    fig.add_object(surf_def_control_plot)

    surf_gt_plot = VedoSurfacePlot(
        surf_gt,
        "GT Displacement",
        cmap="coolwarm",
        zlevels=12,
        show_scalarbar=True,
        scalarbar_title="Displacement [mm]",
    )
    fig.add_object(surf_gt_plot)

    fig.add_object(
        VedoSurfacePlot(
            surf_est,
            "Estimated Displacement",
            cmap="coolwarm",
            zlevels=12,
            show_scalarbar=True,
            scalarbar_title="Displacement [mm]",
        )
    )

    fig.add_object(
        VedoSurfacePlot(
            surf_err,
            "Displacement Error",
            cmap="coolwarm",
            zlevels=12,
            show_scalarbar=True,
            scalarbar_title="Error [mm]",
        )
    )

    fig.show().interactive().close()


## Old display with vedo

# def make_surface_function(X, Y, Z):
#     """
#     Create a callable f(x,y) from grid data for vedo.plot
#     """
#     x = X[0, :]
#     y = Y[:, 0]
#
#     from scipy.interpolate import RegularGridInterpolator
#     interp = RegularGridInterpolator(
#         (x, y),
#         Z.T,  # note transpose!
#         bounds_error=False,
#         fill_value=np.nan,
#     )

#     def f(xv, yv):
#         pts = np.c_[xv, yv]
#         return interp(pts)

#     return f

# def visualize_with_vedo_plot_old(
#     P, deformation_gt, deformation_est, resolution, z_exaggeration=1.0
# ):
#     print("ugly function")
#     P_def_gt = P + deformation_gt
#     P_def_est = P + deformation_est

#     deformation_error = deformation_est - deformation_gt

#     # Reshape grids
#     X, Y, Z = reshape_to_grid(P, resolution)
#     _, _, Z_def = reshape_to_grid(P_def_gt, resolution)
#     dX, dY, dZ = reshape_to_grid(deformation_gt, resolution)
#     dX_est, dY_est, dZ_est = reshape_to_grid(deformation_est, resolution)
#     dX_err, dY_err, dZ_err = reshape_to_grid(deformation_error, resolution)

#     # dx = np.zeros_like(Z)
#     # dy = np.zeros_like(Z)
#     # deformation_field = np.stack([dx.ravel(), dy.ravel(), deformation_gt[:, 2]], axis=1)

#     # disp = Z_def - Z
#     # disp_gt = deformation_gt[:,2]
#     # disp_est = deformation_est[:,2]
#     # disp_error = disp_est - disp_gt

#     # Optional Z exaggeration (visual only)
#     Z_vis = Z * z_exaggeration
#     Z_def_vis = Z_def * z_exaggeration

#     # Build callable surfaces
#     f_orig = make_surface_function(X, Y, Z * z_exaggeration)
#     f_def = make_surface_function(X, Y, Z_def * z_exaggeration)
#     f_disp_gt = make_surface_function(X, Y, dZ * z_exaggeration)
#     f_disp_est = make_surface_function(X, Y, dZ_est * z_exaggeration)
#     f_disp_error = make_surface_function(X, Y, dZ_err * z_exaggeration)

#     # Axis limits
#     xlim = [X.min(), X.max()]
#     ylim = [Y.min(), Y.max()]
#     Zlim = [min(Z_vis.min(), Z_def_vis.min()), max(Z_vis.max(), Z_def_vis.max())]

#     s1 = plot(
#         f_orig,
#         xlim=xlim,
#         ylim=ylim,
#         zlim=Zlim,
#         c="viridis",
#     )

#     s2 = plot(
#         f_def,
#         xlim=xlim,
#         ylim=ylim,
#         zlim=Zlim,
#         c="viridis",
#     )

#     # Third plot: deformed surface with control area rectangle
#     s3 = plot(f_def, xlim=xlim, ylim=ylim, c="lightgray")
#     rect = create_control_rect(X, Y)
#     s3 += rect

#     ## Displacement fields
#     s4 = plot(
#         f_disp_gt,
#         xlim=xlim,
#         ylim=ylim,
#         c="coolwarm",
#         zlevels=12,
#     )

#     line = Line((1, -1), (1, 1))
#     line.cmap(
#         "coolwarm", [1000 * dZ.min() * z_exaggeration, 1000 * dZ.max() * z_exaggeration]
#     )
#     scbar4 = ScalarBar3D(
#         line,
#         title="Displacement [mm]",
#         label_rotation=90,
#         c="black",
#     )
#     scbar4 = scbar4.clone2d([-0.95, -0.7], size=0.11, ontop=True)

#     s5 = plot(f_disp_est, xlim=xlim, ylim=ylim, c="coolwarm", zlevels=12)
#     line = Line((1, -1), (1, 1))
#     line.cmap(
#         "coolwarm",
#         [1000 * dZ_est.min() * z_exaggeration, 1000 * dZ_est.max() * z_exaggeration],
#     )
#     scbar5 = ScalarBar3D(
#         line,
#         title="Displacement [mm]",
#         label_rotation=90,
#         c="black",
#     )
#     scbar5 = scbar5.clone2d([-0.95, -0.7], size=0.11, ontop=True)

#     s6 = plot(f_disp_error, xlim=xlim, ylim=ylim, c="coolwarm", zlevels=12)
#     line = Line((1, -1), (1, 1))
#     line.cmap(
#         "coolwarm",
#         [1000 * dZ_err.min() * z_exaggeration, 1000 * dZ_err.max() * z_exaggeration],
#     )
#     scbar6 = ScalarBar3D(
#         line,
#         title="Displacement [mm]",
#         label_rotation=90,
#         c="black",
#     )
#     scbar6 = scbar6.clone2d([-0.95, -0.7], size=0.11, ontop=True)

#     show(
#         [
#             (s1, "Original Surface"),
#             (s2, "Deformed Surface"),
#             (s3, "Deformed + Control Area"),
#             ((s4, scbar4), "Displacement Field"),
#             ((s5, scbar5), "Estimated Displacement Field"),
#             ((s6, scbar6), "Displacement Error Field"),
#         ],
#         N=6,
#         sharecam=True,
#         axes=0,
#         pos=(1920, 0),
#     ).close()

#     # show(
#     #     [
#     #         (s1, "Original Surface"),
#     #         (s2, "Deformed Surface"),
#     #         (s3, "Displacement Field"),
#     #     ],
#     #     N=3,
#     #     sharecam=True,
#     #     axes=0,
#     #     pos=(1920,0)
#     # ).close()
