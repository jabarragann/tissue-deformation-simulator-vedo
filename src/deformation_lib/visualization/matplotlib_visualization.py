import matplotlib.pyplot as plt
from matplotlib import cm


def plot_surface(fig, X, Y, Z, subplot_pos, title: str, subsample):
    ax1 = fig.add_subplot(subplot_pos, projection="3d")
    surf1 = ax1.plot_surface(
        X[::subsample, ::subsample],
        Y[::subsample, ::subsample],
        Z[::subsample, ::subsample],
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
    )
    ax1.set_title(title)
    return ax1


def visualize_surfaces_with_matplotlib(P, P_def, resolution, subsample=2):
    """
    Visualize original and deformed surfaces using triangulated surfaces
    and color-mapped displacement.
    """
    from deformation_lib.utils import reshape_to_grid

    X, Y, Z = reshape_to_grid(P, resolution)
    _, _, Z_def = reshape_to_grid(P_def, resolution)

    disp = Z_def - Z  # displacement field

    fig = plt.figure(figsize=(14, 6))

    ax1 = plot_surface(fig, X, Y, Z, 131, "Original Surface", subsample)
    ax2 = plot_surface(fig, X, Y, Z_def, 132, "Deformed Surface", subsample)
    ax3 = plot_surface(fig, X, Y, disp, 133, "Displacement Field", subsample)

    # # Deformed surface (colored by displacement)
    # ax2 = fig.add_subplot(122, projection='3d')
    # surf2 = ax2.plot_surface(
    #     X[::subsample, ::subsample],
    #     Y[::subsample, ::subsample],
    #     Z_def[::subsample, ::subsample],
    #     facecolors=cm.coolwarm(
    #         (disp - disp.min()) / (disp.max() - disp.min())
    #     ),
    #     linewidth=0,
    #     antialiased=True
    # )
    # ax2.set_title("Deformed Surface (Color = Z Displacement)")

    mappable = cm.ScalarMappable(cmap=cm.coolwarm)
    mappable.set_array(disp)
    fig.colorbar(mappable, ax=ax2, shrink=0.6, label="Displacement [m]")

    for ax in [ax1, ax2]:
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")

    plt.tight_layout()
    plt.show()
