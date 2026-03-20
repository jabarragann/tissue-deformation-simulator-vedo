import numpy as np
from vedo import Grid, Plotter

# ----------------------------------
# Create a 2D grid in XY plane
# ----------------------------------
grid = (
    Grid(
        pos=(0, 0, 0),
        s=(100, 100),  # grid physical size
        res=(40, 40),  # number of cells
    )
    .wireframe()
    .c("black")
)

# Get grid vertices
pts = grid.points
print(pts.shape)


# ----------------------------------
# Example deformation field
# (replace with your CV deformation)
# ----------------------------------
def deformation_field(p):
    x, y, z = p
    dx = 0
    dy = 10 * np.sin(x / 20) * np.cos(y / 20)
    dz = 5 * np.sin(x / 20) * np.cos(y / 20)
    return np.array([dx, dy, dz])


disp = np.array([deformation_field(p) for p in pts])

# ----------------------------------
# Apply deformation
# ----------------------------------
warped_pts = pts + disp
grid.points = warped_pts

# ----------------------------------
# Visualize
# ----------------------------------
plt = Plotter()
plt.show(grid, axes=1)
