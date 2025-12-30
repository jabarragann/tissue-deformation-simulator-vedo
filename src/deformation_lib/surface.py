from scipy.interpolate import RegularGridInterpolator
import numpy as np

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

class SurfaceData:
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z 

    def surface_function(self):
        return make_surface_function(self.X, self.Y, self.Z)

    @property
    def xlim(self):
        return [self.X.min(), self.X.max()]

    @property
    def ylim(self):
        return [self.Y.min(), self.Y.max()]

    @property
    def zlim(self):
        return [self.Z.min(), self.Z.max()]