def reshape_to_grid(P, resolution):
    """
    Reshape point cloud back to (X, Y, Z) grids.
    """
    X = P[:, 0].reshape(resolution, resolution)
    Y = P[:, 1].reshape(resolution, resolution)
    Z = P[:, 2].reshape(resolution, resolution)
    return X, Y, Z