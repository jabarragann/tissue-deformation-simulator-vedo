from pathlib import Path

import numpy as np
import numpy.typing as npt

from deformation_lib.visualization.Plotter_3D import PointCloudPlotter

# np.set_printoptions(precision=4, suppress=True)


def load_sample_data():
    """
    Sample date from tracked pins in video
    """
    pts_50_55_0 = np.array(
        [
            [6.9, -33.89, 155.13],
            [53.46, -32.5, 145.04],
            [37.04, 11.15, 140.47],
            [7.29, 26.7, 155.13],
            [-25.92, 10.57, 145.04],
        ],
        dtype=np.float32,
    )
    pts_50_55_1 = np.array(
        [
            [7.22, -32.34, 148.67],
            [53.66, -32.78, 143.87],
            [37.22, 10.45, 140.47],
            [7.67, 27.46, 162.18],
            [-26.44, 10.92, 147.44],
        ],
        dtype=np.float32,
    )

    pts_50_100_0 = np.array(
        [
            [6.9, -33.89, 155.13],
            [53.46, -32.5, 145.04],
            [37.04, 11.15, 140.47],
            [7.29, 26.7, 155.13],
            [-25.92, 10.57, 145.04],
        ],
        dtype=np.float32,
    )
    pts_50_100_1 = np.array(
        [
            [2.55, -25.73, 149.92],
            [46.55, -35.0, 146.23],
            [36.89, 9.34, 145.04],
            [7.96, 29.72, 148.67],
            [-24.61, 20.77, 132.15],
        ],
        dtype=np.float32,
    )

    pts_100_150_0 = np.array(
        [
            [2.55, -25.73, 149.92],
            [46.55, -35.0, 146.23],
            [36.89, 9.34, 145.04],
            [7.96, 29.72, 148.67],
            [-24.61, 20.77, 132.15],
        ],
        dtype=np.float32,
    )
    pts_100_150_1 = np.array(
        [
            [1.36, -24.74, 152.48],
            [39.74, -44.5, 153.79],
            [36.87, 0.36, 140.47],
            [16.21, 29.4, 153.79],
            [-17.58, 28.27, 130.22],
        ],
        dtype=np.float32,
    )

    return [
        (pts_50_55_0, pts_50_55_1),
        (pts_50_100_0, pts_50_100_1),
        (pts_100_150_0, pts_100_150_1),
    ]


def pretty_print(matrix):
    print(
        np.array2string(
            matrix * 1000, precision=2, suppress_small=True, max_line_width=175
        )
    )


def build_lstsq_matrices(
    X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    assert X.shape == Y.shape
    assert X.shape[1] == 3
    N = X.shape[0]

    # Build A matrix (3N x 12)
    A_mat = np.zeros((3 * N, 12), dtype=np.float32)
    B = np.zeros((3 * N,), dtype=np.float32)

    for i in range(N):
        x, y, z = X[i]
        xp, yp, zp = Y[i]

        # x' equation
        A_mat[3 * i] = [x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        B[3 * i] = xp

        # y' equation
        A_mat[3 * i + 1] = [0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0]
        B[3 * i + 1] = yp

        # z' equation
        A_mat[3 * i + 2] = [0, 0, 0, 0, 0, 0, 0, 0, x, y, z, 1]
        B[3 * i + 2] = zp

    return A_mat, B


def build_lstsq_matrices_vectorized(
    X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:

    assert X.shape == Y.shape
    assert X.shape[1] == 3

    N = X.shape[0]

    # Split coordinates
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    xp = Y[:, 0]
    yp = Y[:, 1]
    zp = Y[:, 2]

    zeros = np.zeros(N, dtype=np.float32)
    ones = np.ones(N, dtype=np.float32)

    # Build blocks (each is N x 12)
    A_x = np.stack(
        [x, y, z, ones, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros], axis=1
    )

    A_y = np.stack(
        [zeros, zeros, zeros, zeros, x, y, z, ones, zeros, zeros, zeros, zeros], axis=1
    )

    A_z = np.stack(
        [zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, x, y, z, ones], axis=1
    )

    # Interleave rows: x', y', z'
    A_mat = np.empty((3 * N, 12), dtype=np.float32)
    A_mat[0::3] = A_x
    A_mat[1::3] = A_y
    A_mat[2::3] = A_z

    # Build B (interleaved the same way)
    B = np.empty((3 * N,), dtype=np.float32)
    B[0::3] = xp
    B[1::3] = yp
    B[2::3] = zp

    return A_mat, B


def affine_least_squares_3d(X, Y):
    A_mat, B = build_lstsq_matrices_vectorized(X, Y)
    x, residuals, rank, s = np.linalg.lstsq(A_mat, B, rcond=None)
    return x, residuals, rank, s


def compare_matrices(X, Y):
    A_mat, B = build_lstsq_matrices(X, Y)
    A_mat_vectorized, B_vectorized = build_lstsq_matrices_vectorized(X, Y)
    assert np.allclose(A_mat, A_mat_vectorized)
    assert np.allclose(B, B_vectorized)

    # print("A matrix")
    # pretty_print(A_mat)
    # print("A matrix vectorized")
    # pretty_print(A_mat_vectorized)


def compare_lstsq_results(X, Y):
    A_mat, B = build_lstsq_matrices(X, Y)
    x, residuals, rank, s = np.linalg.lstsq(A_mat, B, rcond=None)

    A_mat_vectorized, B_vectorized = build_lstsq_matrices_vectorized(X, Y)
    x_vectorized, residuals_vectorized, rank_vectorized, s_vectorized = np.linalg.lstsq(
        A_mat_vectorized, B_vectorized, rcond=None
    )

    assert np.allclose(x, x_vectorized)
    assert np.allclose(residuals, residuals_vectorized)
    assert np.allclose(rank, rank_vectorized)
    assert np.allclose(s, s_vectorized)

    print("Results are equal")
    print("Non-vectorized")
    pretty_print(x)
    print("Vectorized")
    pretty_print(x_vectorized)


def unpack_affine(sol):
    A = np.array([sol[0:3], sol[4:7], sol[8:11]])
    t = np.array([sol[3], sol[7], sol[11]]).reshape(-1, 1)

    return A, t


def test_vectorized_implementation():

    pts0, pts1 = load_sample_data()[1]

    compare_matrices(pts0, pts1)
    print("Matrices are equal")

    compare_lstsq_results(pts0, pts1)
    print("Results are equal")


def print_data():
    video_path = Path("./data/phantom_demo/video_20_08_15/left.mp4")
    tracked_points_path = video_path.parent / "tracked_frames" / "3d_points.npz"

    data = np.load(str(tracked_points_path))["arr_0"]

    print("data from 50 and 55")
    pts0 = data[50]
    pts1 = data[55]
    pretty_print(pts0)
    pretty_print(pts1)

    print("data from 50 and 100")
    pts0 = data[50]
    pts1 = data[100]
    pretty_print(pts0)
    pretty_print(pts1)

    print("data from 100 and 150")
    pts0 = data[100]
    pts1 = data[150]
    pretty_print(pts0)
    pretty_print(pts1)


def main():
    pts0, pts1 = load_sample_data()[1]

    A_mat, B = build_lstsq_matrices_vectorized(pts0, pts1)
    x, residuals, rank, s = np.linalg.lstsq(A_mat, B, rcond=None)

    print("lstsq results")
    pretty_print(x)
    print("residuals", residuals)
    print("rank", rank)
    print("s", s)

    A, t = unpack_affine(x)

    pts0_transformed = A @ pts0.T + t
    pts0_transformed = pts0_transformed.T

    error = np.sum(np.linalg.norm(pts1 - pts0_transformed, axis=1))
    print("error", error)

    drawer = PointCloudPlotter()
    drawer.plot_marker_positions(pts0, "Before")
    drawer.plot_marker_positions(pts1, "After")
    drawer.draw_lines(pts0, pts1)
    drawer.plot_marker_positions(pts0_transformed, "estimated_marker_pos")
    drawer.show()


if __name__ == "__main__":
    # print_data()
    # test_vectorized_implementation()
    main()
