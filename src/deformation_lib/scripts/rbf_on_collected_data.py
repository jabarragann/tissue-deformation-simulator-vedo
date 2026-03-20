import numpy as np
import numpy.typing as npt
import yaml
from matplotlib import pyplot as plt
from natsort import natsorted
from pyprojroot.here import here

from deformation_lib.scripts.deformation_fields.RBF import FullRBF


def load_marker_positions2(
    keys: list[str], data_dict: dict[str, list[float]]
) -> npt.NDArray[np.float32]:
    data = []
    for k in keys:
        data.append(np.array(data_dict[k]))
    return np.array(data)


def plot_marker_positions(
    data_before: npt.NDArray[np.float32], data_after: npt.NDArray[np.float32]
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        data_before[:, 0], data_before[:, 1], data_before[:, 2], c="r", label="Before"
    )
    ax.scatter(
        data_after[:, 0], data_after[:, 1], data_after[:, 2], c="b", label="After"
    )

    # Draw lines between corresponding points
    for before, after in zip(data_before, data_after):
        ax.plot(
            [before[0], after[0]],
            [before[1], after[1]],
            [before[2], after[2]],
            c="k",
            linestyle="--",
            linewidth=1,
        )

    ax.legend()
    plt.show()


class Drawer:
    def __init__(self, colormap_name: str = "tab10"):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.colormap = plt.get_cmap(colormap_name)
        self.color_idx = 0
        self.colors_used = {}

    def plot_marker_positions(self, data: npt.NDArray[np.float32], label: str):
        # Assign a unique color to each label (point set) using the colormap
        if label not in self.colors_used:
            color = self.colormap(self.color_idx % self.colormap.N)
            self.colors_used[label] = color
            self.color_idx += 1
        else:
            color = self.colors_used[label]

        self.ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color], label=label)

    def draw_lines(self, p1: npt.NDArray[np.float32], p2: npt.NDArray[np.float32]):
        for before, after in zip(p1, p2):
            self.ax.plot(
                [before[0], after[0]],
                [before[1], after[1]],
                [before[2], after[2]],
                c="k",
                linestyle="--",
                linewidth=1,
            )

    def show(self):
        self.ax.legend()
        plt.show()


def load_data(data_dict: dict[str, list[float]], set_name: str):

    sorted_keys_before = natsorted(data_dict[set_name].keys())
    data = load_marker_positions2(sorted_keys_before, data_dict[set_name])

    return data


def main():
    root = here()
    markers_file = root / "output/data_deformation1/object_positions.yaml"

    assert markers_file.exists(), f"Markers file not found: {markers_file}"

    with open(markers_file) as f:
        markers = yaml.safe_load(f)

    ## Manually selected
    train_before = load_data(markers, "landmarks_before")
    train_after = load_data(markers, "landmarks_after")

    ## GT points
    test_before = load_data(markers, "spheres_before_deformation")
    test_after = load_data(markers, "spheres_after_deformation")

    displacement_field = FullRBF(sigma=0.01)
    displacement = train_after - train_before
    displacement_field.fit(
        train_before, displacement[:, 0], displacement[:, 1], displacement[:, 2]
    )

    est_train_field = displacement_field.predict(train_before)
    warped_train_points = train_before + est_train_field
    mse = np.square(train_after - warped_train_points).mean()
    print(f"MSE of train field: {mse}")

    print("Evaluation with GT points")
    est_test_field = displacement_field.predict(test_before)
    warped_test_points = test_before + est_test_field
    mse = np.square(test_after - warped_test_points).mean()
    deformation_mse = np.square(test_after - test_before).mean()

    print(f"MSE due to deformation: {deformation_mse}")
    print(f"MSE of test field: {mse}")

    drawer = Drawer()
    drawer.plot_marker_positions(test_before, "before_gt")
    drawer.plot_marker_positions(test_after, "after_gt")
    drawer.draw_lines(test_before, test_after)
    drawer.plot_marker_positions(warped_test_points, "estimated_marker_pos")
    drawer.show()


if __name__ == "__main__":
    main()
