import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt


class PointCloudPlotter:
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

        self.ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color], label=label)  # type: ignore

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
