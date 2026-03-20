import matplotlib.pyplot as plt
import numpy as np

data = np.load("./data/phantom_demo/video_20_08_15/tracked_frames/3d_points.npz")
points = data["arr_0"]  # shape: (535, N, 3)
n_frames = points.shape[0]

marker_count = points.shape[1]
colors = plt.cm.get_cmap("tab10", marker_count)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
component_names = ["x", "y", "z"]

for i, ax in enumerate(axs):
    for m in range(marker_count):
        ax.plot(
            np.arange(n_frames),
            points[:, m, i],
            label=f"Marker {m} ({component_names[i]})",
            color=colors(m),
        )
    ax.set_ylabel(component_names[i])
    ax.legend(loc="upper right", fontsize="small", ncol=2)
axs[-1].set_xlabel("Frame")
fig.suptitle("Marker 3D Positions Over Time")
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()
