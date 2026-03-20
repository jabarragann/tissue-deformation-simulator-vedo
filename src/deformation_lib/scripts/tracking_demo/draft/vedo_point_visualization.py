import time

import numpy as np
from vedo import Plotter, Sphere, Text2D

# -----------------------------
# Load your data
# -----------------------------
data = np.load(
    "./data/phantom_demo/video_20h08m15/rosbag_videos/video_20_08_15/tracked_frames/3d_points.npz"
)
points = data["arr_0"]  # shape: (535, 5, 3)
n_frames = points.shape[0]

# -----------------------------
# Create plotter
# -----------------------------
plt = Plotter(title="3D Point Displacement", bg="black", interactive=True)

# Distinct colors for 5 points
colors = ["red", "green", "blue", "yellow", "magenta"]

# -----------------------------
# Create spheres (radius = 3 mm)
# -----------------------------
spheres = []
for i in range(5):
    s = Sphere(r=0.003)  # 3 mm in meters
    s.c(colors[i])
    spheres.append(s)
    plt.add(s)

# -----------------------------
# Frame counter text (upper-left)
# -----------------------------
frame_text = Text2D("Frame: 0", pos="top-left", c="white", font="Courier", s=1.2)
plt.add(frame_text)

# -----------------------------
# Animation loop (20 FPS)
# -----------------------------
fps = 20
dt = 1.0 / fps

f = 0


def loop_func(event):
    global f, plt
    if f < n_frames:
        # Update sphere positions
        for i in range(5):
            spheres[i].pos(points[f, i])

        # Update frame text
        frame_text.text(f"Frame: {f}")

        f += 1

    plt.render()


timer_id = plt.timer_callback("create", dt=int(dt * 1000))
plt.add_callback("timer", loop_func)

plt.interactive().close()
