import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh

# ------------------------------------------------------------
# CAMERA INTRINSICS (from your snippet)
# ------------------------------------------------------------

fx = 1943.76
fy = 1943.76
cx = 600.08
cy = 601.25

img_width = 1300
img_height = 1024


# ------------------------------------------------------------
# IMAGE PATH
# ------------------------------------------------------------

image_path = "./data/phantom_demo/video_20h08m15/rosbag_videos/video_20_08_15/tracked_frames/rect/left/frame_178_left.png"


# ------------------------------------------------------------
# MARKER POSES
# ------------------------------------------------------------

data = np.load(
    "./data/phantom_demo/video_20h08m15/rosbag_videos/video_20_08_15/tracked_frames/3d_points.npz"
)
points = data["arr_0"]  # shape: (535, 5, 3)
n_frames = points.shape[0]

marker_poses = []

for i in range(4):
    pose = np.eye(4)
    pose[:3, 3] = points[0, i]
    marker_poses.append(pose)


# ------------------------------------------------------------
# LOAD BACKGROUND IMAGE
# ------------------------------------------------------------

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore


# ------------------------------------------------------------
# CREATE PYRENDER SCENE
# ------------------------------------------------------------

scene = pyrender.Scene(bg_color=[0, 0, 0, 0])


# ------------------------------------------------------------
# CAMERA (OpenCV intrinsics)
# ------------------------------------------------------------

camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

camera_pose = np.eye(4)  # camera at origin
flip = np.diag([1, -1, -1, 1])
camera_pose = camera_pose @ flip

scene.add(camera, pose=camera_pose)


# ------------------------------------------------------------
# ADD LIGHT (needed for sphere shading)
# ------------------------------------------------------------

light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light, pose=camera_pose)


# ------------------------------------------------------------
# CREATE SPHERE GEOMETRY (3mm radius)
# ------------------------------------------------------------

sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.0015)

sphere_mesh = pyrender.Mesh.from_trimesh(sphere, smooth=True)


# ------------------------------------------------------------
# ADD SPHERES AT MARKER POSES
# ------------------------------------------------------------

for pose in marker_poses:
    scene.add(sphere_mesh, pose=pose)


# ------------------------------------------------------------
# OFFSCREEN RENDER
# ------------------------------------------------------------

renderer = pyrender.OffscreenRenderer(img_width, img_height)

color, depth = renderer.render(scene)  # type: ignore


# ------------------------------------------------------------
# OVERLAY WITH CAMERA IMAGE
# ------------------------------------------------------------

alpha = (depth > 0)[..., None]

overlay = image.copy()
overlay[alpha[:, :, 0]] = color[alpha[:, :, 0]]


# ------------------------------------------------------------
# DISPLAY RESULT
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))
plt.imshow(overlay)
# plt.imshow(color)
plt.axis("off")
plt.show()
