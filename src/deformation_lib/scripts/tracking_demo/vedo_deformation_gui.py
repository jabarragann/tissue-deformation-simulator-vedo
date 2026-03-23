import time
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from numpy import typing as npt
from vedo import Grid, Image, Plotter, Points
from vedo.shapes import Sphere, Text2D

from deformation_lib.deformation_fields.RBF import FullRBF

PINS_TO_TRACK = 5
colors = ["purple", "pink", "yellow", "red", "green"]


class AnimationStatus(Enum):
    PLAY = "play"
    PAUSE = "pause"

    def toggle(self):
        if self == AnimationStatus.PLAY:
            return AnimationStatus.PAUSE
        else:
            return AnimationStatus.PLAY


def time_init(func: Callable[..., Any]):
    @wraps(func)
    def wrapper(class_pt: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(class_pt, *args, **kwargs)
        end = time.time()
        print(f"{class_pt.__class__.__name__}.__init__ took {end - start:.6f} seconds")
        return result

    return wrapper


class VideoPlayer:
    def __init__(self, video_path: Path, start_frame: int = 0):
        self.cap = self.open_video(video_path, start_frame)

    def open_video(self, video_path: Path, start_frame: int = 0) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Unable to open video file: {video_path}")
            raise ValueError(f"Unable to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame < 0 or start_frame >= total_frames:
            print(
                f"start_frame {start_frame} is out of range. Must be between 0 and {total_frames - 1}"
            )
            cap.release()
            raise ValueError(
                f"start_frame {start_frame} is out of range. Must be between 0 and {total_frames - 1}"
            )

        # Set starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        return cap

    def get_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame")
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame.astype(np.uint8)


class AnimationViewer(Plotter):
    @time_init
    def __init__(self, video_player: VideoPlayer, tracked_points_path: Path):

        self.title = "3D Point Displacement"
        self.bg = "black"
        self.video_player = video_player
        self.animation_status = AnimationStatus.PAUSE

        kwargs: dict[str, Any] = {
            "size": (1916, 666),
            "pos": (1920, 0),
            "title": self.title,
        }
        kwargs.update({"bg": "black", "bg2": "black"})

        super().__init__(shape=(1, 3), sharecam=False, axes=1, **kwargs)
        self.interactor.RemoveObservers("KeyPressEvent")  # type: ignore
        self.add_callback("KeyPress", self.key_press_cb)  # type: ignore

        self.load_data(tracked_points_path)

        # Set scene requires point locations to be loaded (in self.load_data())
        self.set_scene()
        self.set_cameras_pose()

        self.init_animation()

    def key_press_cb(self, evt: Any):
        """Handle keyboard events"""

        key = evt.keypress
        if key == "q":
            self.break_interaction()
        elif key.lower() == "p":
            self.animation_status = self.animation_status.toggle()
            self.timer_callback("destroy", self.timer_id)  # type: ignore

            if self.animation_status == AnimationStatus.PLAY:
                self.timer_id = self.timer_callback("create", dt=int(self.dt * 1000))

            print(f"Animation {self.animation_status.value}")

        elif key.lower() == "h":
            # print camera positions and orientations
            print("--------------------------------")
            for i in [0, 1, 2]:
                print(f"Camera {i}")
                print(f"Position: {self.at(i).camera.GetPosition()}")
                print(f"Orientation: {self.at(i).camera.GetOrientation()}")
                print(f"Focal Point: {self.at(i).camera.GetFocalPoint()}")
                print(f"View Up: {self.at(i).camera.GetViewUp()}")
                print(f"Clipping Range: {self.at(i).camera.GetClippingRange()}")
                print("--------------------------------")
        elif key.lower() == "c":
            self.at(2).camera.SetPosition([0, 0, 0])
            self.at(2).camera.SetFocalPoint([0, 0, 1])
            self.at(2).camera.SetViewUp([0, -1, 0])
            self.at(2).camera.SetClippingRange([0.01, 0.4])
            self.at(2).camera.SetViewAngle(29.5138)
            self.at(2).render()
        elif key.lower() == "g":
            print(f"Window size {self.window.GetSize()} ")  # type: ignore

    def load_data(self, input_path: Path):
        data = np.load(str(input_path))
        self.points = data["arr_0"]  # shape: (535, 5, 3)
        self.n_frames = self.points.shape[0]

    def init_animation(self):

        self.add_callback("timer", self.loop_func)  # type: ignore

        self.fps = 20
        self.current_frame_id = 0
        self.dt = 1.0 / self.fps
        self.timer_id = self.timer_callback("create", dt=int(self.dt * 1000))

        self.animation_status = AnimationStatus.PLAY

    def set_scene(self):

        # Split 1 - point from tracked data
        self.sphere_list: list[Sphere] = []

        for i in range(PINS_TO_TRACK):
            pos = self.points[0, i]
            s = Sphere(r=0.003, pos=pos)  # 3 mm in meters
            s.c(colors[i])  # type: ignore
            self.sphere_list.append(s)
            self.at(0).add(s)  # type: ignore

        self.frame_text = Text2D(
            "Frame: 0", pos="top-left", c="white", font="Courier", s=1.2
        )
        self.at(0).add(self.frame_text)  # type: ignore

        # Split 2 - Image
        black_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.image = Image(black_frame)

        # self.image = Image(
        #     "./data/phantom_demo/video_20h08m15/rosbag_videos/video_20_08_15/tracked_frames/frame_178.png"
        # )
        self.at(1).add(self.image)  # type: ignore

        # Temporary 3D shape
        # self.cube = Cube(
        #     side=0.003,
        #     c="tomato",
        #     alpha=0.8,
        # )
        # self.cube.lighting("plastic")
        # self.at(1).add(self.cube)  # type: ignore

        ## Split 3 - Grid
        self.initial_markers = Points(self.points[0]).color("red", 0.5).ps(10)  # type: ignore
        self.spheres_list2 = [
            Sphere(r=0.003, pos=pos).color(colors[i], 0.5)
            for i, pos in enumerate(self.points[0])
        ]

        center_of_mass = self.initial_markers.center_of_mass()  # type: ignore
        self.grid = Grid(pos=center_of_mass, s=(0.1, 0.1), res=(20, 20)).c("white")  # type: ignore
        self.at(2).add([self.grid, self.initial_markers, *self.spheres_list2])  # type: ignore

    def loop_func(self, event: Any):
        if self.current_frame_id < self.n_frames:
            # Update Video Frames
            frame = self.video_player.get_frame()
            if frame is not None:
                self.image = Image(frame)
                self.at(1).add(self.image)  # type: ignore

            # Update sphere positions
            for i in range(PINS_TO_TRACK):
                self.sphere_list[i].pos(self.points[self.current_frame_id, i])

            # Update frame text
            self.frame_text.text(f"Frame: {self.current_frame_id}")  # type: ignore

            self.grid_deformation()  # Not working

            self.current_frame_id += 1

            self.render()

    def set_cameras_pose(self):
        # fmt: off
        # Camera 0
        cam0 = self.at(0).camera
        cam0.SetPosition(0.24658161000629916, -0.0011930366023919698, 0.19102308777216948)
        cam0.SetFocalPoint(0.0019791912677067348, -0.009777626557221052, 0.13788993141927075)
        cam0.SetViewUp(-0.21378672966623002, 0.05467717024840851, 0.97534898434983)
        cam0.SetClippingRange(0.14459957775087026, 0.37014639201429905)
        # Camera 1
        cam1 = self.at(1).camera
        cam1.SetPosition(713.3395769394505, 490.7322577287704, 2048.4002145854743)
        cam1.SetFocalPoint(713.3395769394505, 490.7322577287704, 0.0)
        cam1.SetViewUp(0.0, 1.0, 0.0)
        cam1.SetClippingRange(1918.6916429681105, 2216.617447214372)
        # Camera 2
        cam2 = self.at(2).camera
        cam2.SetPosition(0.011143963749165302, -0.0005579948994434529, -0.06875175994781133)
        cam2.SetFocalPoint(0.011143963749165302, -0.0005579948994434529, 0.14825883507728577)
        cam2.SetViewUp(0.0, -1.0, 0.0)
        cam2.SetClippingRange(0.176207764938732, 0.4615800675857036)

        self.cam2_parameters = {
            "position": (0.011143963749165302, -0.0005579948994434529, -0.06875175994781133),
            "focal_point": (0.011143963749165302, -0.0005579948994434529, 0.14825883507728577),
            "viewup": (0.0, -1.0, 0.0),
            "distance": None,
            "clipping_range": (0.176207764938732, 0.4615800675857036),
            "parallel_scale": None,
            "thickness": None,
            "view_angle": None,
            "roll": None,
        }

        # fmt: on
        print("Camera poses set.")

    def grid_deformation(self):

        displacement_field = FullRBF(sigma=0.01)
        if self.current_frame_id > 0:
            displacement = (
                self.points[self.current_frame_id]
                - self.points[self.current_frame_id - 1]
            )
            error = displacement_field.fit(
                self.points[self.current_frame_id - 1],
                displacement[:, 0],
                displacement[:, 1],
                displacement[:, 2],
            )
            print(f"Fitting error for frame {self.current_frame_id:04d}: {error: 0.4e}")
            grid_points = self.grid.points  # type: ignore
            distort_grid_points = grid_points + displacement_field.predict(grid_points)
            self.grid.points = distort_grid_points  # type: ignore

            self.initial_markers.points = self.points[self.current_frame_id]  # type: ignore
            for i in range(PINS_TO_TRACK):
                self.spheres_list2[i].pos(self.points[self.current_frame_id, i])  # type: ignore

    def custom_show(self):
        ## Add axes to the first subplot
        self.at(0).show(axes=1)

        # Note:
        # Not sure why camera 2 are not being set in the set camera function.
        # The line below is needed to set camera to desired position.
        self.at(2).show(camera=self.cam2_parameters)

        self.show().interactive().close()


def main1():

    video_path = Path("./data/phantom_demo/video_20_08_15/left.mp4")
    tracked_points_path = video_path.parent / "tracked_frames" / "3d_points.npz"

    video_player = VideoPlayer(
        video_path=video_path,
        start_frame=178,
    )
    viewer = AnimationViewer(video_player, tracked_points_path)  # type: ignore

    print("Finish setup..")

    viewer.custom_show()
    # viewer.show().interactive().close()  # type: ignore


if __name__ == "__main__":
    main1()
