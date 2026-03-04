from dataclasses import dataclass
import json
from pathlib import Path
import cv2
import numpy as np
import numpy.typing as npt
import yaml

opencv_window_name = "Video playback"


def print_pixel_colors(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get current frame
        current_frame = param["frame"]
        # Get BGR value
        bgr = current_frame[y, x].tolist()
        rgb = [bgr[2], bgr[1], bgr[0]]
        # Convert to HSV
        hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        hsv = hsv_frame[y, x].tolist()
        print(
            f"At (x={x:04d}, y={y:04d}): RGB={[f'{v:03d}' for v in rgb]}, HSV={[f'{v:03d}' for v in hsv]}"
        )


@dataclass
class CameraCalibration:
    mtx: npt.NDArray[np.float32]
    dist: npt.NDArray[np.float32]
    rectification_matrix: npt.NDArray[np.float32]
    projection_matrix: npt.NDArray[np.float32]


def load_yaml_camera_calibration(yaml_file: Path) -> CameraCalibration:
    with open(yaml_file, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    mtx = np.array(data["camera_matrix"]["data"], dtype=np.float32).reshape(3, 3)
    dist = np.array(data["distortion_coefficients"]["data"], dtype=np.float32).reshape(
        5, 1
    )
    rectification_matrix = np.array(
        data["rectification_matrix"]["data"], dtype=np.float32
    ).reshape(3, 3)
    projection_matrix = np.array(
        data["projection_matrix"]["data"], dtype=np.float32
    ).reshape(3, 4)

    return CameraCalibration(mtx, dist, rectification_matrix, projection_matrix)


def play_video(
    video_path: Path, camera_name: str, camera_calibration: CameraCalibration, start_frame: int = 0
):
    output_path = video_path.parent / "tracked_frames"
    output_path.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow(opencv_window_name, cv2.WINDOW_NORMAL)
    # Resize window to 90% of 1920x1080 = 1728x972
    cv2.resizeWindow(opencv_window_name, 1728, 972)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame < 0 or start_frame >= total_frames:
        print(
            f"start_frame {start_frame} is out of range. Must be between 0 and {total_frames - 1}"
        )
        cap.release()
        return

    # Set starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # tracker = cv2.TrackerCSRT_create()
    tracker_initialized = False
    trackers = []
    bboxes = []

    # Image rectifier
    image_size = (1920, 1080)
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_calibration.mtx,
        camera_calibration.dist,
        camera_calibration.rectification_matrix,
        camera_calibration.projection_matrix,
        image_size,
        cv2.CV_16SC2,
    )

    while True:
        ret, frame_raw = cap.read()

        frame = cv2.remap(frame_raw, map1, map2, cv2.INTER_LINEAR)

        if not ret:
            print("Reached end of video or cannot fetch the frame.")
            break

        # Get current frame number
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # OpenCV is 0-indexed

        if not tracker_initialized:
            bboxes_file = output_path / f"{camera_name}_bbox.json"
            if bboxes_file.exists():
                with open(bboxes_file, "r") as f:
                    bboxes = json.load(f)
            else:
                for i in range(5):
                    bbox = cv2.selectROI(opencv_window_name, frame, True)
                    bboxes.append(bbox)

                # save bounded boxes in yaml
                with open(bboxes_file, "w") as f:
                    json.dump(bboxes, f, indent=4)

            for bbox in bboxes:
                tracker = cv2.TrackerCSRT_create()
                ok = tracker.init(frame, bbox)
                trackers.append(tracker)
                tracker_initialized = True
                print(f"Tracker initialized with bbox: {bbox}. Status: {ok}")



        ## Method3: Opencv tracker
        for tracker, bbox in zip(trackers, bboxes):
            ok, bbox = tracker.update(frame)
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        ############ END of tracking

        # Draw frame number on left upper corner
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(opencv_window_name, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:  # Press 'q' or ESC to quit
            print("Video stopped by user.")
            break
        # Add Pause/Play functionality with 'p' key

        if key == ord("p"):  # Press 'p' to toggle pause/play
            cv2.setMouseCallback(
                opencv_window_name, print_pixel_colors, {"frame": frame}
            )

            paused = True
            print("Paused. Press 'p' to resume.")
            while paused:
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord("p"):
                    print("Resumed.")
                    paused = False
                elif key2 == ord("s"):
                    # Save current frame to output path (to be set later)
                    frame_path = output_path / f"frame_{frame_idx}.png"
                    cv2.imwrite(str(frame_path), frame)

                elif key2 == ord("q") or key2 == 27:
                    print("Video stopped by user during pause.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    camera_name = "right"
    video_file = Path(f"data/phantom_demo/video_20h08m15/rosbag_videos/video_20_08_15/{camera_name}.mp4")
    camera_calibration_file = Path(
        f"data/phantom_demo/video_20h08m15/ros_calibration/{camera_name}.yaml"
    )
    camera_calibration = load_yaml_camera_calibration(camera_calibration_file)

    start_at_frame = 170  # Change this to the desired starting frame index
    play_video(video_file, camera_name, camera_calibration, start_frame=start_at_frame)
