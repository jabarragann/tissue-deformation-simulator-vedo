from dataclasses import dataclass
import json
from pathlib import Path
import cv2
import numpy as np
import numpy.typing as npt
import yaml

opencv_window_name = "Video playback"
COLORS = [
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (128, 128, 0),  # Olive
    (0, 128, 128),  # Teal
    (128, 0, 0),  # Maroon
]


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


def open_video(video_path: Path, start_frame: int = 0) -> cv2.VideoCapture:
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


def initialize_trackers(
    output_path: Path,
    camera_name: str,
    frame: npt.NDArray[np.uint8],
) -> tuple[list[tuple[int, int, int, int]], list[cv2.TrackerCSRT]]:

    bboxes: list[tuple[int, int, int, int]] = []
    trackers: list[cv2.TrackerCSRT] = []
    bboxes_file = output_path / f"{camera_name}_bbox.json"
    if bboxes_file.exists():
        with open(bboxes_file, "r") as f:
            bboxes = json.load(f)
    else:
        for _ in range(5):
            bbox = cv2.selectROI(opencv_window_name, frame, True)
            bboxes.append(bbox)

        # save bounded boxes in yaml
        with open(bboxes_file, "w") as f:
            json.dump(bboxes, f, indent=4)

    for bbox in bboxes:
        tracker = cv2.TrackerCSRT_create()
        ok = tracker.init(frame, bbox)
        trackers.append(tracker)
        print(f"Tracker initialized with bbox: {bbox}. Status: {ok}")

    return bboxes, trackers


def play_video(
    video_path: Path,
    camera_name: str,
    camera_calibration: CameraCalibration,
    start_frame: int = 0,
):
    output_path = video_path.parent / "tracked_frames"
    output_path.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow(opencv_window_name, cv2.WINDOW_NORMAL)
    # Resize window to 90% of 1920x1080 = 1728x972
    cv2.resizeWindow(opencv_window_name, 1728, 972)

    cap = open_video(video_path, start_frame)
    # tracker = cv2.TrackerCSRT_create()
    tracker_initialized = False
    trackers: list[cv2.TrackerCSRT] = []
    bboxes: list[tuple[int, int, int, int]] = []

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
            bboxes, trackers = initialize_trackers(output_path, camera_name, frame)
            tracker_initialized = True

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


def play_stereo_video(
    left_path: Path,
    right_path: Path,
    left_calibration: CameraCalibration,
    right_calibration: CameraCalibration,
    start_frame: int = 0,
):
    opencv_window_name_left = opencv_window_name + "_left"
    opencv_window_name_right = opencv_window_name + "_right"
    output_path = left_path.parent / "tracked_frames"

    # Resize window to 90% of 1920x1080 = 1728x972
    cv2.namedWindow(opencv_window_name_left, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(opencv_window_name_left, 1728, 972)
    cv2.namedWindow(opencv_window_name_right, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(opencv_window_name_right, 1728, 972)

    left_cap = open_video(left_path, start_frame)
    right_cap = open_video(right_path, start_frame)

    # tracker = cv2.TrackerCSRT_create()
    tracker_initialized = False
    left_trackers: list[cv2.TrackerCSRT] = []
    right_trackers: list[cv2.TrackerCSRT] = []
    left_bboxes: list[tuple[int, int, int, int]] = []
    right_bboxes: list[tuple[int, int, int, int]] = []

    # Image rectifier
    image_size = (1920, 1080)
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_calibration.mtx,
        left_calibration.dist,
        left_calibration.rectification_matrix,
        left_calibration.projection_matrix,
        image_size,
        cv2.CV_16SC2,
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_calibration.mtx,
        right_calibration.dist,
        right_calibration.rectification_matrix,
        right_calibration.projection_matrix,
        image_size,
        cv2.CV_16SC2,
    )

    all_3d_points = []
    while True:
        ret_left, left_frame_raw = left_cap.read()
        ret_right, right_frame_raw = right_cap.read()

        if not ret_left or not ret_right:
            print("Reached end of video or cannot fetch the frame.")
            break

        left_frame = cv2.remap(left_frame_raw, left_map1, left_map2, cv2.INTER_LINEAR)
        right_frame = cv2.remap(
            right_frame_raw, right_map1, right_map2, cv2.INTER_LINEAR
        )

        # Get current frame number
        frame_idx = (
            int(left_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        )  # OpenCV is 0-indexed

        if not tracker_initialized:
            left_bboxes, left_trackers = initialize_trackers(
                output_path, "left", left_frame
            )
            right_bboxes, right_trackers = initialize_trackers(
                output_path, "right", right_frame
            )
            tracker_initialized = True

        ## Method3: Opencv tracker
        left_bboxes = []
        right_bboxes = []
        for camera_name, trackers, bboxes, frame in zip(
            ["left", "right"],
            [left_trackers, right_trackers],
            [left_bboxes, right_bboxes],
            [left_frame, right_frame],
        ):
            for idx, tracker in enumerate(trackers):
                ok, bbox = tracker.update(frame)
                if ok:
                    # p1 = (int(bbox[0]), int(bbox[1]))
                    # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    # cv2.rectangle(frame, p1, p2, COLORS[idx], 2, 1)
                    bboxes.append(bbox)
                else:
                    print(f"{camera_name} Tracker {idx} failed to update with bbox: {bbox}")
        ## Draw boxes
        for bboxes, frame in zip([left_bboxes, right_bboxes], [left_frame, right_frame]):
            for idx, bbox in enumerate(bboxes):
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, COLORS[idx], 2, 1)

        ## Get triangulation_points from bounding boxes
        left_points = []
        right_points = []
        for left_bbox, right_bbox in zip(left_bboxes, right_bboxes):
            left_center = [
                left_bbox[0] + left_bbox[2] / 2,
                left_bbox[1] + left_bbox[3] / 2,
            ]
            right_center = [
                right_bbox[0] + right_bbox[2] / 2,
                right_bbox[1] + right_bbox[3] / 2,
            ]
            left_points.append(left_center)

            right_center[1] = left_center[1]  # Assuming correct rectification
            right_points.append(right_center)

        left_points = np.array(left_points)
        right_points = np.array(right_points)

        # Triangulate with OpenCV
        points3d = cv2.triangulatePoints(
            left_calibration.projection_matrix,
            right_calibration.projection_matrix,
            left_points.T,
            right_points.T,
        )
        points3d = points3d / points3d[3, :]
        points3d = points3d[:3, :].T

        # Triangulate with baseline
        disparity = left_points[:, 0] - right_points[:, 0]
        baseline = -(
            right_calibration.projection_matrix[0, 3]
            / right_calibration.projection_matrix[0, 0]
        )
        cx = right_calibration.projection_matrix[0, 2]
        cy = right_calibration.projection_matrix[1, 2]
        fx = right_calibration.projection_matrix[0, 0]
        fy = right_calibration.projection_matrix[1, 1]

        # Avoid division by zero
        disparity[disparity == 0] = 1e-6
        # Depth
        Z = fx * baseline / disparity
        # Back-project to 3D
        X = (left_points[:, 0] - cx) * Z / fx
        Y = (left_points[:, 1] - cy) * Z / fy

        points3d_2 = np.stack((X, Y, Z), axis=1)
        if points3d_2.shape[0] != 5:
            print(f"Frame {frame_idx} has {points3d_2.shape[0]} points. Expected 5.")
        else:
            all_3d_points.append(points3d_2)

        # Draw frame number on left upper corner
        cv2.putText(
            left_frame,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(opencv_window_name_left, left_frame)
        cv2.imshow(opencv_window_name_right, right_frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == 27:  # Press 'q' or ESC to quit
            print("Video stopped by user.")
            break
        # Add Pause/Play functionality with 'p' key

        if key == ord("p"):  # Press 'p' to toggle pause/play
            paused = True
            print("Paused. Press 'p' to resume.")
            while paused:
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord("p"):
                    print("Resumed.")
                    paused = False

                elif key2 == ord("q") or key2 == 27:
                    print("Video stopped by user during pause.")
                    left_cap.release()
                    cv2.destroyAllWindows()
                    return

    left_cap.release()
    cv2.destroyAllWindows()

    all_3d_points = np.array(all_3d_points, dtype=np.float32)
    with open(output_path / "3d_points.npz", "wb") as f:
        np.savez(f, all_3d_points)


if __name__ == "__main__":
    start_at_frame = 170  # Change this to the desired starting frame index

    # camera_name = "right"
    # video_file = Path(
    #     f"data/phantom_demo/video_20h08m15/rosbag_videos/video_20_08_15/{camera_name}.mp4"
    # )
    # camera_calibration_file = Path(
    #     f"data/phantom_demo/video_20h08m15/ros_calibration/{camera_name}.yaml"
    # )
    # camera_calibration = load_yaml_camera_calibration(camera_calibration_file)

    # play_video(video_file, camera_name, camera_calibration, start_frame=start_at_frame)

    left_path = Path(
        "data/phantom_demo/video_20h08m15/rosbag_videos/video_20_08_15/left.mp4"
    )
    left_calibration_file = Path(
        "data/phantom_demo/video_20h08m15/ros_calibration/left.yaml"
    )
    left_calibration = load_yaml_camera_calibration(left_calibration_file)

    right_path = Path(
        "data/phantom_demo/video_20h08m15/rosbag_videos/video_20_08_15/right.mp4"
    )
    right_calibration_file = Path(
        "data/phantom_demo/video_20h08m15/ros_calibration/right.yaml"
    )
    right_calibration = load_yaml_camera_calibration(right_calibration_file)

    play_stereo_video(
        left_path,
        right_path,
        left_calibration,
        right_calibration,
        start_frame=start_at_frame,
    )
