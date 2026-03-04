import json
from pathlib import Path
import cv2
import numpy as np
import numpy.typing as npt

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


def detect_colored_pins_with_hough_circles(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.medianBlur(frame_gray, 5)

    # print(frame_gray.shape, frame_gray.dtype)

    circles = cv2.HoughCircles(
        frame_gray,
        cv2.HOUGH_GRADIENT,
        1,
        20,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=80,
    )
    return circles


def detect_colored_pins_with_hsv_filtering(frame):
    """
    HSV segmentation + contour extraction.
    Returns list of [x, y, radius].
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ---- Define hue windows (tight around your measurements) ----
    # Format: (h_min, h_max)

    hue_ranges = [
        (125, 145),  # purple (center ~135)
        (70, 95),  # green (~84)
        (20, 35),  # yellow (~25)
        (0, 5),  # red low (~7 but avoid background 6-9)
        (170, 180),  # red high (~176)
    ]

    min_sat = 70  # removes brown/dark areas
    sat_ranges = [
        (50, 255),  # purple
        (50, 255),  # green
        (min_sat, 255),  # yellow
        (min_sat, 255),  # red low
        (min_sat, 255),  # red high
    ]

    min_value = 110  # critical threshold
    value_ranges = [
        (70, 255),  # purple
        (70, 255),  # green
        (min_value, 255),  # yellow
        (min_value, 255),  # red low
        (min_value, 255),  # red high
    ]

    masks = []

    for hue_range, value_range, sat_range in zip(hue_ranges, value_ranges, sat_ranges):
        lower = np.array([hue_range[0], sat_range[0], value_range[0]])
        upper = np.array([hue_range[1], sat_range[1], value_range[1]])
        mask = cv2.inRange(hsv, lower, upper)
        masks.append(mask)

    # Combine masks
    mask = np.zeros_like(masks[0])
    for m in masks:
        mask = cv2.bitwise_or(mask, m)

    # ---- Morphological cleanup ----
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ---- Find contours ----
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Tune this depending on image scale
        if area < 150:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)

        detections.append([int(x), int(y), int(radius)])

    # Optional: sort left-to-right for stable ordering
    detections = sorted(detections, key=lambda c: c[0])
    detections = np.array(detections)
    detections = np.expand_dims(detections, axis=0)

    return detections


def play_video(video_path: Path, start_frame: int = 0):
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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video or cannot fetch the frame.")
            break

        # Get current frame number
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # OpenCV is 0-indexed

        if not tracker_initialized:
            bboxes_file = output_path / "left_bbox.json"
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


            # bbox = cv2.selectROI(opencv_window_name, frame, False)
            # ok = tracker.init(frame, bbox)
            # tracker_initialized = True
            # print(f"Tracker initialized with bbox: {bbox}. Status: {ok}")

        ## Detect circles in the frame

        ##Method 1: Hough Circles
        # circles = detect_colored_pins_with_hough_circles(frame)

        ##Method 2: HSV segmentation + contour extraction
        # circles = detect_colored_pins_with_hsv_filtering(frame)

        ## Draw
        # if circles is not None:
        #     print(f"Found circles:  {circles.shape}")

        #     circles: npt.NDArray[np.uint16] = np.uint16(np.around(circles))
        #     for i in circles[0, :]:
        #         center: tuple[int, int] = (i[0], i[1])
        #         radius: int = i[2]
        #         cv2.circle(frame, center, radius, (0, 255, 0), 2)
        # else:
        #     print("No circles detected")

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
    # Example usage, replace with your own video file path and starting frame if needed
    video_file = Path(
        "data/phantom_demo/video_20h08m15/rosbag_videos/video_20_08_15/left.mp4"
    )
    start_at_frame = 170  # Change this to the desired starting frame index
    play_video(video_file, start_frame=start_at_frame)
