"""Collect a beginner-friendly hand-landmark dataset for the workshop.

Run this file from the project directory with:

    python src/collect_workshop_data.py

The MediaPipe model file must be available at ``hand_landmarker.task`` in the
project directory, or MODEL_PATH can be changed below.
"""

import csv
import time
from urllib.request import urlretrieve
from pathlib import Path

import cv2
import mediapipe as mp


# These constants are intentionally kept together so workshop participants can
# easily change the collection settings.
GESTURES = ["OPEN_PALM", "FIST", "PEACE", "POINTING"]
HANDS = ["RIGHT", "LEFT"]
SAMPLES_PER_COMBINATION = 20
SAMPLE_INTERVAL_SECONDS = 0.2  # About five samples per second.
CAMERA_INDEX = 0
MODEL_PATH = Path(__file__).resolve().parent.parent / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"

WINDOW_NAME = "Workshop Hand Gesture Data Collection"
CSV_COLUMNS = (
    [coordinate + str(index) for index in range(21) for coordinate in ("x", "y", "z")]
    + ["label", "handedness", "member", "session"]
)

mp_vision = mp.tasks.vision


def ask_member_id():
    """Ask for and validate the member ID used in the output filename."""
    while True:
        member_id = input("Enter member ID (A/B/C): ").strip().upper()
        if member_id in {"A", "B", "C"}:
            return member_id
        print("Please enter only A, B, or C.")


def get_output_path(member_id):
    """Return the CSV path for one member."""
    return DATA_DIRECTORY / f"member_{member_id.lower()}_gestures.csv"


def confirm_overwrite(output_path):
    """Warn before replacing an existing member dataset."""
    if not output_path.exists():
        return True

    print(f"\nWarning: {output_path} already exists.")
    while True:
        answer = input("Overwrite this dataset? (y/n): ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter y or n.")


def print_instructions():
    """Explain how to collect useful variation instead of duplicate frames."""
    print(
        """
Collection instructions
-----------------------
During each recording:
  - Keep the intended gesture clearly recognisable.
  - Slowly move your hand around the frame.
  - Move slightly left/right and up/down.
  - Move slightly closer to and farther from the camera.
  - Slightly tilt or rotate the hand.
  - Allow natural variation in finger positions.
  - Keep the entire hand visible.

Avoid:
  - Very fast movement.
  - Extreme hand rotation.
  - Partially hiding the hand.
  - Changing to another gesture during the recording.

The goal is to collect different examples of the SAME gesture, not many
nearly identical frames. Press Q at any time to quit.
"""
    )


def ensure_model_file():
    """Find the model or ask the user whether it should be downloaded."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    print(f"\nMediaPipe model not found at {MODEL_PATH}.")
    print("The model is required before webcam collection can begin.")
    answer = input("Download it now from the official MediaPipe URL? (y/n): ")
    if answer.strip().lower() not in {"y", "yes"}:
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Download hand_landmarker.task "
            "or run the script again and choose y."
        )

    print("Downloading hand_landmarker.task...")
    try:
        urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as error:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise RuntimeError(
            "The model download failed. Check your internet connection and "
            "try again."
        ) from error

    print(f"Model downloaded to {MODEL_PATH}")
    return MODEL_PATH


def create_landmarker():
    """Create a MediaPipe Hand Landmarker configured for one hand."""
    model_path = ensure_model_file()

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def draw_hand_landmarks(frame, result):
    """Draw the detected hand skeleton on an OpenCV frame."""
    if not result.hand_landmarks:
        return

    hand_connections = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
    frame_height, frame_width = frame.shape[:2]

    for landmarks in result.hand_landmarks:
        # MediaPipe returns normalised coordinates in the range 0.0 to 1.0.
        points = [
            (
                int(landmark.x * frame_width),
                int(landmark.y * frame_height),
            )
            for landmark in landmarks
        ]

        # Draw the skeleton directly with OpenCV. This works with the newer
        # MediaPipe Tasks API, which does not include mp.solutions.
        for connection in hand_connections:
            start = points[connection.start]
            end = points[connection.end]
            cv2.line(frame, start, end, (0, 255, 0), 2, cv2.LINE_AA)

        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1, cv2.LINE_AA)


def get_detected_hand(result, requested_hand):
    """Return landmarks for the requested hand, or None if it was not found."""
    for index, handedness_list in enumerate(result.handedness):
        if not handedness_list:
            continue
        detected_hand = handedness_list[0].category_name.upper()
        if detected_hand == requested_hand:
            return result.hand_landmarks[index]
    return None


def landmarks_to_features(landmarks):
    """Convert 21 landmarks to wrist-relative x, y, and z features."""
    wrist = landmarks[0]
    features = []
    for landmark in landmarks:
        features.extend(
            [
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z,
            ]
        )
    return features


def put_text(frame, text, position, color=(255, 255, 255), scale=0.7):
    """Draw readable text with a small dark outline."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def read_frame(camera, landmarker, timestamp_ms):
    """Read one webcam frame and run MediaPipe on it."""
    success, frame = camera.read()
    if not success:
        return None, None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    return frame, result


def wait_for_space(camera, landmarker, gesture, hand, timestamp_ms):
    """Show the live camera until the participant presses Space or Q."""
    while True:
        frame, result = read_frame(camera, landmarker, timestamp_ms)
        timestamp_ms += 33
        if frame is None:
            continue

        draw_hand_landmarks(frame, result)
        put_text(frame, f"Next: {gesture} - {hand} HAND", (20, 35))
        put_text(frame, "Press SPACE when ready", (20, 75))
        put_text(frame, "Press Q to quit", (20, 115), color=(0, 200, 255))
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None, timestamp_ms
        if key == ord(" "):
            return True, timestamp_ms


def countdown(camera, landmarker, timestamp_ms):
    """Display a three-second countdown before recording starts."""
    for message in ("3", "2", "1", "START!"):
        end_time = time.monotonic() + (1 if message != "START!" else 0.5)
        while time.monotonic() < end_time:
            frame, result = read_frame(camera, landmarker, timestamp_ms)
            timestamp_ms += 33
            if frame is None:
                continue
            draw_hand_landmarks(frame, result)
            put_text(frame, message, (260, 250), color=(0, 255, 0), scale=2)
            put_text(frame, "Press Q to quit", (20, 40), color=(0, 200, 255))
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False, timestamp_ms
    return True, timestamp_ms


def collect_combination(
    camera, landmarker, gesture, hand, member_id, session, timestamp_ms
):
    """Collect valid, wrist-relative samples for one gesture and hand."""
    samples = []
    captured_feedback_until = 0
    next_sample_time = time.monotonic()

    while len(samples) < SAMPLES_PER_COMBINATION:
        frame, result = read_frame(camera, landmarker, timestamp_ms)
        timestamp_ms += 33
        if frame is None:
            continue

        draw_hand_landmarks(frame, result)
        landmarks = get_detected_hand(result, hand)

        if landmarks is None:
            put_text(frame, "HAND NOT DETECTED", (20, 155), color=(0, 0, 255))
        else:
            put_text(frame, f"Gesture: {gesture}", (20, 35))
            put_text(frame, f"Hand: {hand}", (20, 70))

            now = time.monotonic()
            if now >= next_sample_time:
                samples.append(
                    landmarks_to_features(landmarks)
                    + [gesture, hand, member_id, session]
                )
                captured_feedback_until = now + 0.25
                next_sample_time = now + SAMPLE_INTERVAL_SECONDS

        put_text(
            frame,
            f"Samples: {len(samples)} / {SAMPLES_PER_COMBINATION}",
            (20, 110),
        )
        if time.monotonic() < captured_feedback_until:
            put_text(frame, "CAPTURED!", (20, 195), color=(0, 255, 0), scale=1)
        put_text(frame, "Press Q to quit", (20, 235), color=(0, 200, 255))
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return samples, timestamp_ms, False

    return samples, timestamp_ms, True


def write_dataset(output_path, rows):
    """Write all collected rows with one header and no duplicate headers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def main():
    """Run the complete collection flow."""
    member_id = ask_member_id()
    output_path = get_output_path(member_id)

    if not confirm_overwrite(output_path):
        print("Collection cancelled; the existing dataset was not changed.")
        return

    print_instructions()
    input("Press Enter when you are ready to open the webcam...")

    rows = []
    session = 1
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    timestamp_ms = 0
    try:
        with create_landmarker() as landmarker:
            should_continue = True
            for gesture in GESTURES:
                for hand in HANDS:
                    print(f"\nNext: {gesture} - {hand} HAND")
                    ready, timestamp_ms = wait_for_space(
                        camera, landmarker, gesture, hand, timestamp_ms
                    )
                    if ready is None:
                        should_continue = False
                        break

                    ready, timestamp_ms = countdown(camera, landmarker, timestamp_ms)
                    if not ready:
                        should_continue = False
                        break

                    new_rows, timestamp_ms, completed = collect_combination(
                        camera,
                        landmarker,
                        gesture,
                        hand,
                        member_id,
                        session,
                        timestamp_ms,
                    )
                    rows.extend(new_rows)
                    if not completed:
                        should_continue = False
                        break

                    print(f"COMPLETE: {gesture} - {hand}")
                if not should_continue:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if rows:
        write_dataset(output_path, rows)
        print(
            f"\nDataset collection complete!\n"
            f"Member: {member_id}\n"
            f"Total samples: {len(rows)}\n"
            f"Saved to: {output_path}"
        )
    else:
        print("\nNo samples were collected. No CSV was written.")


if __name__ == "__main__":
    main()
