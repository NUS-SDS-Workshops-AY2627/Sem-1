import argparse
import time
from pathlib import Path

import cv2


def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the camera. Check camera permissions or camera index."
        )

    return cap


def capture_gesture_data(
    participant,
    gestures,
    split="train",
    samples_per_gesture=100,
    output_dir="data",
    camera_index=0,
    delay=0.15,
):
    participant_dir = Path(output_dir) / split / participant
    participant_dir.mkdir(parents=True, exist_ok=True)

    cap = initialize_camera(camera_index)

    print(f"\nParticipant: {participant}")
    print(f"Saving data to: {participant_dir}")
    print("Press 'q' to stop.\n")

    try:
        for gesture in gestures:
            gesture_dir = participant_dir / gesture
            gesture_dir.mkdir(parents=True, exist_ok=True)

            existing_files = list(gesture_dir.glob("*.jpg"))
            start_count = len(existing_files)

            print(f"Get ready for gesture: {gesture}")
            time.sleep(3)

            count = start_count

            while count < start_count + samples_per_gesture:
                success, frame = cap.read()

                if not success:
                    print("Failed to capture image.")
                    continue

                display_frame = frame.copy()

                cv2.putText(
                    display_frame,
                    f"{gesture}: {count - start_count + 1}/{samples_per_gesture}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    display_frame,
                    "Press q to stop",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                cv2.imshow("Gesture Capture", display_frame)

                image_path = gesture_dir / f"{participant}_{gesture}_{count:04d}.jpg"
                cv2.imwrite(str(image_path), frame)

                count += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Collection stopped.")
                    return

                time.sleep(delay)

            print(f"Completed {gesture}.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nFinished collecting data for {participant}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect hand gesture images using a laptop camera."
    )

    parser.add_argument(
        "--participant",
        required=True,
        help="Participant name, e.g. member_1",
    )

    parser.add_argument(
        "--split",
        choices=["train", "test"],
        required=True,
        help="Use train for members 1-3 and test for member 4.",
    )

    parser.add_argument(
        "--gestures",
        nargs="+",
        required=True,
        help="Gesture names, e.g. fist palm thumbs_up",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of images per gesture.",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index. Usually 0.",
    )

    args = parser.parse_args()

    capture_gesture_data(
        participant=args.participant,
        gestures=args.gestures,
        split=args.split,
        samples_per_gesture=args.samples,
        camera_index=args.camera_index,
    )