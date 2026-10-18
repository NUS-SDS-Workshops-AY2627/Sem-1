# Hand Gesture Data Collection Instructions

This guide explains how to collect hand-landmark data for the workshop hand
gesture classifier.

Member A and Member B data will be used for training. Member C data will be
kept separate as an unseen-person test set, so do not mix the member datasets.

## 1. Prepare the project

Open a terminal in the `hand-gesture-classifier` project directory:

```text
hand-gesture-classifier/
```

Install the required Python packages if you have not already done so:

```bash
pip install -r requirements.txt
```

The script needs the MediaPipe model file at:

```text
hand-gesture-classifier/hand_landmarker.task
```

If the file is missing, the script will ask whether it should download the
model from the official MediaPipe model repository. Enter `y` to download it
automatically, or `n` to stop and install it manually.

<https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task>

## 2. Run the collection script

From the `hand-gesture-classifier` directory, run:

```bash
python src/collect_workshop_data.py
```

When prompted, enter your assigned member ID:

```text
Enter member ID (A/B/C):
```

Enter only `A`, `B`, or `C`.

Each member must use their own ID:

| Member | Output file |
| --- | --- |
| A | `data/member_a_gestures.csv` |
| B | `data/member_b_gestures.csv` |
| C | `data/member_c_gestures.csv` |

If the output file already exists, the script will ask whether it should be
overwritten. Choose `n` if you want to keep the existing data.

## 3. Collection sequence

The script collects these four gestures:

1. `OPEN_PALM`
2. `FIST`
3. `PEACE`
4. `POINTING`

For each gesture, collect both:

- `RIGHT` hand
- `LEFT` hand

This produces:

```text
4 gestures x 2 hands x 20 samples = 160 samples per member
```

For every gesture and hand combination:

1. Read the instruction shown in the terminal.
2. Position the requested hand in front of the webcam.
3. Press **Space** when ready.
4. Wait for the `3`, `2`, `1`, and `START!` countdown.
5. Continue holding the requested gesture while moving naturally.
6. Wait until 20 valid samples have been collected.
7. Press **Q** at any time to stop early.

## 4. How to move during recording

Keep the intended gesture clearly recognisable while collecting different
examples of the same gesture.

Do:

- Slowly move your hand around the camera frame.
- Move slightly left and right.
- Move slightly up and down.
- Move slightly closer to and farther from the camera.
- Slightly tilt or rotate your hand.
- Allow natural variation in finger positions.
- Keep the entire hand visible.

Avoid:

- Moving very quickly.
- Rotating the hand extremely far.
- Partially hiding the hand.
- Changing to a different gesture.
- Moving outside the camera frame.

The goal is to collect varied examples, not many nearly identical frames.

## 5. Understanding the camera window

The camera window shows:

- The MediaPipe hand landmarks and skeleton when a hand is detected.
- `HAND NOT DETECTED` when the requested hand is not being tracked.
- The current gesture and hand.
- The number of samples collected.
- `CAPTURED!` whenever a valid sample is saved.

Only frames where MediaPipe detects the requested hand are saved. If
`HAND NOT DETECTED` is shown, adjust your position or lighting until the
landmarks appear.

## 6. After collection

When all combinations are complete, the script prints a summary similar to:

```text
Dataset collection complete!
Member: A
Total samples: 160
Saved to: .../data/member_a_gestures.csv
```

Check that your CSV file exists in the `data` folder and contains a header.
The file contains 63 wrist-relative landmark features followed by:

```text
label, handedness, member, session
```

The metadata columns are for evaluation and analysis. Do not use `member`,
`handedness`, or `session` as model input features.

## 7. Important separation rule

Do not rename, merge, or copy Member C's CSV into the training data. The
intended split is:

- **Training:** Member A and Member B
- **Unseen-person testing:** Member C

Keeping Member C separate allows us to measure how well the classifier works
on a person it did not see during training.
