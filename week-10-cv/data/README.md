# Data

This directory contains the datasets and image assets used for the hand gesture recognition project and Computer Vision workshop.

---

## Gesture Landmark Dataset

The gesture landmark dataset was collected by the workshop development team using webcams and **MediaPipe Hand Landmarker**.

The goal is to train a gesture classifier using hand-landmark features and evaluate whether it can generalise to a participant whose data was not used during training.

### Gestures

The dataset contains four gesture classes:

- ✋ `OPEN_PALM`
- ✊ `FIST`
- ✌️ `PEACE`
- ☝️ `POINTING`

### Data Collection

Three contributors, identified anonymously as **Member A**, **Member B**, and **Member C**, collected gesture samples using the data collection script.

Each contributor performs every gesture using both:

- Left hand
- Right hand

During collection, contributors introduce natural variation by slowly changing:

- Hand position within the frame
- Distance from the camera
- Hand height
- Hand orientation / slight rotation
- Natural finger positioning

The intended gesture should remain clearly recognisable throughout collection.

### Landmark Extraction

For every valid sample, MediaPipe Hand Landmarker extracts **21 hand landmarks**.

Each landmark contains three coordinates:

```text
(x, y, z)
```

This gives:

```text
21 landmarks × 3 coordinates = 63 landmark values
```

To reduce dependence on where the hand appears in the image, the landmarks are represented relative to the wrist (landmark `0`):

```python
relative_x = landmark.x - wrist.x
relative_y = landmark.y - wrist.y
relative_z = landmark.z - wrist.z
```

The resulting 63 values are used as features for gesture classification.

### Dataset Structure

Each row represents one detected hand sample.

The dataset contains:

```text
x0, y0, z0,
x1, y1, z1,
...
x20, y20, z20,
label,
handedness,
member,
session
```

Where:

- `x0 ... z20` — wrist-relative landmark features
- `label` — gesture class
- `handedness` — `LEFT` or `RIGHT`
- `member` — anonymous contributor identifier (`A`, `B`, or `C`)
- `session` — collection session identifier

`handedness`, `member`, and `session` are metadata and are **not used as input features for the gesture classifier**.

### Train/Test Design

The contributors are intentionally separated during model evaluation:

```text
Member A ─── Held-out Test

Member B ─┐
          ├── Training
Member C ─┘
```

Member A's samples are not used during model training.

This allows the project to evaluate how well the gesture classifier generalises to a person whose hand data it has not seen during training.

### Collecting the Data

Each team member can collect their dataset by running:

```bash
python src/collect_workshop_data.py
```

The script guides the contributor through each gesture and handedness combination and saves only frames where MediaPipe successfully detects the hand.

---

## Workshop Image Assets

The hand gesture images in `workshop_images/` were generated using **OpenAI's image generation tools** for use as demonstration materials in this Computer Vision workshop.

### Images

- `open_palm.png` — Open palm gesture
- `fist.png` — Fist gesture
- `peace.png` — Peace gesture

### Note

These images are AI-generated and do not depict real individuals.

They are used as workshop demonstration assets and are separate from the hand-landmark dataset collected by the workshop development team.

---

## Tools

Hand landmark extraction is performed using **Google's MediaPipe Hand Landmarker**.