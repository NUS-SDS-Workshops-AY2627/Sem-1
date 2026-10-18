# Data

This directory contains the hand gesture images used in the **Computer Vision workshop**.

## HaGRID Hand Gesture Dataset

We use a subset of the **HaGRID (HAnd Gesture Recognition Image Dataset)** containing 400 images across four gesture classes.

| Gesture | Folder | Images |
|---|---|---|
| ✊ Fist | `fist/` | 100 |
| 👍 Thumbs Up | `like/` | 100 |
| ✋ Open Palm | `palm/` | 100 |
| ✌️ Peace | `peace/` | 100 |

### Directory Structure

```text
data/
├── README.md
└── hagrid/
    ├── fist/
    ├── like/
    ├── palm/
    └── peace/
```

Each gesture folder contains 100 images.

### How the Dataset Is Used

During the workshop, participants will:

1. Use **MediaPipe Hand Landmarker** to extract 21 landmarks from each image.
2. Convert the landmarks into 63 wrist-relative, scale-normalised features.
3. Train a Random Forest classifier using the extracted features.
4. Evaluate the classifier on held-out images.

Images where MediaPipe cannot detect a hand are skipped.

### Train/Test Split

The extracted features are split into:

- **80% training:** Used to train the gesture classifier.
- **20% testing:** Used to evaluate predictions on held-out images.

The split is stratified to maintain similar class proportions.

**Note:** The split is performed at the image level, so test images may include people already represented in the training set.

### Dataset Source and Licence

- **Original dataset:** [HaGRID](https://github.com/hukenovs/hagrid)
- **Downloaded from:** [HaGRID Subsets on Hugging Face](https://huggingface.co/datasets/GestureDetectionConnoisseurs/hagrid_subsets)
- **Licence:** [HaGRID Public Licence](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf)

For this workshop, we selected and organised a 400-image subset into four gesture folders. No image content was intentionally modified.

The images are redistributed under the original **HaGRID Public Licence**, a modified version of CC BY-SA 4.0.

**Attribution:** HaGRID was developed by Alexander Kapitanov and collaborators. The original dataset and its licence are linked above.

## Tools

Hand landmark extraction is performed using **Google's MediaPipe Hand Landmarker**.