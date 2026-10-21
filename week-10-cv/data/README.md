# Data

This directory contains the datasets used in the Computer Vision workshop.

## Sample Labrador Photograph

The introductory notebook uses `images/sample.jpg`. The file `YellowLabradorLooking_new.jpg` contains an identical copy.

- **Title:** *YellowLabradorLooking new.jpg*
- **Credit:** Original photograph by Elf; derivative work and retouching by Djmirko and FT2.
- **Source:** [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:YellowLabradorLooking_new.jpg).
- **Licence:** [Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/).

The bundled photographs are unchanged copies of the Wikimedia file; `sample.jpg` has only been renamed. The notebook demonstrates resizing, colour-channel separation, filtering and Gaussian blur. Adapted image views are shared under CC BY-SA 3.0 with the same attribution.

Redistribution must retain the attribution, source and licence link. Shared adaptations must indicate the changes made and use the same or a compatible licence.

This notice applies to the sample photograph and its adaptations. Other datasets retain their separately stated licences.

## Mini CNN Pet Subset

`pet_mini.npz` is a balanced, prepared subset of the Oxford-IIIT Pet Dataset used by the mini CNN notebook. It contains 320 training, 64 validation and 64 test photographs resized to 96 × 96, with cat/dog labels and binary animal masks. The compressed file is about 10 MB, so participants do not need to download the full dataset.

Source: O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar, *Cats and Dogs*, CVPR 2012. The Oxford-IIIT Pet Dataset is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); image copyrights remain with their original owners. This prepared subset contains resized adaptations and is shared under the same licence.

## HaGRID Hand Gesture Dataset

We use a subset of the **HaGRID (HAnd Gesture Recognition Image Dataset)** containing 400 images across four gesture classes.

| Gesture | Folder | Images |
|---|---|---|
| ✊ Fist | `fist/` | 100 |
| 👍 Thumbs Up | `like/` | 100 |
| ✋ Open Palm | `palm/` | 100 |
| ✌️ Peace | `peace/` | 100 |

### How the HaGRID Dataset Is Used

During the workshop, participants will:

1. Use **MediaPipe Hand Landmarker** to extract 21 landmarks from each image.
2. Convert the landmarks into 63 wrist-relative, scale-normalised features.
3. Train a Random Forest classifier using the extracted features.
4. Evaluate the classifier on held-out images.

Images where MediaPipe cannot detect a hand are skipped.

### Train/Test Split

The extracted features use a stratified 80% training and 20% testing split. Because the split is performed at the image level, test images may include people represented in the training set.

### Dataset Source and Licence

- **Original dataset:** [HaGRID](https://github.com/hukenovs/hagrid)
- **Downloaded from:** [HaGRID Subsets on Hugging Face](https://huggingface.co/datasets/GestureDetectionConnoisseurs/hagrid_subsets)
- **Licence:** [HaGRID Public Licence](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf)

For this workshop, we selected and organised a 400-image subset into four gesture folders. No image content was intentionally modified. The images are redistributed under the original HaGRID Public Licence, a modified version of CC BY-SA 4.0.

**Attribution:** HaGRID was developed by Alexander Kapitanov and collaborators. The original dataset and its licence are linked above.

### Tools

Hand landmark extraction is performed using Google's MediaPipe Hand Landmarker.
