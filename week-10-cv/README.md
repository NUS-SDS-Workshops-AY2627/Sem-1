# NUS SDS Computer Vision Workshop

This workshop contains two participant notebooks. You can run them either
locally in VS Code or in Google Colab. Choose one environment and run the
notebooks in the order below.

## Workshop route

1. **`(Participant Template) SDS_CV_Workshop.ipynb`**  
   Learn how images are represented as RGB pixels, how a pretrained CNN makes
   predictions, and why model scores do not guarantee reliable classification.
2. **`(Participant Template) hand-gesture-recogniser.ipynb`**  
   Use MediaPipe to extract 21 hand landmarks, construct wrist-relative
   features, train a Random Forest classifier, evaluate it on a held-out person,
   and test the complete pipeline on a new hand image.

Do not use files beginning with **`(Worked Solutions)`** during the participant
exercise. They are provided for presenters and self-checking.

## Project structure

```text
week-10-cv/
├── (Participant Template) SDS_CV_Workshop.ipynb
├── (Participant Template) hand-gesture-recogniser.ipynb
├── (Worked Solutions) hand-gesture-recogniser.ipynb
├── (Worked Solutions) SDS_CV_Workshop.ipynb
├── data/
│   ├── member_a_gestures.csv
│   ├── member_b_gestures.csv
│   ├── member_c_gestures.csv
│   └── workshop_images/
└── requirements.txt
```

The `src/` directory contains organiser-only data-collection materials and is
excluded from the participant copy through `.gitignore`. Participants do not
need those files to complete either notebook.

The supplied gesture dataset contains four labels:
`OPEN_PALM`, `FIST`, `PEACE`, and `POINTING`.

The gesture notebook intentionally uses this person-separated split:

```text
Training:       Member B + Member C
Held-out test:  Member A
```

Member A is not used during training. This tests generalisation to a person
whose hand data the classifier has not seen. It does not prove that the model
works for every person, camera, background, or lighting condition.

## Option A: Run locally in VS Code

### 1. Get the workshop folder

Use the repository URL supplied by the workshop organisers. Do not copy the
GitHub web-page URL ending in `/tree/main/week-10-cv` into `git clone`.

```powershell
git clone https://github.com/NUS-SDS-Workshops-AY2627/Sem-1.git
cd NUS-SDS-Workshops-AY2627\Sem-1\week-10-cv
code .
```

If the organisers provide a ZIP file instead, extract it and open the
`week-10-cv` folder in VS Code. Open the folder, not just an individual
notebook, so that the relative `data\` paths work.

### 2. Install VS Code extensions

Install these extensions from the VS Code Extensions view:

- **Python** (`ms-python.python`)
- **Jupyter** (`ms-toolsai.jupyter`)

### 3. Create a virtual environment

Open a PowerShell terminal at the `week-10-cv` folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install tensorflow pillow
```

Python 3.10 or 3.11 is recommended because TensorFlow availability depends on
the Python version and operating system. The gesture notebook itself does not
need TensorFlow, but the first notebook does.

If PowerShell blocks activation, use the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install tensorflow pillow
```

### 4. Select the VS Code kernel

1. Open either participant notebook.
2. Click **Select Kernel** in the top-right corner.
3. Choose the interpreter containing `.venv\Scripts\python.exe`.
4. If it is not listed, choose **Select Another Kernel > Python Environments**
   or **Enter interpreter path**, then select `.venv\Scripts\python.exe`.
5. Restart the kernel if you changed the selection.

Verify the environment from the VS Code terminal:

```powershell
python -c "import tensorflow, mediapipe, cv2, pandas, sklearn; print('Environment is ready')"
```

### 5. Run the notebooks

Open and run the notebooks from top to bottom with **Shift+Enter**:

1. `(Participant Template) SDS_CV_Workshop.ipynb`
2. `(Participant Template) hand-gesture-recogniser.ipynb`

The first notebook downloads MobileNetV2 weights and a sample image. The
gesture notebook uses `hand_landmarker.task` and the files in `data\`.
Internet access is required for downloads; a GPU is not required.

For local image testing in the first notebook, keep `image_source = "sample"`
for the workshop example. If you use `image_source = "upload"`, replace the
Colab-only upload cell with a local path such as:

```python
from pathlib import Path
from PIL import Image, ImageOps

image_path = Path("data") / "workshop_images" / "peace.png"
with Image.open(image_path) as image_file:
    img = ImageOps.exif_transpose(image_file).convert("RGB")
image_name = image_path.name
```

## Option B: Run in Google Colab

Colab provides the Python runtime in the cloud, but uploading only an
`.ipynb` file does **not** upload the `data` folder or `hand_landmarker.task`.
Use one of the two file-transfer methods below before running the notebooks.

### Method 1: Upload the workshop folder as a ZIP

1. On your computer, compress the complete `week-10-cv` folder into a ZIP file.
2. Open [Google Colab](https://colab.research.google.com/).
3. Upload and open the participant notebook.
4. Run this setup cell first:

```python
from google.colab import files
import io
import os
import zipfile

uploaded = files.upload()
zip_name = next(
    name for name in uploaded
    if name.lower().endswith(".zip")
)

with zipfile.ZipFile(io.BytesIO(uploaded[zip_name])) as archive:
    archive.extractall("/content")

candidate_roots = [
    "/content/week-10-cv",
    "/content/Sem-1/week-10-cv",
]
PROJECT_ROOT = next(
    root for root in candidate_roots
    if os.path.exists(os.path.join(root, "data"))
)
os.chdir(PROJECT_ROOT)
print("Project root:", os.getcwd())
```

Then install the packages:

```python
%pip install -q -r requirements.txt
%pip install -q tensorflow pillow
```

If the first notebook is the only notebook being run, the gesture packages
are not needed, but installing the full requirements keeps the two notebooks
consistent.

### Method 2: Clone the repository

Use this only when the organisers provide a cloneable repository URL:

```python
!git clone <WORKSHOP_REPOSITORY_URL> /content/workshop
%cd /content/workshop/Sem-1/week-10-cv
%pip install -q -r requirements.txt
%pip install -q tensorflow pillow
```

If the repository itself is already the `week-10-cv` project, change the
`%cd` path to the folder containing `requirements.txt`, `data`, and the
notebooks.

### Run in Colab

1. Run the file-transfer and installation cells above.
2. Confirm the working directory:

   ```python
   from pathlib import Path
   print(Path.cwd())
   print(Path("data").exists())
   ```

3. Run the participant notebooks from top to bottom.
4. When the runtime disconnects or resets, rerun the setup, file-transfer, and
   installation cells before continuing.

The first notebook's `image_source = "upload"` option works in Colab because
it uses `google.colab.files`. For the gesture notebook, change
`UNSEEN_IMAGE_PATH` to another file under `data/workshop_images/` if desired.

## Exercise checkpoints

Complete the cells marked **TODO** in the hand-gesture notebook:

1. Make every landmark coordinate relative to the wrist.
2. Train the Random Forest with `X_train` and `y_train`.
3. Choose an image for the final unseen-image test.

If a checkpoint raises `NotImplementedError`, edit the preceding TODO cell and
run it again before continuing. Record your observations in the markdown
answer cells.

## Troubleshooting

### `No module named ...`

In VS Code, confirm that the notebook kernel is `.venv\Scripts\python.exe`.
Then restart the kernel and run:

```powershell
python -m pip install -r requirements.txt
python -m pip install tensorflow pillow
```

In Colab, rerun the `%pip install` cells and then use **Runtime > Restart
session** before running the notebook again.

### TensorFlow will not install

Check the Python version and platform:

```powershell
python --version
python -c "import platform; print(platform.platform())"
```

Use Python 3.10 or 3.11 for local execution if the current Python version
does not have a compatible TensorFlow wheel. A CPU runtime is sufficient.

### `FileNotFoundError` for `data`, an image, or a CSV

In VS Code, open the `week-10-cv` folder as the workspace. In Colab, make sure
the ZIP was extracted or the repository was cloned, then change into the
directory containing `data`:

```python
from pathlib import Path
print(Path.cwd())
print(list(Path("data").iterdir()))
```

The correct image directory is `data/workshop_images` with an underscore.

### `hand_landmarker.task` is missing

Run the gesture notebook's model-download cell again. The notebook downloads
the model from the official MediaPipe model URL and saves it in the project
root. Do not download model files from untrusted sources.

### Stale output or a stuck cell

Use **Restart Kernel** in VS Code, or **Runtime > Restart session** in Colab.
Then run the setup and notebook cells from the top. Clear old outputs if
necessary.

### Unexpected accuracy or predictions

Check that:

- the feature cell ran before the training cell;
- all 63 landmark features (`x0` through `z20`) are selected;
- Member A remains completely outside the training data; and
- the confusion matrix is inspected instead of accuracy alone.

The held-out-person result is an experiment under controlled collection
conditions, not a guarantee of real-world accuracy.

## Dataset and workshop assets

See [`data/README.md`](data/README.md) for the landmark format, collection
methodology, labels, and train/test design. The `data/` directory and the
pretrained `hand_landmarker.task` file are the only workshop assets needed by
participants.
