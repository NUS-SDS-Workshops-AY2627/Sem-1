# NUS SDS Computer Vision Workshop

This workshop uses two Jupyter notebooks. Participants should clone this
repository, open the repository folder in VS Code, create a Python environment,
and run the notebooks locally.

## Workshop route

Run the notebooks in this order:

1. **`(Participant Template) SDS_CV_Workshop.ipynb`**  
   An introduction to images, RGB pixels, a pretrained CNN, inference, model
   scores, and common classification mistakes. Complete the cells marked
   `TODO`.
2. **`hands-on (draft).ipynb`**  
   A hands-on gesture-recognition activity. It uses MediaPipe to extract 21
   hand landmarks, converts them into wrist-relative features, trains a
   Random Forest classifier, and evaluates it on an unseen participant.

Do not use the files beginning with **`(Worked Solution)`** during the
participant exercise. They are provided for presenters and self-checking.

## Project structure

```text
week-10-cv/
├── (Participant Template) SDS_CV_Workshop.ipynb
├── hands-on (draft).ipynb
├── (Worked Solution) hand-gesture-recogniser.ipynb
├── (Worked Solutions) SDS_CV_Workshop.ipynb
├── data/
│   ├── member_a_gestures.csv
│   ├── member_b_gestures.csv
│   ├── member_c_gestures.csv
│   └── workshop_images/
├── requirements.txt
└── src/
    └── collect_workshop_data.py
```

The supplied gesture dataset contains four labels:
`OPEN_PALM`, `FIST`, `PEACE`, and `POINTING`. Members B and C are used for
training; Member A is held out as an unseen-person test set.

## 1. Clone and open the project in VS Code

Open a PowerShell terminal and run:

```powershell
git clone https://github.com/NUS-SDS-Workshops-AY2627/Sem-1/tree/main/week-10-cv
cd week-10-cv
code .
```

Replace `https://github.com/NUS-SDS-Workshops-AY2627/Sem-1/tree/main/week-10-cv` with the repository URL supplied by the workshop
organisers. Open the **`week-10-cv` folder**, not an individual notebook, in
VS Code. Keeping the repository root open allows the notebooks to find the
model, images, and CSV files.

Install the **Python** and **Jupyter** extensions in VS Code if they are not
already installed.

## 2. Create and select a Python environment

Python 3.10 or newer is recommended. Create a project-specific virtual
environment from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install tensorflow pillow
```

The `requirements.txt` file covers the gesture notebook. TensorFlow and Pillow
are installed separately because they are used by the first notebook.

In VS Code:

1. Open either notebook.
2. Click **Select Kernel** in the top-right corner.
3. Choose the interpreter whose path contains
   `.venv\Scripts\python.exe`.
4. If the environment is not listed, choose **Enter interpreter path** and
   select `.venv\Scripts\python.exe`.

Verify the environment before starting the workshop:

```powershell
python -c "import tensorflow, mediapipe, cv2, pandas, sklearn; print('Environment is ready')"
```

If this command fails, fix the environment before running notebook cells. See
the troubleshooting section below.

## 3. Run the participant template

Open **`(Participant Template) SDS_CV_Workshop.ipynb`** and run cells from top
to bottom with **Shift+Enter**.

The first setup run downloads the sample Labrador image and MobileNetV2 model
weights, so internet access is required. A GPU is not required.

For the first run, keep this setting unchanged:

```python
image_source = "sample"
```

The notebook's `image_source = "upload"` option uses
`google.colab.files`, which is specific to Google Colab and will not work
unchanged in local VS Code. To test a local image, keep the sample mode for the
workshop or replace the upload cell with a local path, for example:

```python
from pathlib import Path
from PIL import Image, ImageOps

image_path = Path("data") / "workshop_images" / "peace.png"
with Image.open(image_path) as image_file:
    img = ImageOps.exif_transpose(image_file).convert("RGB")
image_name = image_path.name
```

Complete the exercises in order. If a cell reports that a `TODO` is unfinished,
edit that cell and run it again before continuing.

## 4. Run the hand-gesture notebook

Open **`hands-on (draft).ipynb`** after completing the participant template.
Run the setup, landmark, feature, training, evaluation, and prediction cells
in order.

The notebook uses:

- `hand_landmarker.task` as the pretrained MediaPipe model;
- the example images in `data/workshop_images/`;
- `data/member_a_gestures.csv`, `data/member_b_gestures.csv`, and
  `data/member_c_gestures.csv`.

The model file is already included. If it is missing, the notebook can download
it from the official MediaPipe model URL. Do not download model files from
untrusted sources.

### Important path fix for the current draft notebook

The current draft contains paths written as if the notebook were inside a
`notebooks` subfolder. Because the notebook is currently in the repository
root, change these paths if you see `FileNotFoundError`:

```python
# Current draft values
"../data/member_a_gestures.csv"
"../data/workshop images/open_palm.png"

# Use these values from the repository root
"data/member_a_gestures.csv"
"data/workshop_images/open_palm.png"
```

Apply the same change to the Member B and Member C CSV paths. The actual folder
name is `workshop_images` with an underscore.

## Debugging and common problems

### The notebook says `No module named ...`

The notebook is using a different Python interpreter from the one where the
packages were installed. Select `.venv\Scripts\python.exe` with **Select
Kernel**, then restart the kernel and rerun the setup cells.

Install packages through the selected interpreter:

```powershell
python -m pip install -r requirements.txt
python -m pip install tensorflow pillow
```

Using `python -m pip` is safer than using `pip` because it installs into the
Python environment currently selected in the terminal.

### TensorFlow will not install

Confirm the Python version and platform:

```powershell
python --version
python -c "import platform; print(platform.platform())"
```

Use Python 3.10 or 3.11 if the current workshop machine cannot install a
compatible TensorFlow wheel. On Windows, a CPU-only run is sufficient; do not
spend workshop time configuring CUDA or a GPU.

### `google.colab` cannot be imported

This is expected when running the local VS Code version of the first notebook.
Keep `image_source = "sample"` or use the local-image code shown above. Do not
install a package called `google.colab`.

### `FileNotFoundError` for `data`, an image, or a CSV

Check that VS Code opened the `week-10-cv` folder as the workspace and that the
relative path starts with `data\` (or `data/` inside Python strings). For the
gesture notebook, apply the path fix above, including the
`workshop_images` underscore.

You can check the notebook's working directory with:

```python
from pathlib import Path
print(Path.cwd())
```

### `hand_landmarker.task` is missing

Run the gesture notebook's model-download cell again, or download the model
from the official MediaPipe URL shown in the notebook. Make sure the file is
saved directly in the repository root beside the notebooks.

### A notebook cell is stuck or shows stale output

Use **Restart Kernel**, then run all cells from the top. Variables from an
earlier run can hide missing setup steps or outdated paths. If the problem
continues, use **Clear All Outputs** and rerun the notebook in order.

### The prediction or accuracy is unexpected

Check that the feature cell ran before the training cell and that the notebook
uses all 63 landmark features (`x0` through `z20`). Inspect the confusion
matrix rather than accuracy alone. Member A is intentionally unseen during
training, so its score may be lower than a random train/test split.

When asking for help, include the exact error message, the notebook cell that
failed, the output of `Path.cwd()`, and the Python version. Do not include
personal images or credentials.

## Dataset and workshop assets

See [`data/README.md`](data/README.md) for the landmark format, collection
methodology, gesture labels, and train/test design. The images in
`data/workshop_images/` are demonstration assets and are separate from the
landmark CSV dataset.
