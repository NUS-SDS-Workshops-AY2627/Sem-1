# See, Detect, Understand: Introduction to Computer Vision ✋

Welcome to the **NUS SDS Computer Vision Workshop**!

This workshop introduces image classification, hand landmark detection and gesture recognition through two hands-on notebooks.

All notebooks are designed to run **locally in VS Code**.

## Contents

| Notebook | Topic |
|---|---|
| `(Participant Template) SDS_CV_Workshop.ipynb` | Explore RGB images, CNNs and image classification using pretrained MobileNetV2. |
| `(Participant Template) hand-gesture-recogniser.ipynb` | Extract hand landmarks using MediaPipe, train a gesture classifier and test live webcam recognition. |

Files beginning with `(Worked Solutions)` contain completed exercises for reference.

For dataset details and licensing, see [`data/README.md`](data/README.md).

---

## Participant Setup

Follow these steps in order. **Steps 1–5 only need to be completed once.**

We recommend completing the setup **before the workshop**.

### 1. Install Python 3.12

This workshop uses **Python 3.12** for compatibility with the required libraries.

If you already have Python 3.12, skip this step. If you have another version installed, you can install Python 3.12 alongside it without uninstalling your existing version.

**Windows:**

1. Download the [Python 3.12 installer](https://www.python.org/downloads/release/python-31210/).
2. Select the **Windows installer (64-bit)**.
3. Run the installer and select **Add Python to PATH**, if available.

Verify in PowerShell:

```powershell
py -3.12 --version
```

**macOS:**

1. Download the [Python 3.12 installer](https://www.python.org/downloads/release/python-31210/).
2. Select the **macOS 64-bit universal2 installer**.
3. Follow the installation instructions.

Verify in Terminal:

```bash
python3.12 --version
```

Both commands should display `Python 3.12.x`.

### 2. Install VS Code and Extensions

Download [Visual Studio Code](https://code.visualstudio.com/) if you haven't already.

In VS Code, open the **Extensions** tab (`Ctrl + Shift + X` on Windows / `Cmd + Shift + X` on macOS).

Search for and install these two extensions by Microsoft:

- [**Python**](https://marketplace.visualstudio.com/items?itemName=ms-python.python) — Python support and environment selection.
- [**Jupyter**](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) — Run `.ipynb` notebooks in VS Code.

### 3. Get the Workshop Files

You can download the repository as a ZIP file or clone it using Git.

**Option A: Download ZIP**

1. Open the [workshop repository](https://github.com/NUS-SDS-Workshops-AY2627/Sem-1).
2. Click **Code → Download ZIP**.
3. Extract the ZIP file.
4. Open the `week-10-cv` folder in VS Code using **File → Open Folder**.

**Option B: Clone using Git**

First, install [Git](https://git-scm.com/downloads) if needed.

In VS Code, open **Terminal → New Terminal**.

Navigate to the folder where you want to store the repository:

- **Windows:** Open File Explorer, navigate to your preferred folder and copy its path from the address bar.
- **macOS:** In Finder, select your preferred folder and press `Option + Command + C` to copy its path.

Then run `cd` followed by your copied path, for example:

```bash
cd "YOUR_FOLDER_PATH"
```

Clone the repository:

```bash
git clone https://github.com/NUS-SDS-Workshops-AY2627/Sem-1.git
```

Navigate to the workshop directory:

```bash
cd Sem-1/week-10-cv
```

Open it in VS Code:

```bash
code .
```

If `code .` is not recognised, use **File → Open Folder** instead.

**Important:** Open the entire `week-10-cv` folder, not just an individual notebook, so that the relative file paths work correctly.

### 4. Create and Activate a Virtual Environment

A virtual environment (`.venv`) keeps the workshop dependencies separate from your other Python projects.

Open **Terminal → New Terminal** in VS Code and ensure you're inside the `week-10-cv` folder.

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Once activated, `(.venv)` should appear in your terminal.

Verify the Python version:

```bash
python --version
```

The output should show `Python 3.12.x`.

**Note:** One virtual environment is sufficient for both notebooks. You do not need to recreate it each time.

### 5. Install Python Dependencies

With `.venv` activated, run the following commands in the VS Code terminal:

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install workshop dependencies
python -m pip install -r requirements.txt

# Install Jupyter kernel
python -m pip install ipykernel
```

The requirements include TensorFlow, Pillow, MediaPipe, OpenCV, NumPy, pandas, Matplotlib and scikit-learn.

Verify the installation:

```bash
python -c "import tensorflow, mediapipe, cv2, sklearn, ipykernel; print('Setup successful!')"
```

If you see `Setup successful!`, you're ready to proceed.

### 6. Select the Jupyter Kernel

1. Open a participant notebook in VS Code.
2. Click **Select Kernel** in the top-right corner.
3. Choose **Python Environments** and select your `.venv` environment.

If `.venv` is not listed:

1. Press `Ctrl + Shift + P` (Windows) or `Cmd + Shift + P` (macOS).
2. Search for **Python: Select Interpreter**.
3. Refresh the environment list or choose **Enter interpreter path**.

The Python interpreter is located at:

| System | Interpreter path |
|---|---|
| Windows | `.venv\Scripts\python.exe` |
| macOS | `.venv/bin/python` |

Select the same `.venv` kernel for both notebooks.

### 7. Run the Notebooks

Run the notebooks in this order:

1. `(Participant Template) SDS_CV_Workshop.ipynb`
2. `(Participant Template) hand-gesture-recogniser.ipynb`

Run cells from top to bottom using **Shift + Enter** or the run button beside each cell.

An internet connection is required for the initial pretrained model downloads.

For the live gesture recognition activity, ensure your webcam is available and allow camera access when prompted.

---

## Troubleshooting

### Python 3.12 is not recognised

Check your installation:

**Windows:**

```powershell
py -3.12 --version
```

**macOS:**

```bash
python3.12 --version
```

If the command fails, confirm that Python 3.12 is installed correctly.

You can have multiple Python versions installed simultaneously. You do not need to uninstall your existing version.

### PowerShell blocks virtual environment activation

If you encounter a script execution policy error, open a **Command Prompt** terminal in VS Code instead and run:

```cmd
.venv\Scripts\activate.bat
```

Alternatively, install packages directly into the environment without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install ipykernel
```

### `.venv` does not appear in VS Code

1. Press `Ctrl + Shift + P` (Windows) or `Cmd + Shift + P` (macOS).
2. Search for **Python: Select Interpreter**.
3. Refresh the environment list.
4. If it still does not appear, select **Enter interpreter path** and locate `.venv` manually.

Restart VS Code if necessary.

### `ModuleNotFoundError`

Make sure your notebook is using the `.venv` kernel.

If the issue persists, activate `.venv` and run:

```bash
python -m pip install -r requirements.txt
python -m pip install ipykernel
```

Restart the notebook kernel afterwards.

### `FileNotFoundError`

Make sure you've opened the `week-10-cv` folder in VS Code.

Check the working directory in a notebook cell:

```python
from pathlib import Path

print("Current directory:", Path.cwd())
print("Dataset found:", Path("data/hagrid").exists())
```

The second line should print `True`.

### MediaPipe model download fails

The hand gesture notebook automatically downloads `hand_landmarker.task` if it is missing.

Check your internet connection and rerun the model download cell.

### Webcam does not open

If the live webcam activity does not work:

1. Check that your operating system allows camera access.
2. Close other applications using the webcam.
3. Restart the notebook kernel.
4. If the wrong camera is selected or the webcam does not open, try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` to access another available camera.

Press **Q** while the webcam window is focused to stop recognition.

### Notebook is stuck or showing outdated results

Click **Restart Kernel** in VS Code and rerun the notebook from the beginning.