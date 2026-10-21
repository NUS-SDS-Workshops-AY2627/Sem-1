# See, Detect, Understand: Introduction to Computer Vision ✋

Welcome to the **NUS SDS Computer Vision Workshop**!

This workshop introduces image classification, semantic segmentation, hand landmark detection and gesture recognition through three hands-on notebooks.

All notebooks are designed to run **locally in VS Code**.

## Contents

| Notebook | Topic |
|---|---|
| `(Participant Template) 1_SDS_CV_Workshop.ipynb` | Explore RGB images, CNNs and image classification using pretrained MobileNetV2. |
| `(Participant Template) 2_mini_cnn.ipynb` | Train one CNN backbone from scratch, then change its output head and loss for cat/dog classification and animal segmentation. |
| `(Participant Template) 3_hand_gesture_recogniser.ipynb` | Extract hand landmarks using MediaPipe, train a gesture classifier and test live webcam recognition. |

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

Download the [workshop repository](https://github.com/NUS-SDS-Workshops-AY2627/Sem-1) using either method below.

**Option A: Download ZIP (Recommended)**

1. Open the repository link above.
2. Click the green **Code → Download ZIP** button.
3. Extract the ZIP file to your preferred location.
4. In VS Code, select **File → Open Folder** and open the `week-10-cv` folder inside the extracted repository.

**Option B: Clone with Git**

If you have [Git](https://git-scm.com/downloads) installed:

1. Open **Terminal → New Terminal** in VS Code.
2. Navigate to your preferred folder using `cd "YOUR_FOLDER_PATH"`.
   - **Windows:** Copy the folder path from File Explorer's address bar.
   - **macOS:** Select the folder in Finder and press `Option + Command + C` to copy its path.
3. Run:

```bash
git clone https://github.com/NUS-SDS-Workshops-AY2627/Sem-1.git
cd Sem-1/week-10-cv
code .
```

If `code .` is not recognised, use **File → Open Folder** in VS Code instead.

**Important:** Open the entire `week-10-cv` folder, not just an individual notebook, so that the relative dataset paths work correctly.

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

**Note:** One virtual environment is sufficient for all three notebooks. You do not need to recreate it each time.

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

The requirements include PyTorch, TensorFlow, Pillow, MediaPipe, OpenCV, NumPy, pandas, Matplotlib and scikit-learn.

Verify the installation:

```bash
python -c "import torch, tensorflow, mediapipe, cv2, sklearn, ipykernel; print('Setup successful!')"
```

If you see `Setup successful!`, you're ready to proceed.

### 6. Select the Jupyter Kernel

1. Open a participant notebook in VS Code.
2. Click **Select Kernel** (top-right) → **Python Environments**.
3. Select the `.venv` environment created earlier.

If `.venv` is missing, open the Command Palette (`Ctrl/Cmd + Shift + P`) → **Python: Select Interpreter** → **Enter interpreter path**.

- **Windows:** `.venv\Scripts\python.exe`
- **macOS:** `.venv/bin/python`

Use the same environment for all three notebooks.

### 7. Run the Notebooks

Run the notebooks in order:

1. `(Participant Template) 1_SDS_CV_Workshop.ipynb`
2. `(Participant Template) 2_mini_cnn.ipynb`
3. `(Participant Template) 3_hand_gesture_recogniser.ipynb`

The Mini CNN notebook connects the introductory CNN material to the hand gesture activity. It shows how the same backbone design can support different computer vision tasks by changing the prediction head, target and loss.

Run cells from top to bottom using **Shift + Enter**.

The Mini CNN dataset is already included as `data/pet_mini.npz`. An internet connection is required for the initial pretrained model and MediaPipe model downloads used by the other notebooks.

For live gesture recognition, ensure your webcam is available and camera access is enabled.

---

## Troubleshooting

| Issue | What to do |
|---|---|
| **Python 3.12 not recognised** | Check your installation using `py -3.12 --version` (Windows) or `python3.12 --version` (macOS). |
| **PowerShell blocks `.venv` activation** | Open a Command Prompt terminal in VS Code and run `.venv\Scripts\activate.bat`. |
| **`.venv` not appearing in VS Code** | Open the Command Palette (`Ctrl/Cmd + Shift + P`) → **Python: Select Interpreter** → Refresh or **Enter interpreter path**. |
| **`ModuleNotFoundError`** | Select the correct `.venv` kernel, reinstall dependencies with `python -m pip install -r requirements.txt`, then restart the kernel. |
| **`FileNotFoundError`** | Open the `week-10-cv` folder in VS Code. Check that `data/pet_mini.npz` and `data/hagrid/` exist relative to the notebooks. |
| **MediaPipe model download fails** | Check your internet connection and rerun the model download cell. |
| **Webcam not opening** | Allow camera access, close other apps using the webcam, or try `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`. |
| **Notebook stuck or outdated output** | Restart the Jupyter kernel and rerun the notebook from the beginning. |

**Tip:** If your environment is not working, first check that the notebook is using the correct `.venv` kernel. This is a common cause of setup errors.
