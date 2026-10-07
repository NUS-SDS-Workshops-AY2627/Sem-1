# Week 8: Machine Learning 🤖

This week covers three core ML topics plus a hands-on exercise where you build and compare your
own models.

## Contents

| Notebook | Topic |
|---|---|
| [`part_1_time_series.ipynb`](./part_1_time_series.ipynb) | Time series forecasting (moving average and related methods, using Algeria's exports) |
| [`part_2_ensemble.ipynb`](./part_2_ensemble.ipynb) | Ensemble methods (XGBoost and friends), predicting student success |
| [`part_3_unsupervised.ipynb`](./part_3_unsupervised.ipynb) | Unsupervised learning: KMeans, DBSCAN, Apriori association rules |
| [`hands_on_ensemble.ipynb`](./hands_on_ensemble.ipynb) | **Hands-on exercise:** fit KNN, Decision Tree, Random Forest and XGBoost, then stack them into an ensemble and see if the team beats any single model |

---

## Participant steps

Follow these in order. You only need to do steps 1 to 3 once.

### 1. Get the files

Download (green **Code** button, then **Download ZIP**) or clone
https://github.com/NUS-SDS-Workshops-AY2627/Sem-1, then open the `week-8-ml` folder. It should
contain the four notebooks above, a `data/` folder, and `requirements.txt`. Keep this folder
structure as is, don't move the notebooks out on their own.

### 2. Check your Python version

This workshop needs **Python 3.12** specifically.

```
python3 --version        # Windows: py -3.12 --version
```

- **Python 3.11 or earlier:** will fail to install, because `numpy==2.5.3` in `requirements.txt` requires
  3.12+.
- **Python 3.13/3.14:** some packages (notably XGBoost) don't reliably have stable wheels yet, and
  may error on setup (for example, a `libomp` error on Mac).

If you're not on 3.12, download it from **https://www.python.org/downloads/** before continuing.

### 3. Create and activate a virtual environment

A virtual environment (venv) keeps this workshop's packages separate from anything else on your
machine. **One venv covers all four notebooks** and you only set this up once for the whole folder.

```
cd week-8-ml
python3.12 -m venv .venv         # Windows: py -3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Next time you open the project, just activate it again (`source .venv/bin/activate`). No need to
recreate or reinstall unless you delete `.venv`.

### 4. Open a notebook in VS Code and select the right kernel

All participants should use **VS Code** (with the Jupyter extension installed) to run these
notebooks.

1. Open the `.ipynb` file you want.
2. Click the kernel picker in the top-right corner.
3. Choose the Python interpreter inside `.venv` (not your system Python or any other environment).

Repeat this for each notebook you open. The kernel choice is per-notebook, but it's the same
`.venv` every time.

### 5. Run the notebooks, in this order

1. `part_1_time_series.ipynb`
2. `part_2_ensemble.ipynb`
3. `part_3_unsupervised.ipynb`
4. `hands_on_ensemble.ipynb` (the hands-on exercise, see below)

Run cells top to bottom with Shift+Enter.

### 6. Do the hands-on exercise: `hands_on_ensemble.ipynb`

**Goal:** predict `sleep_debt_category` (Optimal Recovery, Mild Deficit, Moderate Debt, Severe Sleep
Debt) from evening habits, scored by macro F1. You'll fit four individual models, then combine them
into a stacked ensemble, and see whether the team beats the best single model.

1. Run cells top to bottom.
2. Only edit cells marked **YOUR TURN**.
3. Judge your models with the CV (Cross validation) score while you experiment.
4. Answer the three reflection questions in the blank cells provided.
5. Once you've settled on your final settings, run the test set check in section 6, once.

## Datasets

The `data/` folder already has what `part_1`, `part_2` and `hands_on_ensemble` need:
- `2026_WS_ses_demo.csv` (part 1)
- `student.csv` (part 2)
- `bedtime_screentime_sleep_debt.csv` (hands-on exercise)

`part_3_unsupervised.ipynb` downloads its datasets automatically via `kagglehub` (Mall Customers,
sklearn moons, Groceries dataset). No local file needed, but it does need internet access the
first time it runs.

## If something breaks

- **`pip install` fails on numpy with a "Requires-Python" error:
- 1. (cmd+shift+p) -> (type: select interpreter) -> refresh to look for the fresh venv.
- 2. If 1 does not work, i.e., venv not popping up, continue by "Enter interpreter path" -> key in week-8-ml/.venv/bin/python3.12 -> this should point the vscode to the find the correct venv.

- **Mac + XGBoost error mentioning `libomp`:** run `brew install libomp` in Terminal, then restart the
  kernel.

- **Can't find a CSV:** make sure `data/` sits right next to the notebook you're running.

- **Wrong kernel selected in VS Code:** click the kernel picker top-right and confirm it points at the Python interpreter inside `.venv`, not your system Python or another environment.
