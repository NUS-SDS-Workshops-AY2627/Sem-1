"""Sleep debt challenge: beat the base models.

Goal: predict each person's sleep_debt_category (Optimal Recovery, Mild Deficit,
Moderate Debt, Severe Sleep Debt) from their evening habits.

Score: macro F1, an average score across the 4 categories where each category
counts equally, so rare ones like Severe Sleep Debt matter.

We removed total_sleep_hours and next_day_fatigue_score because the category is
a simple rule on those two columns, so using them would make the task trivial.
The data is probably synthetic, so treat the numbers as practice, not science.

How to play:
  1. Run:  python challenge.py
  2. Change the numbers in the YOUR TURN block.
  3. Run again.
  4. Repeat until both the tree and the KNN say WIN.

Rules: edit only the YOUR TURN block, nothing else.
"""
import argparse
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ===== CONFIG (do not edit) =====
SEED = 42
MARGIN_TREE = 0.06  # your tree must beat the base tree by this much
MARGIN_KNN = 0.01   # your KNN must beat the base KNN by this much

TARGET = "sleep_debt_category"
CLASS_NAMES = ["Optimal Recovery", "Mild Deficit", "Moderate Debt", "Severe Sleep Debt"]
LABELS = {name: i for i, name in enumerate(CLASS_NAMES)}  # numbers, so every model accepts them
NUMERIC = ["age", "bedtime_phone_minutes", "screen_brightness_pct", "blue_light_filter_active",
           "caffeine_post_5pm_mg", "physical_activity_min", "sleep_latency_min",
           "deep_sleep_pct", "rem_sleep_pct", "morning_alarm_snoozes"]
CATEGORICAL = ["gender", "occupation_type", "chronotype", "primary_bedtime_app"]
DATA_FILE = "bedtime_screentime_sleep_debt.csv"


# ===== HELPERS (do not edit) =====
def load_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FILE)
    if not os.path.exists(path):
        import kagglehub  # only needed if the CSV is not next to this file
        folder = kagglehub.dataset_download("samartalwar/sleep-debt-and-screen-time-late-night-phone-habits")
        path = os.path.join(folder, DATA_FILE)
    return pd.read_csv(path)


df = load_data()
X = df[NUMERIC + CATEGORICAL]
y = df[TARGET].map(LABELS)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)


def pipeline(model):
    prep = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("prep", prep), ("model", model)])


def cv_score(model):
    """Average macro F1 over 15 practice rounds on the TRAIN data only."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    res = cross_validate(model, X_train, y_train, cv=cv, scoring="f1_macro",
                         return_train_score=True, n_jobs=-1)
    return res["test_score"].mean(), res["train_score"].mean()


# ===== BASE MODELS (do not edit) =====
base_tree = pipeline(DecisionTreeClassifier(max_depth=3, random_state=SEED))
base_knn = pipeline(KNeighborsClassifier())


# ===== YOUR TURN: change these four numbers =====
TREE_MAX_DEPTH = 13          # how many questions the tree may ask. Too small = too simple,
                             # too big = memorises the training data. Try 2 to 10.
TREE_MIN_SAMPLES_LEAF = 1    # smallest group allowed at the end of a branch. Bigger = calmer
                             # tree. Try 1, 5, 10, 20, 40.
KNN_NEIGHBOURS = 5           # how many similar people vote. Try 3 to 40.
KNN_WEIGHTS = "uniform"      # "uniform" or "distance" (closer people count more)
# Stuck? Optional for experienced people: look up GridSearchCV in scikit-learn.
# ===== END OF YOUR TURN =====

tuned_tree = pipeline(DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH,
                                             min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
                                             random_state=SEED))
tuned_knn = pipeline(KNeighborsClassifier(n_neighbors=KNN_NEIGHBOURS, weights=KNN_WEIGHTS))


# ===== MAIN =====
def report(name, base, tuned, margin, memorise_tip, nudge):
    base_cv, _ = cv_score(base)
    your_cv, your_train = cv_score(tuned)
    target = base_cv + margin
    win = your_cv >= target
    print(f"\n{name}")
    print("  Base score | Your score | Difference | Target | Result")
    print(f"  {base_cv:10.3f} | {your_cv:10.3f} | {your_cv - base_cv:+10.3f} | {target:6.3f} | "
          f"{'WIN' if win else 'NOT YET'}")
    # "distance" KNN always scores 1.0 on its own training data, so skip the check there
    distance_knn = name == "KNN" and KNN_WEIGHTS == "distance"
    if your_train - your_cv > 0.10 and not distance_knn:
        print(f"  Your model is memorising the training data. {memorise_tip}")
    if not win:
        print(f"  {nudge}")
    return win


def main():
    tree_win = report("DECISION TREE", base_tree, tuned_tree, MARGIN_TREE,
                      "Try a smaller depth.", "Try a depth between 4 and 8.")
    knn_win = report("KNN", base_knn, tuned_knn, MARGIN_KNN,
                     "Try a larger number of neighbours.", "Try more neighbours, or weights = \"distance\".")
    if tree_win and knn_win:
        print("\nChallenge complete! Show the facilitator your four numbers.")
    print(f"\nYour numbers: TREE_MAX_DEPTH={TREE_MAX_DEPTH}, TREE_MIN_SAMPLES_LEAF={TREE_MIN_SAMPLES_LEAF}, "
          f"KNN_NEIGHBOURS={KNN_NEIGHBOURS}, KNN_WEIGHTS=\"{KNN_WEIGHTS}\"")


def final():
    """Facilitator only. Scores each model once on the hidden test split."""
    print("WARNING: scoring again and again on the same test split leaks information. Run this once.")
    models = {"base tree": base_tree, "your tree": tuned_tree, "base KNN": base_knn, "your KNN": tuned_knn}
    for name, model in models.items():
        pred = model.fit(X_train, y_train).predict(X_test)
        recalls = recall_score(y_test, pred, average=None, labels=list(range(4)))
        print(f"\n{name}: macro F1 {f1_score(y_test, pred, average='macro'):.3f}, "
              f"accuracy {accuracy_score(y_test, pred):.3f}")
        for cls, r in zip(CLASS_NAMES, recalls):
            print(f"  recall {cls:<18} {r:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="facilitator only: score on the test split")
    final() if parser.parse_args().final else main()
