"""Build the data files for the Week 11 NLP workshop.

Fetches every module from NUSMods, keeps the ones worth searching, and embeds
them twice (with and without the title). Organisers run this once; participants
only download the results.

    pip install -r week-11-nlp/requirements.txt
    python week-11-nlp/scripts/build_data.py

Afterwards, set FETCH_DATE in rag_utils.py to the date printed at the end.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_utils import AY, DATA_DIR, DEFAULT_FIELDS, MODEL_NAME, build_chunk

URL = f"https://api.nusmods.com/v2/{AY}/moduleInfo.json"

# NUSMods lists every module on record, including ones not running this year.
OFFERED_ONLY = True

# NUSMods never leaves a description blank; it uses one of these instead.
PLACEHOLDER_DESCRIPTIONS = {"", "not available", "not applicable", "nil"}

KEEP_FIELDS = (
    "moduleCode",
    "title",
    "description",
    "workload",
    "prerequisite",
    "faculty",
    "department",
    "moduleCredit",
)

EMBEDDING_FILES = {
    "embeddings.npy": DEFAULT_FIELDS,
    "embeddings_no_title.npy": tuple(f for f in DEFAULT_FIELDS if f != "title"),
}


def has_description(module):
    text = module["description"].strip().rstrip(".").lower()
    return text not in PLACEHOLDER_DESCRIPTIONS


def slim(module):
    """Keep only the fields the workshop uses."""
    slimmed = {field: module[field] for field in KEEP_FIELDS if module.get(field)}
    # A handful of workloads are free-text strings; keep only the hours lists.
    if not isinstance(slimmed.get("workload"), list):
        slimmed.pop("workload", None)
    slimmed["semesters"] = [s["semester"] for s in module["semesterData"]]
    slimmed["hasExam"] = any("examDate" in s for s in module["semesterData"])
    return slimmed


def main():
    print(f"Fetching {URL}")
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    all_modules = response.json()
    print(f"  {len(all_modules)} modules on record")

    kept = all_modules
    if OFFERED_ONLY:
        kept = [m for m in kept if m["semesterData"]]
        print(f"  {len(all_modules) - len(kept)} dropped: not offered in AY {AY}")
    described = [m for m in kept if has_description(m)]
    print(f"  {len(kept) - len(described)} dropped: placeholder description")

    modules = sorted((slim(m) for m in described), key=lambda m: m["moduleCode"])
    print(f"  {len(modules)} modules kept")

    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "modules_slim.json", "w", encoding="utf-8") as file:
        json.dump(modules, file, ensure_ascii=False, indent=1)

    model = SentenceTransformer(MODEL_NAME)
    for filename, fields in EMBEDDING_FILES.items():
        chunks = [build_chunk(module, fields) for module in modules]

        token_counts = [len(ids) for ids in model.tokenizer(chunks)["input_ids"]]
        too_long = sum(count > model.max_seq_length for count in token_counts)
        print(
            f"{filename}: fields={list(fields)}, median {int(np.median(token_counts))} tokens, "
            f"{too_long} chunks over the {model.max_seq_length}-token limit (truncated)"
        )

        embeddings = model.encode(
            chunks, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        ).astype(np.float32)
        assert embeddings.shape == (len(modules), 384)
        np.save(DATA_DIR / filename, embeddings)

    print(f"Done. Set FETCH_DATE = \"{date.today().isoformat()}\" in rag_utils.py")


if __name__ == "__main__":
    main()
