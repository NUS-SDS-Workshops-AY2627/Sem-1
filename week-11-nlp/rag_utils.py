"""Shared helpers for the Week 11 NLP workshop (RAG over NUSMods).

Typical setup cell:

    import rag_utils
    from rag_utils import build_chunk, search, keyword_search, ask_gemini
    modules, embeddings, model = rag_utils.load()
"""

import json
from pathlib import Path

import numpy as np

# --- Constants (Section 1) ---------------------------------------------------
AY = "2026-2027"  # NUSMods academic year the data was fetched for
FETCH_DATE = "2026-10-10"
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.5-flash-lite"

DATA_DIR = Path(__file__).resolve().parent / "data"

# --- Shared state (Section 2), filled in by load() ---------------------------
modules = None  # list of dicts from modules_slim.json; row i matches embeddings[i]
embeddings = None  # numpy array (N, 384), float32, L2-normalized
model = None  # SentenceTransformer(MODEL_NAME)


def load_embeddings(filename="embeddings.npy"):
    """Load a precomputed embedding matrix from data/."""
    return np.load(DATA_DIR / filename)


def load():
    """Load modules, embeddings and the model, and return all three."""
    global modules, embeddings, model
    # Imported here so the lightweight helpers work without loading torch.
    from sentence_transformers import SentenceTransformer

    with open(DATA_DIR / "modules_slim.json", encoding="utf-8") as file:
        loaded_modules = json.load(file)
    loaded_embeddings = load_embeddings()
    if len(loaded_modules) != loaded_embeddings.shape[0]:
        raise ValueError(
            f"modules_slim.json has {len(loaded_modules)} modules but embeddings.npy "
            f"has {loaded_embeddings.shape[0]} rows. Rebuild the data files together."
        )

    modules = loaded_modules
    embeddings = loaded_embeddings
    model = SentenceTransformer(MODEL_NAME)
    return modules, embeddings, model


def _check_loaded():
    if modules is None:
        raise RuntimeError("Data not loaded yet. Run rag_utils.load() first.")


# --- Chunking (Section 2) ----------------------------------------------------
DEFAULT_FIELDS = ("moduleCode", "title", "description", "workload")
FIELD_LABELS = {"workload": "Workload: ", "prerequisite": "Prerequisites: "}
WORKLOAD_PARTS = ("lecture", "tutorial", "lab", "project", "preparation")


def format_workload(workload):
    """Turn NUSMods' weekly-hours list into words, e.g. [3, 0, 1, 3, 3]."""
    if not isinstance(workload, list):
        return workload or ""
    return ", ".join(
        f"{hours}h {part}" for hours, part in zip(workload, WORKLOAD_PARTS) if hours
    )


def build_chunk(module, fields=DEFAULT_FIELDS):
    """Return the text that gets embedded for one module."""
    parts = []
    for field in fields:
        value = module.get(field)
        if field == "workload":
            value = format_workload(value)
        if value:
            parts.append(f"{FIELD_LABELS.get(field, '')}{value}")
    return "\n".join(parts)


# --- Retrieval (Section 2) ---------------------------------------------------
def search(query, k=5, embeddings=None, modules=None):
    """Return the top k (module, score) pairs for a query.

    Pass `embeddings` to search a different matrix, e.g.
    load_embeddings("embeddings_no_title.npy"). Pass `modules` as well when
    that matrix covers only some of the modules.
    """
    _check_loaded()
    matrix = globals()["embeddings"] if embeddings is None else embeddings
    rows = globals()["modules"] if modules is None else modules
    query_vector = model.encode(query, normalize_embeddings=True)
    scores = matrix @ query_vector
    top = np.argsort(-scores)[:k]
    return [(rows[i], float(scores[i])) for i in top]


# --- Section 1 ---------------------------------------------------------------
def keyword_search(query):
    """Plain substring match over descriptions, like Ctrl+F."""
    _check_loaded()
    needle = query.lower()
    return [module for module in modules if needle in module["description"].lower()]


def ask_gemini(prompt):
    """One Gemini call with search grounding off; returns canned text on error."""
    raise NotImplementedError("Section 1: implement ask_gemini.")
