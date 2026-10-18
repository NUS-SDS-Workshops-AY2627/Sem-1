"""Build workshop.ipynb from notebooks/section1.ipynb, section2.ipynb and section3.ipynb.

Edit the section notebooks, never workshop.ipynb, then run from week-11-nlp/:

    python scripts/build_workshop.py

The script joins the sections under the title below, drops every cell that
starts with "# Standalone setup", drops the sys.path.append("..") lines that
only the notebooks/ copies need, and clears all outputs.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ["section1.ipynb", "section2.ipynb", "section3.ipynb"]
PATH_LINE = re.compile(r'^sys\.path\.append\("\.\."\).*\n?', re.MULTILINE)

TITLE = """# Week 11: NLP

In this workshop you build a **RAG system**, retrieval-augmented generation, over every NUS module:

- **Section 1 - Intro to NLP and LLMs:** why text is hard for computers, how an LLM writes, and where it gets things wrong.
- **Section 2 - Retrieval Engine:** embeddings, chunking and semantic search over the NUSMods data.
- **Section 3 - Generation + Evaluation:** let Gemini answer from the retrieved modules, then measure how good the retrieval is.

Run the cells from top to bottom. Cells marked **YOUR TURN** are for you to edit."""


def source(cell):
    return "".join(cell["source"])


def for_workshop(cell):
    """Return a copy of the cell with outputs cleared and notebooks/-only lines removed."""
    cell = dict(cell, metadata={})
    if cell["cell_type"] == "code":
        cell.update(execution_count=None, outputs=[])
        text = source(cell)
        if PATH_LINE.search(text):
            text = PATH_LINE.sub("", text)
            if "sys." not in text:
                text = re.sub(r"^import sys\n", "", text, flags=re.MULTILINE)
            text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
            cell["source"] = text.splitlines(keepends=True)
    return cell


def main():
    title = {"cell_type": "markdown", "id": "workshop-title", "metadata": {}, "source": TITLE.splitlines(keepends=True)}
    cells = [title]
    metadata = None
    for name in SECTIONS:
        notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        metadata = metadata or notebook["metadata"]
        kept = [cell for cell in notebook["cells"] if not source(cell).startswith("# Standalone setup")]
        cells += [for_workshop(cell) for cell in kept]
        print(f"{name}: {len(kept)} of {len(notebook['cells'])} cells")

    ids = [cell["id"] for cell in cells]
    duplicates = sorted({cell_id for cell_id in ids if ids.count(cell_id) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate cell ids across sections: {duplicates}")

    workshop = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "workshop.ipynb"
    path.write_text(json.dumps(workshop, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)} with {len(cells)} cells")


if __name__ == "__main__":
    main()
