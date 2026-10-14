"""Week 11 NLP workshop: Generation, Evaluation, and Limitations.

Place this file beside rag_utils.py. The functions here reuse the shared
NUSMods data, embeddings, and retrieval helpers supplied by rag_utils.py.

Expected evaluation CSV: one row per query, with a query column and a column
containing the correct module code (for example: query,moduleCode).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import rag_utils


# ---------------------------------------------------------------------------
# Section 6 — Generation
# ---------------------------------------------------------------------------

def make_context(results: list[tuple[dict[str, Any], float]]) -> str:
    """Format retrieved modules and their similarity scores as readable context."""
    blocks = []
    for rank, (module, score) in enumerate(results, start=1):
        code = module.get("moduleCode", "Unknown code")
        title = module.get("title", "Untitled")
        description = module.get("description", "No description available")
        workload = rag_utils.format_workload(module.get("workload", []))
        prerequisites = module.get("prerequisite", "Not listed")
        semesters = module.get("semesters", [])
        has_exam = module.get("hasExam")

        details = [
            f"[{rank}] {code} — {title}",
            f"Description: {description}",
        ]
        if workload:
            details.append(f"Workload: {workload}")
        if prerequisites:
            details.append(f"Prerequisites: {prerequisites}")
        if semesters:
            details.append(f"Semesters offered in AY {rag_utils.AY}: {', '.join(map(str, semesters))}")
        if has_exam is not None:
            details.append(f"Has exam listed: {'yes' if has_exam else 'no'}")
        details.append(f"Retrieval similarity: {score:.3f}")
        blocks.append("\n".join(details))
    return "\n\n".join(blocks) if blocks else "No modules were retrieved."


def build_generation_prompt(query: str, context: str) -> str:
    """Build a transparent, grounded prompt for the generation demo."""
    return f"""You are a helpful assistant answering questions about NUS modules.
Answer the user's question using only the supplied NUSMods context.

Rules:
- If the context does not contain enough evidence, say that the information is not available in the supplied context.
- Do not invent module codes, titles, prerequisites, workload hours, semesters, or exam details.
- When making a claim about a module, cite its module code in square brackets, e.g. [CS1231S].
- Keep the answer concise and distinguish explicit facts from uncertainty.
- The retrieval similarity score is not evidence that a claim is true.

Retrieved context:
{context}

User question: {query}

Answer:"""


def generate_with_gemini(prompt: str) -> str:
    """Generate an answer with Gemini without modifying shared rag_utils.py."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it in your notebook before generation."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "Gemini SDK is missing. Install it with: %pip install google-genai"
        ) from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=rag_utils.GEMINI_MODEL,
        contents=prompt,
    )

    if not getattr(response, "text", None):
        raise RuntimeError(
            "Gemini returned no text. Check the model name, API access, and quota."
        )
    return response.text.strip()
def generate_with_gemini(prompt):
    """Generate an answer using Gemini without modifying shared code."""
    import os
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in your notebook."
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=rag_utils.GEMINI_MODEL,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text.strip()

def generate_answer(query: str, k: int = 5, results=None) -> dict[str, Any]:
    """Retrieve context, print the exact prompt, and ask Gemini for an answer.

    Pass `results` to test correct vs deliberately incorrect context without
    changing the retrieval implementation.
    """
    if results is None:
        results = rag_utils.search(query, k=k)
    context = make_context(results)
    prompt = build_generation_prompt(query, context)

    print("=" * 72)
    print("PROMPT SENT TO GEMINI")
    print("=" * 72)
    print(prompt)
    print("=" * 72)

    answer = generate_with_gemini(prompt)
    return {"query": query, "retrieved": results, "context": context,
            "prompt": prompt, "answer": answer}


def demo_wrong_context(query: str, wrong_query: str | None = None, k: int = 5) -> dict[str, Any]:
    """Compare answers from normal retrieval and deliberately mismatched context.

    `wrong_query` should be a query about a different topic so its retrieved
    modules are likely to be irrelevant to the original question.
    """
    correct = generate_answer(query, k=k)
    wrong_query = wrong_query or "modules about economics and financial markets"
    wrong_results = rag_utils.search(wrong_query, k=k)
    print("\n\n### DEMO: ORIGINAL QUESTION WITH MISMATCHED CONTEXT ###\n")
    mismatched = generate_answer(query, k=k, results=wrong_results)
    return {"correct_context_run": correct, "wrong_context_run": mismatched,
            "wrong_context_query": wrong_query}


# ---------------------------------------------------------------------------
# Section 7 — Evaluation
# ---------------------------------------------------------------------------

QUERY_COLUMN_CANDIDATES = ("query", "question", "search_query", "user_query")
TARGET_COLUMN_CANDIDATES = ("moduleCode", "module_code", "target", "correct_module",
                            "expected_module", "answer", "label", "gold_module")


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...], explicit: str | None) -> str:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"Column {explicit!r} not found. Available columns: {fieldnames}")
        return explicit
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(
        f"Could not identify a column from {candidates}. CSV columns are: {fieldnames}. "
        "Pass query_col=... or target_col=... explicitly."
    )


def load_eval_set(csv_path: str | Path, query_col: str | None = None,
                  target_col: str | None = None) -> list[dict[str, str]]:
    """Load labelled query-to-module pairs from CSV with flexible column names."""
    with open(csv_path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Evaluation CSV has no header: {csv_path}")
        q_col = _pick_column(reader.fieldnames, QUERY_COLUMN_CANDIDATES, query_col)
        t_col = _pick_column(reader.fieldnames, TARGET_COLUMN_CANDIDATES, target_col)
        rows = []
        for row in reader:
            query = (row.get(q_col) or "").strip()
            target = (row.get(t_col) or "").strip().upper()
            if query and target:
                rows.append({"query": query, "target": target})
    if not rows:
        raise ValueError("No usable rows found. Each row needs a query and correct module code.")
    return rows


def evaluate_retrieval(eval_rows: list[dict[str, str]], k: int = 5,
                       embeddings=None, modules=None) -> dict[str, Any]:
    """Calculate Hit@k and MRR for a labelled query-to-module evaluation set."""
    if rag_utils.model is None or rag_utils.modules is None or rag_utils.embeddings is None:
        rag_utils.load()

    matrix = rag_utils.embeddings if embeddings is None else embeddings
    rows = rag_utils.modules if modules is None else modules
    hit_values = []
    reciprocal_ranks = []
    details = []

    for item in eval_rows:
        query, target = item["query"], item["target"].upper()
        results = rag_utils.search(query, k=k, embeddings=matrix, modules=rows)
        ranked_codes = [str(module.get("moduleCode", "")).upper() for module, _ in results]
        try:
            rank = ranked_codes.index(target) + 1
        except ValueError:
            rank = 0
        hit_values.append(1 if rank and rank <= k else 0)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        details.append({
            "query": query,
            "expected_module": target,
            f"top_{k}_module_codes": " | ".join(ranked_codes),
            "rank_of_correct_module": rank if rank else "not retrieved",
            f"hit_at_{k}": int(bool(rank and rank <= k)),
            "reciprocal_rank": round(1.0 / rank, 4) if rank else 0.0,
        })

    return {
        "n_queries": len(eval_rows),
        "k": k,
        f"hit_at_{k}": float(np.mean(hit_values)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "details": details,
    }


def save_eval_details(details: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write per-query evaluation results to CSV."""
    if not details:
        raise ValueError("No evaluation details to save.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)


def compare_title_embeddings(eval_rows: list[dict[str, str]], k: int = 5,
                             no_title_filename: str = "embeddings_no_title.npy") -> dict[str, Any]:
    """Compare default embeddings with precomputed embeddings made without titles."""
    with_title = evaluate_retrieval(eval_rows, k=k)
    no_title_path = rag_utils.DATA_DIR / no_title_filename
    if not no_title_path.exists():
        raise FileNotFoundError(
            f"Missing {no_title_path}. Run scripts/build_data.py to create both embedding files."
        )
    without_title_matrix = rag_utils.load_embeddings(no_title_filename)
    without_title = evaluate_retrieval(eval_rows, k=k, embeddings=without_title_matrix)
    return {"with_title": with_title, "without_title": without_title,
            "hit_at_k_delta": without_title[f"hit_at_{k}"] - with_title[f"hit_at_{k}"],
            "mrr_delta": without_title["mrr"] - with_title["mrr"]}


def print_eval_summary(result: dict[str, Any], label: str = "Retrieval evaluation") -> None:
    k = result["k"]
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Queries evaluated: {result['n_queries']}")
    print(f"Hit@{k}: {result[f'hit_at_{k}']:.3f} ({result[f'hit_at_{k}']:.1%})")
    print(f"MRR:    {result['mrr']:.3f}")


# ---------------------------------------------------------------------------
# Section 8 — Limitations and failure demonstrations
# ---------------------------------------------------------------------------

def show_semantic_results(query: str, k: int = 5) -> None:
    """Print semantic-search results, including scores, for a demo query."""
    print(f"\nSemantic search for: {query!r}")
    for rank, (module, score) in enumerate(rag_utils.search(query, k=k), start=1):
        print(f"{rank}. {module.get('moduleCode')} — {module.get('title')} (score={score:.3f})")


def show_filter_limitation(module_code: str = "4MC", semester: int = 2,
                           no_exam: bool = True, k: int = 5) -> dict[str, Any]:
    """Contrast semantic search with exact metadata filtering for a structured query.

    This deliberately shows that semantic retrieval is not the same as applying
    hard constraints. The metadata result is only as reliable as the source data.
    """
    query = f"{module_code}, semester {semester}, no exam"
    semantic = rag_utils.search(query, k=k)
    filtered = []
    for module in rag_utils.modules or []:
        code_ok = module_code.upper() in str(module.get("moduleCode", "")).upper()
        semesters = module.get("semesters", [])
        semester_ok = semester in semesters or str(semester) in [str(s) for s in semesters]
        exam_ok = (not module.get("hasExam", False)) if no_exam else True
        if code_ok and semester_ok and exam_ok:
            filtered.append(module)

    print(f"\nStructured/filter query: {query!r}")
    print("Semantic top results:")
    for rank, (module, score) in enumerate(semantic, start=1):
        print(f"  {rank}. {module.get('moduleCode')} — {module.get('title')} ({score:.3f})")
    print("\nExact metadata-filter results:")
    if filtered:
        for module in filtered:
            print(f"  {module.get('moduleCode')} — {module.get('title')}")
    else:
        print("  No matching modules in the loaded metadata.")
    return {"query": query, "semantic_results": semantic, "filtered_results": filtered}


def show_short_query_limitations() -> None:
    """Run example greetings/short queries and display the retrieved results."""
    for query in ("hi", "AI", "exam", "4MC, sem 2, no exam"):
        show_semantic_results(query, k=3)


def print_limitations_summary() -> None:
    print("""
RAG limitations to discuss:
1. Retrieval errors: relevant modules may not appear in the top-k results.
2. Generation errors: the LLM may misread context or make unsupported claims.
3. Garbage context: the generator can sound confident even when retrieved context is irrelevant.
4. Short/ambiguous queries: greetings and underspecified terms provide little semantic signal.
5. Structured constraints: semantic similarity does not reliably enforce exact semester/exam filters.
6. Data and time limits: results reflect the fetched NUSMods academic year and can become stale.

Possible improvements:
- Hybrid search combines semantic retrieval with keyword matching.
- Reranking reorders a larger candidate set using a stronger relevance model.
- Query rewriting expands or clarifies short/ambiguous user queries.
- Metadata filters enforce exact constraints such as semester or exam status.
""")


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_evaluation(csv_path: str | Path | None = None, k: int = 5,
                   query_col: str | None = None, target_col: str | None = None,
                   output_path: str | Path = "results/rag_eval_details.csv") -> dict[str, Any]:
    """Load the workshop assets, evaluate the labelled set, and save details."""
    if rag_utils.modules is None:
        rag_utils.load()
    if csv_path is None:
        csv_path = rag_utils.DATA_DIR / "eval_set.csv"
    eval_rows = load_eval_set(csv_path, query_col=query_col, target_col=target_col)
    result = evaluate_retrieval(eval_rows, k=k)
    print_eval_summary(result)
    save_eval_details(result["details"], output_path)
    print(f"Per-query results saved to: {output_path}")
    return result


if __name__ == "__main__":
    rag_utils.load()
    print("Loaded NUSMods data. Try one of these in a notebook:")
    print("  from rag_sections import generate_answer, run_evaluation")
    print("  generate_answer('Which NUS modules cover attention mechanisms?')")
    print("  run_evaluation('data/eval_set.csv')")
    print("  show_filter_limitation()")
