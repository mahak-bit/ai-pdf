"""Evaluate retrieval (and optionally answers) against a labelled question set.

Retrieval metrics need no LLM:
  - hit@1 / hit@k: an expected page appears in the top 1 / top k passages
  - MRR: mean reciprocal rank of the first passage from an expected page
  - abstention: whether the relevance floor rejects every passage (expected for
    unanswerable questions, not expected for answerable ones)
  - a threshold sweep showing the trade-off between false and missed abstentions

With --llm, each question also goes through the configured model and we report the
status (answered/abstained) and whether every citation points to an expected page.
Nothing here is hardcoded: all numbers come from running the pipeline.

Usage (from backend/):
  python scripts/evaluate.py
  python scripts/evaluate.py --llm --out eval/results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunker import chunk_pages  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.embeddings import SentenceTransformerEmbedder  # noqa: E402
from app.llm import build_llm_client  # noqa: E402
from app.pdf_parser import extract_pages  # noqa: E402
from app.rag import RAGPipeline, SearchDoc  # noqa: E402
from app.vector_store import DocumentIndex  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default=str(ROOT / "eval" / "sample_eval.json"))
    parser.add_argument("--llm", action="store_true", help="also generate answers with the configured LLM")
    parser.add_argument("--out", help="write per-question results to this JSON file")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between LLM calls (free-tier rate limits)")
    args = parser.parse_args()

    settings = get_settings()
    spec = json.loads(Path(args.set).read_text(encoding="utf-8"))
    parsed = extract_pages((ROOT / spec["document"]).read_bytes(), settings.max_pages)
    chunks = chunk_pages(parsed.pages, settings.chunk_max_chars, settings.chunk_overlap_chars)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    index = DocumentIndex.build(chunks, embedder.embed([c.text for c in chunks]))
    k = settings.top_k

    rows = []
    for item in spec["questions"]:
        query = embedder.embed([item["question"]])[0]
        hits = index.search(query, k)
        expected = set(item["pages"])
        rank = next((i for i, h in enumerate(hits, start=1) if h.chunk.page in expected), None)
        rows.append({
            "question": item["question"],
            "expected_pages": item["pages"],
            "answerable": bool(expected),
            "top_pages": [h.chunk.page for h in hits],
            "top_score": round(hits[0].score, 4) if hits else 0.0,
            "rank": rank,
        })

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    floor = settings.min_relevance

    print(f"Document: {spec['document']}  ({parsed.page_count} pages, {len(chunks)} chunks)")
    print(f"Questions: {len(answerable)} answerable, {len(unanswerable)} unanswerable; k={k}; floor={floor}\n")
    print("Retrieval (answerable questions)")
    print(f"  hit@1  {sum(r['rank'] == 1 for r in answerable) / len(answerable):.3f}")
    print(f"  hit@{k}  {sum(r['rank'] is not None for r in answerable) / len(answerable):.3f}")
    print(f"  MRR    {sum(1 / r['rank'] for r in answerable if r['rank']) / len(answerable):.3f}\n")

    print("Top-1 similarity")
    for label, group in (("answerable", answerable), ("unanswerable", unanswerable)):
        scores = sorted(r["top_score"] for r in group)
        print(f"  {label:<13} min {scores[0]:.3f}  median {scores[len(scores) // 2]:.3f}  max {scores[-1]:.3f}")

    print("\nRelevance-floor sweep (abstain when top-1 score < floor)")
    print("  floor  answerable kept  unanswerable abstained")
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        kept = sum(r["top_score"] >= t for r in answerable)
        rejected = sum(r["top_score"] < t for r in unanswerable)
        marker = "  <- current" if abs(t - floor) < 1e-9 else ""
        print(f"  {t:.2f}   {kept:>2}/{len(answerable)}             {rejected:>2}/{len(unanswerable)}{marker}")

    misses = [r for r in answerable if r["rank"] != 1]
    if misses:
        print("\nAnswerable questions where the top passage was not from an expected page:")
        for r in misses:
            print(f"  - {r['question']}  expected {r['expected_pages']}, got {r['top_pages']}")

    if args.llm:
        llm = build_llm_client(settings)
        if llm is None:
            print("\n--llm given but no LLM is configured (set GEMINI_API_KEY or run Ollama). Skipping.")
        else:
            pipeline = RAGPipeline(embedder, llm, k, floor)
            print(f"\nAnswer generation with {llm.name}:{llm.model}")
            for i, r in enumerate(rows):
                if i and args.delay:
                    time.sleep(args.delay)
                try:
                    result = pipeline.answer([SearchDoc("eval00000000", spec["document"], index)], r["question"])
                except Exception as exc:  # report and keep going
                    r.update(status="error", error=str(exc))
                    continue
                cited_pages = sorted({s.page for s in result.sources})
                r.update(status=result.status, answer=result.answer, cited_pages=cited_pages,
                         citations_on_expected_page=bool(cited_pages) and set(cited_pages) <= set(r["expected_pages"]))
            ans = [r for r in rows if r["answerable"] and r.get("status") != "error"]
            una = [r for r in rows if not r["answerable"] and r.get("status") != "error"]
            errors = [r for r in rows if r.get("status") == "error"]
            print(f"  answerable answered                 {sum(r['status'] == 'answered' for r in ans)}/{len(ans)}")
            print(f"  answered with all citations on expected pages  "
                  f"{sum(r.get('citations_on_expected_page', False) for r in ans)}/{len(ans)}")
            print(f"  unanswerable correctly abstained    {sum(r['status'] == 'abstained' for r in una)}/{len(una)}")
            if errors:
                print(f"  errors: {len(errors)} (first: {errors[0]['error']})")
            print("  Answer correctness is not scored automatically; review the 'answer' fields in --out.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
