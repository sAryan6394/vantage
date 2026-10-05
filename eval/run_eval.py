"""Measures how well Vantage finds and answers questions from eval/questions.json.

Run from the project root:
  python eval/run_eval.py                      retrieval only: free, no Gemini calls
  python eval/run_eval.py --mode full          full pipeline: real answers, logged to
                                               eval_telemetry.db (your normal telemetry
                                               database is not touched)

"Hit" = the passage a question was written from is among the top-k results.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

KS = (1, 3, 5)


def gold_rank(results, gold):
    """1-based position of the gold passage in a list of (score, chunk), else None."""
    for i, (_score, chunk) in enumerate(results, 1):
        if (chunk["source"], chunk["location"]) == gold:
            return i
    return None


def wilson(k, n, z=1.96):
    """95% confidence interval for a hit rate of k hits out of n questions."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / d), min(1.0, (centre + margin) / d)


def summarize(ranks):
    """ranks: list of 1-based rank or None, one per question."""
    n = len(ranks)
    out = {"n": n}
    for k in KS:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        lo, hi = wilson(hits, n)
        out[f"hit@{k}"] = (hits / n if n else 0.0, lo, hi)
    out["mrr@5"] = sum(1 / r for r in ranks if r is not None and r <= 5) / n if n else 0.0
    return out


def fmt(t):
    return f"{100 * t[0]:.0f}% ({100 * t[1]:.0f}-{100 * t[2]:.0f})"


def table(summaries):
    lines = ["| Method | n | Hit@1 | Hit@3 | Hit@5 | MRR@5 |", "|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['n']} | {fmt(s['hit@1'])} | {fmt(s['hit@3'])} | "
                     f"{fmt(s['hit@5'])} | {s['mrr@5']:.2f} |")
    return "\n".join(lines)


def load_stack():
    from dotenv import load_dotenv
    from google import genai
    from sentence_transformers import SentenceTransformer
    from hybrid_search import HybridSearch
    from reranker import Reranker
    from vector_store import VectorStore
    load_dotenv()
    store = VectorStore()
    return {
        "model": SentenceTransformer("all-MiniLM-L6-v2"),
        "client": genai.Client(api_key=os.getenv("GOOGLE_API_KEY")),
        "vector_store": store,
        "hybrid_search": HybridSearch(store),
        "reranker": Reranker(),
    }


def retrieval_eval(questions, stack):
    """Compares three retrieval set-ups on the same questions."""
    ranks = {"Dense only": [], "Hybrid (dense + keyword)": [], "Hybrid + reranker (what the app uses)": []}
    rows = []
    for q in questions:
        gold = (q["source"], q["location"])
        emb = stack["model"].encode(q["question"])
        dense = stack["vector_store"].search(emb, top_k=5)
        hybrid = stack["hybrid_search"].search(q["question"], emb, top_k=5)
        cands = stack["hybrid_search"].search(q["question"], emb, top_k=15)
        reranked = stack["reranker"].rerank(q["question"], cands, top_k=5)
        r = [gold_rank(dense, gold), gold_rank(hybrid, gold), gold_rank(reranked, gold)]
        for name, rank in zip(ranks, r):
            ranks[name].append(rank)
        rows.append([q["id"], q["source"], q["location"], *r])
    return ranks, rows


def full_eval(questions, stack, db_path, sleep):
    import rag_pipeline
    import telemetry
    telemetry.init_db(db_path)
    rows = []
    for i, q in enumerate(questions, 1):
        gold = (q["source"], q["location"])
        res = rag_pipeline.answer_question(
            q["question"], [], model=stack["model"], client=stack["client"],
            vector_store=stack["vector_store"], hybrid_search=stack["hybrid_search"],
            reranker=stack["reranker"], db_path=db_path)
        answer = res["answer"] or ""
        rows.append({
            "id": q["id"],
            "status": "error" if res["error"] else "success",
            "answered": bool(answer) and not answer.lower().startswith("i don't know"),
            "gold_in_top3": gold_rank(res["top_chunks"], gold) is not None,
            "total_ms": round(res["total_latency_ms"]),
        })
        if i % 10 == 0:
            print(f"  {i}/{len(questions)} questions done")
        time.sleep(sleep)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between Gemini calls (full mode)")
    ap.add_argument("--db", default="eval_telemetry.db")
    args = ap.parse_args()

    with open(os.path.join(HERE, "questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        sys.exit("eval/questions.json is empty. Run: python eval/make_questions.py")

    stack = load_stack()
    n_src = len({q["source"] for q in questions})
    print(f"{len(questions)} questions from {n_src} sources, mode = {args.mode}\n")
    md = [f"# Vantage evaluation ({date.today().isoformat()})", "",
          f"- Questions: **{len(questions)}**, written by Gemini from {n_src} of the loaded sources, "
          "one per stored passage (so the correct passage is known).",
          "- Hit@k: the passage a question was written from appears in the top k results. "
          "Percentages show the 95% confidence interval in brackets.",
          "- Caveat: AI-written questions tend to reuse the passage's own words, which can flatter "
          "keyword matching. Treat these as relative comparisons between methods, not absolute accuracy.", ""]

    if args.mode == "retrieval":
        ranks, rows = retrieval_eval(questions, stack)
        summaries = {name: summarize(r) for name, r in ranks.items()}
        md += ["## Retrieval", "", table(summaries), ""]
        with open(os.path.join(HERE, "retrieval_results.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "source", "location", "rank_dense", "rank_hybrid", "rank_hybrid_rerank"])
            w.writerows(rows)
    else:
        rows = full_eval(questions, stack, args.db, args.sleep)
        n = len(rows)
        errors = sum(r["status"] == "error" for r in rows)
        answered = sum(r["answered"] for r in rows)
        found = sum(r["gold_in_top3"] for r in rows)
        md += ["## Full pipeline", "",
               f"- Requests: {n}, errors: {errors} ({100 * errors / n:.1f}%)",
               f"- Answered (not \"I don't know\"): {answered} ({100 * answered / n:.0f}%)",
               f"- Correct passage in the 3 passages sent to the model: {found} ({100 * found / n:.0f}%)",
               f"- Latency, cost per query and failures: run "
               f"`python sql/run_analysis.py {args.db}`", ""]
        with open(os.path.join(HERE, "full_results.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    text = "\n".join(md)
    print(text)
    with open(os.path.join(HERE, f"results_{args.mode}.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
