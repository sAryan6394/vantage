"""Writes a test set of questions from YOUR OWN stored passages.

Each question is generated from one passage and remembers which one, so
run_eval.py can check whether retrieval finds it again. Run from the project
root:   python eval/make_questions.py [--n 120] [--seed 42]
Uses about one short Gemini call per question (a few cents in total).
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from google import genai

import rag_pipeline
from vector_store import VectorStore

PROMPT = """Below is a passage from a document or video transcript.
Write ONE question that a curious reader could ask and that this passage answers.

Rules:
- The question must make sense on its own: never say "the passage", "the text",
  "the video", "the speaker" or "this document".
- Use your own wording; don't copy long phrases from the passage.
- Ask about something specific, not "what is this about?".
- Reply with the question only.

Passage:
{text}
"""


def ask_question(client, text, attempts=4):
    """One question for one passage. Retries with a growing pause (a long one if
    the API says we're going too fast). Returns (question, last_error_message)."""
    last = None
    for attempt in range(attempts):
        try:
            r = client.models.generate_content(
                model=rag_pipeline.GENERATION_MODEL,
                contents=PROMPT.format(text=text[:1500]),
            )
            return (r.text or "").strip().strip('"'), None
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:200]}"
            too_fast = "429" in last or "RESOURCE_EXHAUSTED" in last
            if attempt < attempts - 1:
                time.sleep((20 if too_fast else 3) * (attempt + 1))
    return "", last


def pick_chunks(chunks, n, seed, min_chars=300):
    """Round-robin across sources so one big source can't dominate the test set."""
    rng = random.Random(seed)
    by_source = {}
    for c in chunks:
        if len(c["text"]) >= min_chars:
            by_source.setdefault(c["source"], []).append(c)
    for lst in by_source.values():
        rng.shuffle(lst)
    picked = []
    while len(picked) < n and any(by_source.values()):
        for src in sorted(by_source):
            if by_source[src] and len(picked) < n:
                picked.append(by_source[src].pop())
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    load_dotenv()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    store = VectorStore()
    if not len(store):
        sys.exit("No sources loaded. Add some in the app first.")

    chunks = pick_chunks(store.chunks, args.n, args.seed)
    questions = []
    seen_errors = set()
    for i, c in enumerate(chunks, 1):
        q, err = ask_question(client, c["text"])
        if err:
            if err not in seen_errors:       # show each distinct error once
                seen_errors.add(err)
                print(f"  chunk {i} failed: {err}")
        elif q:
            questions.append({"id": len(questions) + 1, "question": q,
                              "source": c["source"], "location": c["location"]})
        if i % 10 == 0:
            print(f"  {i}/{len(chunks)} passages done, {len(questions)} questions")
        time.sleep(args.sleep)   # always pause, including after a failure

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=1)
    n_src = len({q["source"] for q in questions})
    print(f"\nWrote {len(questions)} questions from {n_src} sources -> {out}")


if __name__ == "__main__":
    main()
