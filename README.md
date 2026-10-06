# Vantage

Ask questions across your own PDFs, text files and YouTube lectures and get answers grounded in
those sources. Vantage also measures itself: every question is logged (speed, cost, retrieval
quality, failures, user feedback), so you can see how the assistant performs and where it falls
short.

> **Screenshots and demo:** _add here: app screenshot, dashboard screenshot, link to a 60-90 second demo video._

## What I measured

All numbers come from this repo's own evaluation scripts and telemetry. Sample sizes are shown
next to every figure. See [Limitations](#limitations) before quoting any of them.

**Retrieval quality** (120 questions; "hit" = the passage a question was written from is among
the top results; 95% confidence interval in brackets):

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|---|
| Dense vector search only | 32% (24-40) | 57% (49-66) | 62% (53-70) | 0.44 |
| Hybrid (vector + keyword) | 35% (27-44) | 59% (50-68) | 66% (57-74) | 0.47 |
| Hybrid + cross-encoder reranker (used by the app) | 39% (31-48) | 59% (50-68) | 70% (61-77) | 0.51 |

The ordering is consistent, but the intervals overlap, so these differences are a trend, not a proven
effect. The app sends 3 passages to the model, and at 3 the hybrid and reranked results are the same.

**End-to-end behaviour** (the same 120 questions run through the real pipeline, one session):

| Metric | Result |
|---|---|
| Requests that failed after automatic retries | 0 of 120 |
| Requests that needed a retry (temporary Gemini 503s) | 5 (4.2%), all recovered |
| Time to answer, median / 95th percentile | 4.8 s / 9.6 s |
| Share of the wait spent in the model call | 87% (retrieval 10%) |
| Cost per question (estimated) | $0.00038 ($0.046 for all 120) |
| Answered rather than "I don't know" | 92% |
| Correct source passage among the 3 sent to the model | 59% |

The slowest quarter of requests spent about 7.9 s in the model call against about 2.8 s for the fastest
quarter, while retrieval time, prompt size and answer length were the same. The variation comes from the
model's response time, not from Vantage's own code.

## How it works

1. **Ingest.** PDFs (including scanned pages via OCR), plain text and YouTube transcripts are split into
   sentence-aware chunks (about 500 characters with overlap), each tagged with its source and location.
2. **Index.** Each chunk is embedded (`all-MiniLM-L6-v2`) into a FAISS index. A BM25 keyword index is built
   from the same chunks.
3. **Retrieve.** A question is searched by meaning (FAISS) and by keywords (BM25). The two ranked lists are
   merged with Reciprocal Rank Fusion, giving 15 candidates.
4. **Rerank.** A cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores the candidates and keeps the best 3.
5. **Answer.** Gemini (`gemini-3.1-flash-lite`) answers using only those passages, or says it doesn't know.
6. **Log.** Timings, tokens, estimated cost, match score, status and later user feedback are saved to SQLite.

## Analytics

Every question becomes one row in `rag_telemetry_logs` (failed requests too):

- **Latency in four parts:** embedding, retrieval, one model call, and end-to-end (including retries and their
  waits).
- **Cost:** estimated from token counts and the model's published price (`telemetry.py`).
- **Quality signals:** the match score of the best passage, and optional thumbs up/down.
- **Reliability:** request status, error type and retry count.

Where to look:

| What | Where |
|---|---|
| Metric definitions (one meaning per KPI, plus Power BI measures) | [`docs/kpi_definitions.md`](docs/kpi_definitions.md) |
| Seven SQL queries (CTEs, window functions, percentiles) | [`sql/analysis.sql`](sql/analysis.sql) |
| Run them on your data | `python sql/run_analysis.py` |
| Power BI export (question text removable) | `python export_for_powerbi.py [--sample]` |
| Anonymised sample data (question text removed) | [`sample_data/`](sample_data/) |
| Dashboard file | `vantage_dashboard.pbix` _(being rebuilt)_ |

## Evaluation

```bash
python eval/make_questions.py        # Gemini writes ~120 questions from your own stored passages
python eval/run_eval.py              # retrieval comparison; free, no model calls
python eval/run_eval.py --mode full  # full pipeline, logged to eval_telemetry.db
python sql/run_analysis.py eval_telemetry.db
```

Results are saved in [`eval/`](eval/) (`results_retrieval.md`, `results_full.md`, `questions.json`).

## Limitations

- **The questions are AI-written** from the same passages being searched. They tend to reuse the passage's own
  words, which can flatter keyword search, and about one in ten is a poor question. Use the results to compare
  methods against each other, not as an absolute accuracy figure.
- **"Hit" is strict.** It counts only the exact passage a question came from. The sources overlap in topic, so
  another passage may have answered the question just as well. A 59% hit rate does not mean 41% wrong answers.
- **Answer correctness is not measured.** "Answered" means the model did not say "I don't know". Whether the answers are
  right has not been checked.
- **The 0.45 low-confidence threshold is unvalidated.** No user ratings exist yet for these test runs, so there
  is no evidence yet that the score separates good answers from bad ones.
- **One session, one model, one set of sources.** Speed and error figures depend on Gemini's load at the
  time (the 503 errors came and went). Cost is an estimate from published prices.
- **Everything runs locally.** There is no deployed version.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Needs a `.env` file with `GOOGLE_API_KEY=your_key_here`, and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
installed for scanned PDFs. The telemetry database and your chat history stay on your machine and are
git-ignored.

## Project layout

```
app.py              Streamlit interface
rag_pipeline.py     embed -> retrieve -> rerank -> answer -> log (shared by the app and the evaluator)
ingestion.py        PDF / text / YouTube loading and chunking
vector_store.py     FAISS index and chunk storage
hybrid_search.py    dense + BM25 search, Reciprocal Rank Fusion
reranker.py         cross-encoder re-ranking
telemetry.py        SQLite logging, cost estimate
export_for_powerbi.py   CSV export (star schema) for Power BI
sql/                analysis queries and a runner
docs/               KPI definitions
eval/               question generator, evaluator, results
sample_data/        anonymised sample of the telemetry
```

## Stack

Python, Streamlit, sentence-transformers, FAISS, rank-bm25, SQLite, Google Gemini API, PyMuPDF, pytesseract,
youtube-transcript-api, Power BI.
