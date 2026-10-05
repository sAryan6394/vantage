# Vantage evaluation (2026-10-05)

- Questions: **120**, written by Gemini from 7 of the loaded sources, one per stored passage (so the correct passage is known).
- Hit@k: the passage a question was written from appears in the top k results. Percentages show the 95% confidence interval in brackets.
- Caveat: AI-written questions tend to reuse the passage's own words, which can flatter keyword matching. Treat these as relative comparisons between methods, not absolute accuracy.

## Full pipeline

- Requests: 120, errors: 0 (0.0%)
- Answered (not "I don't know"): 110 (92%)
- Correct passage in the 3 passages sent to the model: 71 (59%)
- Latency, cost per query and failures: run `python sql/run_analysis.py eval_telemetry.db`

