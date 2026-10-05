# Vantage evaluation (2026-10-05)

- Questions: **120**, written by Gemini from 7 of the loaded sources, one per stored passage (so the correct passage is known).
- Hit@k: the passage a question was written from appears in the top k results. Percentages show the 95% confidence interval in brackets.
- Caveat: AI-written questions tend to reuse the passage's own words, which can flatter keyword matching. Treat these as relative comparisons between methods, not absolute accuracy.

## Retrieval

| Method | n | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|
| Dense only | 120 | 32% (24-40) | 57% (49-66) | 62% (53-70) | 0.44 |
| Hybrid (dense + keyword) | 120 | 35% (27-44) | 59% (50-68) | 66% (57-74) | 0.47 |
| Hybrid + reranker (what the app uses) | 120 | 39% (31-48) | 59% (50-68) | 70% (61-77) | 0.51 |

