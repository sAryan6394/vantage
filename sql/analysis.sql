-- Vantage telemetry analysis (SQLite).
-- Table: rag_telemetry_logs  (one row per question; failed requests are rows too).
-- Run all queries:  python sql/run_analysis.py
-- Every query reports its sample size (n) so no number is read without it.
-- Metric definitions are in docs/kpi_definitions.md.


-- name: 1. Headline KPIs
-- Business question: how reliable, fast and expensive is the assistant overall?
-- Percentiles use the nearest-rank method (SQLite has no PERCENTILE function):
-- the P95 is the smallest latency whose rank is at least 95% of the way up.
WITH ok AS (
    SELECT total_latency_ms AS v,
           ROW_NUMBER() OVER (ORDER BY total_latency_ms) AS rn,
           COUNT(*) OVER ()                              AS n
    FROM rag_telemetry_logs
    WHERE request_status = 'success' AND total_latency_ms IS NOT NULL
),
pct AS (
    SELECT MIN(CASE WHEN rn * 100 >= 50 * n THEN v END) AS p50_ms,
           MIN(CASE WHEN rn * 100 >= 95 * n THEN v END) AS p95_ms,
           MAX(n)                                       AS latency_n
    FROM ok
),
agg AS (
    SELECT COUNT(*)                                            AS requests,
           SUM(request_status = 'error')                       AS errors,
           AVG(CASE WHEN request_status = 'success'
                    THEN estimated_cost_usd END)               AS cost_per_query,
           SUM(estimated_cost_usd)                             AS total_cost
    FROM rag_telemetry_logs
)
SELECT requests,
       errors,
       ROUND(100.0 * errors / requests, 1)  AS error_rate_pct,
       ROUND(p50_ms / 1000.0, 2)            AS p50_total_s,
       ROUND(p95_ms / 1000.0, 2)            AS p95_total_s,
       latency_n                            AS latency_n,
       ROUND(cost_per_query, 6)             AS cost_per_query_usd,
       ROUND(total_cost, 4)                 AS total_cost_usd
FROM agg, pct;


-- name: 2. Where does the waiting time go?
-- Business question: which stage should we optimise first?
-- Successful requests that have an end-to-end time (newer rows only).
WITH t AS (
    SELECT * FROM rag_telemetry_logs
    WHERE request_status = 'success' AND total_latency_ms IS NOT NULL
),
avgs AS (
    SELECT COUNT(*)                     AS n,
           AVG(embedding_latency_ms)    AS embed_ms,
           AVG(retrieval_latency_ms)    AS retrieval_ms,
           AVG(generation_latency_ms)   AS generation_ms,
           AVG(total_latency_ms)        AS total_ms
    FROM t
)
SELECT n,
       ROUND(embed_ms)      AS embed_ms,
       ROUND(retrieval_ms)  AS retrieval_ms,
       ROUND(generation_ms) AS generation_ms,
       ROUND(total_ms)      AS total_ms,
       ROUND(100.0 * embed_ms      / total_ms, 1) AS embed_pct,
       ROUND(100.0 * retrieval_ms  / total_ms, 1) AS retrieval_pct,
       ROUND(100.0 * generation_ms / total_ms, 1) AS generation_pct,
       ROUND(100.0 * (total_ms - embed_ms - retrieval_ms - generation_ms) / total_ms, 1)
           AS other_pct   -- prompt building plus retry waits
FROM avgs;


-- name: 3. Daily trend with rolling averages
-- Business question: is usage, speed or spend changing over time?
-- The "7-day" average is over the last 7 ACTIVE days (days with no queries have no row).
WITH daily AS (
    SELECT substr(timestamp, 1, 10) AS day,
           COUNT(*)                                              AS requests,
           SUM(request_status = 'error')                         AS errors,
           AVG(CASE WHEN request_status = 'success'
                    THEN total_latency_ms END)                   AS avg_total_ms,
           SUM(estimated_cost_usd)                               AS cost_usd
    FROM rag_telemetry_logs
    GROUP BY day
)
SELECT day,
       requests,
       errors,
       ROUND(avg_total_ms)                                        AS avg_total_ms,
       ROUND(AVG(avg_total_ms) OVER (ORDER BY day
             ROWS BETWEEN 6 PRECEDING AND CURRENT ROW))           AS avg_total_ms_7active,
       ROUND(cost_usd, 5)                                         AS cost_usd,
       ROUND(SUM(cost_usd) OVER (ORDER BY day), 5)                AS cumulative_cost_usd,
       requests - LAG(requests) OVER (ORDER BY day)               AS requests_vs_prev_day
FROM daily
ORDER BY day;


-- name: 4. Does a low match score predict a bad answer?
-- Business question: is the 0.45 low-confidence threshold meaningful?
-- Groups answered questions into 0.1-wide similarity bands and compares the
-- share of thumbs-up among rated answers. Read "rated" before the percentage.
WITH answered AS (
    SELECT CAST(top_similarity_score * 10 AS INTEGER) / 10.0 AS band_start,
           user_feedback_rating
    FROM rag_telemetry_logs
    WHERE request_status = 'success'
)
SELECT band_start || ' - ' || (band_start + 0.1)               AS similarity_band,
       COUNT(*)                                                AS answers,
       COUNT(user_feedback_rating)                             AS rated,
       SUM(user_feedback_rating = 1)                           AS helpful,
       SUM(user_feedback_rating = -1)                          AS not_helpful,
       ROUND(100.0 * SUM(user_feedback_rating = 1)
             / NULLIF(COUNT(user_feedback_rating), 0), 1)      AS helpful_pct_of_rated
FROM answered
GROUP BY band_start
ORDER BY band_start;


-- name: 5. What do the slowest requests look like?
-- Business question: are slow answers slow because of long prompts, weak retrieval, or the model?
-- NTILE(4) splits successful requests into latency quartiles (1 = fastest, 4 = slowest).
WITH t AS (
    SELECT *, NTILE(4) OVER (ORDER BY total_latency_ms) AS quartile
    FROM rag_telemetry_logs
    WHERE request_status = 'success' AND total_latency_ms IS NOT NULL
)
SELECT quartile                              AS latency_quartile,
       COUNT(*)                              AS n,
       ROUND(MIN(total_latency_ms))          AS min_total_ms,
       ROUND(MAX(total_latency_ms))          AS max_total_ms,
       ROUND(AVG(retrieval_latency_ms))      AS avg_retrieval_ms,
       ROUND(AVG(generation_latency_ms))     AS avg_generation_ms,
       ROUND(AVG(prompt_tokens))             AS avg_prompt_tokens,
       ROUND(AVG(completion_tokens))         AS avg_completion_tokens,
       ROUND(AVG(top_similarity_score), 2)   AS avg_similarity
FROM t
GROUP BY quartile
ORDER BY quartile;


-- name: 6. What fails, and how often?
-- Business question: which failures matter and should we handle them better?
SELECT CASE WHEN instr(error_message, ':') > 0
            THEN substr(error_message, 1, instr(error_message, ':') - 1)
            ELSE COALESCE(error_message, 'unknown') END           AS error_type,
       COUNT(*)                                                    AS failures,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)          AS share_of_failures_pct,
       ROUND(AVG(total_latency_ms))                                AS avg_wait_before_failing_ms,
       MIN(substr(timestamp, 1, 10))                               AS first_seen,
       MAX(substr(timestamp, 1, 10))                               AS last_seen
FROM rag_telemetry_logs
WHERE request_status = 'error'
GROUP BY error_type
ORDER BY failures DESC;


-- name: 7. What do retries cost the user?
-- Business question: are automatic retries worth their extra waiting time?
SELECT generation_retries                    AS retries_used,
       COUNT(*)                              AS n,
       SUM(request_status = 'success')       AS recovered,
       ROUND(AVG(total_latency_ms))          AS avg_total_ms
FROM rag_telemetry_logs
WHERE total_latency_ms IS NOT NULL
GROUP BY generation_retries
ORDER BY generation_retries;
