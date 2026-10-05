# Vantage: KPI definitions

One definition per metric, so the SQL (`sql/analysis.sql`), the Power BI measures and the README all
mean the same thing. Data source: the `rag_telemetry_logs` table, one row per question (failed
questions are rows too).

| KPI | Definition | Counts which rows | SQL | Power BI (DAX) |
|---|---|---|---|---|
| **Requests (n)** | Questions asked | All | query 1 | `Requests` |
| **Error rate** | Failed requests / all requests | All | query 1 | `Error Rate` |
| **End-to-end latency, P50 / P95** | Seconds from question received to answer ready, including retries and their waits. P95 = 95% of answers were at least this fast | Successful, with a total time | query 1 | `P50 Total (s)`, `P95 Total (s)` |
| **Stage breakdown** | Share of the average wait spent in embedding, retrieval, generation, and "other" (prompt building and retry waits) | Successful, with a total time | query 2 | build from the three stage columns |
| **Cost per query** | Average estimated USD for one successful answer | Successful | query 1 | `Cost per Query` |
| **Total spend** | Sum of estimated USD | All (failed rows cost 0) | query 1 | `Total Spend` |
| **Low-confidence rate** | Share of successful answers whose best-chunk similarity is below 0.45 | Successful | query 4 | `Low Confidence Rate` |
| **Helpful rate** | Thumbs-up / (thumbs-up + thumbs-down). Always shown with the number of rated answers | Successful, rated | query 4 | `Helpful Rate`, `Rated Answers` |
| **Retry rate** | Share of requests that needed at least one retry | With a total time | query 7 | `Retry Rate` |

## Power BI measures

Paste into Power BI Desktop (Modeling > New measure). Table name assumed: `fact_telemetry`.
These have not been run in Power BI. After pasting, check each value against the matching SQL
query on the same data before trusting it.

```dax
Requests = COUNTROWS ( fact_telemetry )

Successful Requests =
    CALCULATE ( COUNTROWS ( fact_telemetry ), fact_telemetry[request_status] = "success" )

Errors =
    CALCULATE ( COUNTROWS ( fact_telemetry ), fact_telemetry[request_status] = "error" )

Error Rate = DIVIDE ( [Errors], [Requests] )

P50 Total (s) =
    DIVIDE (
        CALCULATE ( PERCENTILE.INC ( fact_telemetry[total_latency_ms], 0.5 ),
                    fact_telemetry[request_status] = "success" ),
        1000 )

P95 Total (s) =
    DIVIDE (
        CALCULATE ( PERCENTILE.INC ( fact_telemetry[total_latency_ms], 0.95 ),
                    fact_telemetry[request_status] = "success" ),
        1000 )

Cost per Query =
    CALCULATE ( AVERAGE ( fact_telemetry[estimated_cost_usd] ),
                fact_telemetry[request_status] = "success" )

Total Spend = SUM ( fact_telemetry[estimated_cost_usd] )

Low Confidence Rate =
    DIVIDE (
        CALCULATE ( COUNTROWS ( fact_telemetry ),
                    fact_telemetry[request_status] = "success",
                    fact_telemetry[confidence_category] = "Low Confidence" ),
        [Successful Requests] )

Rated Answers =
    CALCULATE ( COUNTROWS ( fact_telemetry ),
                fact_telemetry[request_status] = "success",
                NOT ISBLANK ( fact_telemetry[user_feedback_rating] ) )

Helpful Rate =
    DIVIDE (
        CALCULATE ( COUNTROWS ( fact_telemetry ), fact_telemetry[user_feedback_rating] = 1 ),
        [Rated Answers] )

Retry Rate =
    DIVIDE (
        CALCULATE ( COUNTROWS ( fact_telemetry ),
                    fact_telemetry[generation_retries] > 0,
                    NOT ISBLANK ( fact_telemetry[total_latency_ms] ) ),
        CALCULATE ( COUNTROWS ( fact_telemetry ),
                    NOT ISBLANK ( fact_telemetry[total_latency_ms] ) ) )
```

`AVERAGE` and `PERCENTILE.INC` skip empty values, so older rows (logged before end-to-end timing
existed) drop out of the latency measures automatically.

## Caveats to state wherever these numbers appear

- **Sample size.** Every figure comes with its n. Small n means a rough estimate, not a result.
- **Older rows.** Rows logged before end-to-end timing was added have no total time. They are
  excluded from latency KPIs and still counted in request, error and cost KPIs.
- **"Source match" is not correctness.** The score is how closely the best retrieved passage matches
  the question. A high score with a wrong answer, or a low score with a right one, is possible.
- **The 0.45 threshold is a starting guess.** It is only meaningful if query 4 shows helpful rates
  differing between bands, with enough rated answers to say so.
- **Cost is an estimate** from token counts and the model's published price in `telemetry.py`.
- **Feedback is voluntary.** Rated answers are a subset and may not represent all answers.
