# Vantage dashboard: build spec

Two pages, built in Power BI Desktop. Each visual answers one question and carries a plain-English
subtitle, so the page can be read in under a minute. Metric meanings are in
[`kpi_definitions.md`](kpi_definitions.md); the theme is [`vantage_dark_theme.json`](vantage_dark_theme.json)
(View > Themes > Browse for themes).

## 1. Load the data

Use the anonymised export so nothing private reaches a screenshot:

```
python export_for_powerbi.py --sample          # writes sample_data/
```

Get Data > Text/CSV, load `fact_telemetry.csv`, `dim_date.csv`, `dim_query_status.csv`, and for page 2
`eval/retrieval_results.csv`. Relationship: `fact_telemetry[confidence_category]` to
`dim_query_status[confidence_category]` (many to one). `fact_telemetry[date]` to `dim_date[date]`.

For page 2, in Power Query select `retrieval_results`, select the three `rank_*` columns, then
Transform > Unpivot Columns. Rename the new columns to `Method` and `Rank`. Rename the values in `Method`
to "Dense only", "Hybrid", "Hybrid + reranker" (Transform > Replace Values).

## 2. Extra measures and columns

The core KPI measures are in `kpi_definitions.md`. Add these:

```dax
Avg Embedding (ms) =
    CALCULATE ( AVERAGE ( fact_telemetry[embedding_latency_ms] ), fact_telemetry[request_status] = "success" )

Avg Retrieval (ms) =
    CALCULATE ( AVERAGE ( fact_telemetry[retrieval_latency_ms] ), fact_telemetry[request_status] = "success",
                NOT ISBLANK ( fact_telemetry[total_latency_ms] ) )

Avg Generation (ms) =
    CALCULATE ( AVERAGE ( fact_telemetry[generation_latency_ms] ), fact_telemetry[request_status] = "success",
                NOT ISBLANK ( fact_telemetry[total_latency_ms] ) )

Median Total (s) =
    DIVIDE ( CALCULATE ( MEDIAN ( fact_telemetry[total_latency_ms] ), fact_telemetry[request_status] = "success" ), 1000 )

Sample Note = "n = " & FORMAT ( [Requests], "#,0" ) & " requests. 120-question test set written by AI from the loaded sources; one session."

-- calculated columns on fact_telemetry
Latency Bucket =
    SWITCH ( TRUE (),
        ISBLANK ( fact_telemetry[total_latency_ms] ), "No timing",
        fact_telemetry[total_latency_ms] < 3000, "1: under 3 s",
        fact_telemetry[total_latency_ms] < 5000, "2: 3-5 s",
        fact_telemetry[total_latency_ms] < 7000, "3: 5-7 s",
        fact_telemetry[total_latency_ms] < 10000, "4: 7-10 s",
        "5: over 10 s" )

Similarity Band = FLOOR ( fact_telemetry[top_similarity_score], 0.1 )

-- page 2 (table: retrieval_results after unpivot)
Questions Tested = COUNTROWS ( retrieval_results )

Hit@1 = DIVIDE ( COUNTROWS ( FILTER ( retrieval_results, NOT ISBLANK ( retrieval_results[Rank] ) && retrieval_results[Rank] <= 1 ) ), [Questions Tested] )
Hit@3 = DIVIDE ( COUNTROWS ( FILTER ( retrieval_results, NOT ISBLANK ( retrieval_results[Rank] ) && retrieval_results[Rank] <= 3 ) ), [Questions Tested] )
Hit@5 = DIVIDE ( COUNTROWS ( FILTER ( retrieval_results, NOT ISBLANK ( retrieval_results[Rank] ) && retrieval_results[Rank] <= 5 ) ), [Questions Tested] )

MRR@5 =
    AVERAGEX ( retrieval_results,
        IF ( ISBLANK ( retrieval_results[Rank] ) || retrieval_results[Rank] > 5, 0, 1 / retrieval_results[Rank] ) )
```

The `NOT ISBLANK` checks matter: in DAX an empty value counts as 0, so without them every miss would
count as a hit. None of these measures have been run in Power BI. After building, check each number
against `python sql/run_analysis.py eval_telemetry.db` and against `eval/results_*.md`.

## 3. Page 1: "Is Vantage reliable, fast and affordable?"

Canvas 1280 x 720, theme applied. Positions are x, y, width, height in pixels.

| # | Visual | Position | Fields | Subtitle (type it in) |
|---|---|---|---|---|
| 0 | Text box (title) | 24, 16, 800, 44 | "Vantage: how well does it perform?" | |
| 1 | Card | 24, 76, 232, 100 | `Requests` | Questions in this sample |
| 2 | Card | 272, 76, 232, 100 | `Error Rate` (format 0.0%) | Failed after automatic retries |
| 3 | Card | 520, 76, 232, 100 | `Median Total (s)` (0.0) | Typical wait for an answer |
| 4 | Card | 768, 76, 232, 100 | `P95 Total (s)` (0.0) | 19 in 20 answers arrive faster |
| 5 | Card | 1016, 76, 240, 100 | `Cost per Query` (format $0.0000) | Estimated, per answer |
| 6 | Stacked bar | 24, 196, 400, 300 | values: `Avg Embedding (ms)`, `Avg Retrieval (ms)`, `Avg Generation (ms)` | Where the waiting time goes |
| 7 | Clustered column | 440, 196, 420, 300 | axis `Latency Bucket`, values `Requests` (sort by axis) | How long do answers take? |
| 8 | Clustered column | 876, 196, 380, 300 | axis `Similarity Band`, values `Requests` | How closely do retrieved passages match? |
| 9 | Table | 24, 512, 760, 160 | `total_latency_ms`, `retrieval_latency_ms`, `generation_latency_ms`, `generation_retries`, `top_similarity_score`; Top N = 10 by `total_latency_ms` | The 10 slowest requests |
| 10 | Slicers | 800, 512, 456, 80 | `confidence_category`, `request_status` (tile style) | Filter |
| 11 | Text box (note) | 24, 684, 1232, 28 | `Sample Note` (use a card showing the measure, small font) | |

Use `Requests`, `Error Rate` and the other measures from `kpi_definitions.md`. Turn off the visual
header icons (the theme does this) and give every visual alt text for accessibility.

## 4. Page 2: "Does it find the right passage?"

| # | Visual | Position | Fields | Subtitle |
|---|---|---|---|---|
| 0 | Text box (title) | 24, 16, 900, 44 | "Retrieval: which method finds the right passage?" | |
| 1 | Clustered column | 24, 76, 760, 420 | axis `Method`; values `Hit@1`, `Hit@3`, `Hit@5` (format 0%) | Share of questions where the source passage is in the top k |
| 2 | Card | 800, 76, 220, 100 | `Questions Tested` | |
| 3 | Table | 800, 196, 456, 180 | `Method`, `MRR@5` | Higher is better (max 1.0) |
| 4 | Text box | 800, 392, 456, 104 | "The differences are small and the 95% confidence intervals overlap, so treat them as a trend. 'Hit' only counts the exact source passage; another passage may also answer the question." | |
| 5 | Text box | 24, 512, 1232, 80 | Method note: "120 questions written by Gemini, one per stored passage. Keyword matching can be flattered because AI-written questions reuse the passage's words." | |

## 5. Before taking screenshots

- Check the card values against `python sql/run_analysis.py eval_telemetry.db` (expected: 120 requests, 0%
  errors, median 4.8 s, P95 9.6 s, cost $0.0004) and the page 2 values against `eval/results_retrieval.md`.
- Save as `vantage_dashboard.pbix`, replacing the old file. Export each page as an image
  (File > Export > Export to PDF, then screenshot) for the README.
- If a value looks wrong, fix the measure, not the chart.
