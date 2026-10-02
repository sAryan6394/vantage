import sqlite3
import uuid
import os
from datetime import datetime, timezone

DB_PATH = os.getenv("TELEMETRY_DB_PATH", "rag_telemetry.db")

# gemini-3.1-flash-lite pricing (Google's published rate):
# $0.25 / 1M input tokens, $1.50 / 1M output tokens.
# Update these two constants any time the generation model in app.py changes.
INPUT_PRICE_PER_1M_TOKENS = 0.25
OUTPUT_PRICE_PER_1M_TOKENS = 1.50

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_telemetry_logs (
    query_id VARCHAR(36) PRIMARY KEY,
    timestamp TEXT NOT NULL,
    user_query TEXT NOT NULL,
    retrieval_latency_ms REAL NOT NULL,
    generation_latency_ms REAL NOT NULL,
    top_similarity_score REAL NOT NULL,
    retrieved_chunks_count INTEGER NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    estimated_cost_usd REAL NOT NULL,
    user_feedback_rating INTEGER
);
"""


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def estimate_cost_usd(prompt_tokens, completion_tokens):
    return (
        (prompt_tokens / 1_000_000) * INPUT_PRICE_PER_1M_TOKENS
        + (completion_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M_TOKENS
    )


def log_query(
    user_query,
    retrieval_latency_ms,
    generation_latency_ms,
    top_similarity_score,
    retrieved_chunks_count,
    prompt_tokens,
    completion_tokens,
    db_path=DB_PATH,
):
    """
    Records one query's full telemetry row. Returns the generated query_id
    so the UI can later attach a thumbs up/down rating to this exact row
    via update_feedback() — the row is inserted with user_feedback_rating
    left NULL until (and unless) that happens.

    top_similarity_score should be the raw dense cosine similarity (0-1) of
    the single best-matching chunk, NOT the hybrid RRF fusion score or the
    cross-encoder's re-rank score — those aren't on a comparable 0-1 scale,
    and the <0.45 low-confidence threshold this schema is built around only
    means something against true cosine similarity.

    retrieved_chunks_count is the number of chunks that actually made it
    into the final LLM prompt (post-rerank), not the wider candidate pool
    hybrid search cast before re-ranking narrowed it down.
    """
    query_id = str(uuid.uuid4())
    cost = estimate_cost_usd(prompt_tokens, completion_tokens)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO rag_telemetry_logs (
                query_id, timestamp, user_query, retrieval_latency_ms,
                generation_latency_ms, top_similarity_score, retrieved_chunks_count,
                prompt_tokens, completion_tokens, estimated_cost_usd, user_feedback_rating
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                query_id,
                datetime.now(timezone.utc).isoformat(),
                user_query,
                retrieval_latency_ms,
                generation_latency_ms,
                top_similarity_score,
                retrieved_chunks_count,
                prompt_tokens,
                completion_tokens,
                cost,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return query_id


def update_feedback(query_id, rating, db_path=DB_PATH):
    """rating: 1 for thumbs up, -1 for thumbs down."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE rag_telemetry_logs SET user_feedback_rating = ? WHERE query_id = ?",
            (rating, query_id),
        )
        conn.commit()
    finally:
        conn.close()
