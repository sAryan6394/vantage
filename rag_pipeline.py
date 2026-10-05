"""The question-answering pipeline: embed -> retrieve -> rerank -> generate -> log.

Shared by the Streamlit app and the batch evaluator (eval/run_eval.py), so
both run exactly the same code and the evaluation numbers describe the app.
"""
import time

import telemetry

# gemini-3.1-flash-lite: ~3x the rate limit of 3.6-flash and far cheaper, fine
# for grounded QA over retrieved context. Retry covers transient 503s.
# If you change this model, update the prices in telemetry.py too.
GENERATION_MODEL = "gemini-3.1-flash-lite"
MAX_RETRIES = 2


def build_contextual_query(current_question, history, max_history=2):
    recent_user_msgs = [m["content"] for m in history if m["role"] == "user"][-max_history:]
    return " ".join(recent_user_msgs + [current_question])


def format_history(history, max_turns=4):
    recent = history[-(max_turns * 2):]
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def build_prompt(question, history, top_chunks):
    combined_context = "\n\n".join(
        f"[Source: {c['source']} @ {c['location']}]\n{c['text']}"
        for score, c in top_chunks
    )
    history_text = format_history(history)

    return f"""You are answering questions using ONLY the context provided below.

Conversation so far:
{history_text if history_text else "(no earlier messages)"}

Context from sources:
{combined_context}

Current question: {question}

Instructions: If the current question is a follow-up (like "explain more", "why", "what about that", etc.), use the conversation above to understand what it's referring to, then answer using the context. Write a clear, complete, well-explained answer in plain prose — no source citations, no brackets, no meta-commentary about where the information came from. If the answer genuinely isn't in the context, say "I don't know based on the provided notes."

Answer:"""


def retrieve(question, history, *, model, vector_store, hybrid_search, reranker):
    """Embedding + retrieval, timed separately. Returns a dict."""
    t0 = time.perf_counter()
    contextual_query = build_contextual_query(question, history)
    query_embedding = model.encode(contextual_query)
    embedding_latency_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    # Separate, cheap dense-only lookup purely for a true 0-1 cosine similarity
    # score to log — the hybrid/reranked results use RRF and cross-encoder
    # scores, which aren't on a comparable scale.
    dense_top = vector_store.search(query_embedding, top_k=1)
    top_similarity_score = dense_top[0][0] if dense_top else 0.0

    candidates = hybrid_search.search(contextual_query, query_embedding, top_k=15)
    top_chunks = reranker.rerank(contextual_query, candidates, top_k=3)
    retrieval_latency_ms = (time.perf_counter() - t1) * 1000

    return {
        "top_chunks": top_chunks,
        "top_similarity_score": top_similarity_score,
        "embedding_latency_ms": embedding_latency_ms,
        "retrieval_latency_ms": retrieval_latency_ms,
    }


def _generate(client, prompt):
    """One question to the model, with retries. generation_latency_ms is the
    last model call only (retry waits are not included)."""
    last_error = None
    generation_latency_ms = 0.0
    response = None
    attempt = 0
    for attempt in range(MAX_RETRIES + 1):
        attempt_start = time.perf_counter()
        try:
            response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
            last_error = None
            generation_latency_ms = (time.perf_counter() - attempt_start) * 1000
            break
        except Exception as e:
            last_error = e
            generation_latency_ms = (time.perf_counter() - attempt_start) * 1000
            print(f"[Vantage] Generation attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    return response, last_error, generation_latency_ms, attempt


def answer_question(question, history, *, model, client, vector_store, hybrid_search,
                    reranker, db_path=None, log=True):
    """Runs the whole pipeline for one question and logs it to telemetry.

    Returns a dict: answer (None on failure), error (None on success),
    query_id, top_similarity_score, top_chunks, the four latencies, retries.
    db_path=None logs to the normal telemetry database.
    """
    request_start = time.perf_counter()
    r = retrieve(question, history, model=model, vector_store=vector_store,
                 hybrid_search=hybrid_search, reranker=reranker)
    prompt = build_prompt(question, history, r["top_chunks"])
    response, last_error, generation_latency_ms, retries = _generate(client, prompt)
    total_latency_ms = (time.perf_counter() - request_start) * 1000

    answer, error = None, None
    prompt_tokens = completion_tokens = 0
    if last_error is not None:
        error = f"{type(last_error).__name__}: {str(last_error)[:200]}"
    elif not getattr(response, "text", None):
        error = "EmptyResponse: the model returned no text"
    else:
        answer = response.text
        usage = response.usage_metadata
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

    result = {
        "answer": answer,
        "error": error,
        "query_id": None,
        "top_similarity_score": r["top_similarity_score"],
        "top_chunks": r["top_chunks"],
        "embedding_latency_ms": r["embedding_latency_ms"],
        "retrieval_latency_ms": r["retrieval_latency_ms"],
        "generation_latency_ms": generation_latency_ms,
        "total_latency_ms": total_latency_ms,
        "retries": retries,
    }

    if log:
        try:
            kwargs = {"db_path": db_path} if db_path else {}
            result["query_id"] = telemetry.log_query(
                user_query=question,
                retrieval_latency_ms=r["retrieval_latency_ms"],
                generation_latency_ms=generation_latency_ms,
                top_similarity_score=r["top_similarity_score"],
                retrieved_chunks_count=len(r["top_chunks"]),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                embedding_latency_ms=r["embedding_latency_ms"],
                total_latency_ms=total_latency_ms,
                request_status="error" if error else "success",
                error_message=error,
                generation_retries=retries,
                **kwargs,
            )
        except Exception as e:
            # Telemetry is a nice-to-have, not worth losing an answer over.
            print(f"[Vantage] Telemetry logging failed: {e}")
    return result
