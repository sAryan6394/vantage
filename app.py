import streamlit as st
import streamlit.components.v1 as components
import os
import pickle
import tempfile
import re
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from ingestion import ingest_text_file, ingest_youtube, ingest_pdf
from vector_store import VectorStore
from hybrid_search import HybridSearch
from reranker import Reranker
import telemetry

load_dotenv()

st.set_page_config(page_title="Vantage", page_icon="🔎", layout="wide")


# Styling only. No fixed heights, no flex chains, no absolute positioning:
# Streamlit's own layout scrolls the page and pins the chat input to the
# bottom of the window, so nothing here can overlap or clip.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;900&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --paper: #EDF0E7;
        --card: #FFFFFF;
        --ink: #000000;
        --stamp: #B0182F;
        --highlight: #F5D949;
        --muted: #5B5F55;
        /* distance from window top to the frame; the script below measures
           the real value, this is only the starting guess */
        --top: 120px;
    }

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {display: none;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid^="stChatMessageAvatar"] {display: none !important;}

    /* The page itself never scrolls; only the chat panel does. */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main {
        overflow: hidden !important;
        background-color: var(--paper);
    }
    .block-container {
        max-width: 100% !important;
        padding: 1.1rem 1.6rem 1rem !important;
    }

    .st-key-header_row {
        border-bottom: 3px solid var(--ink);
        padding-bottom: 0.8rem;
    }
    .st-key-header_row [data-testid="stHorizontalBlock"] { align-items: center; }

    /* The frame. Divider between the two columns is a background line, so it
       always spans the full frame height whatever the columns contain. */
    [class*="st-key-body_row"] {
        border: 3px solid var(--ink);
        /* frame runs from its top edge to 18px above the window bottom */
        height: calc(100vh - var(--top) - 18px);
    }
    .st-key-body_row {
        background: linear-gradient(var(--ink), var(--ink)) no-repeat 28% 0 / 3px 100%;
    }

    .st-key-sources_col_inner {
        padding: 1.1rem 1.2rem;
        max-height: calc(100vh - var(--top) - 24px);
        overflow-y: auto;
    }
    .st-key-chat_col_inner { padding: 1.1rem 1.4rem; }

    /* Chat box = frame height minus frame border (6), chat padding (35),
       gap (16) and the input bar (59). Pure CSS, so no timing issues. */
    .st-key-chat_scroll {
        height: calc(100vh - var(--top) - 134px);
        overflow-y: auto;
        flex: 0 0 auto;
    }

    .app-header {
        font-family: 'Archivo', sans-serif;
        font-size: 2.6rem;
        font-weight: 900;
        line-height: 1.05;
        color: var(--ink);
        letter-spacing: -0.01em;
    }
    .app-subtitle {
        color: var(--muted);
        font-size: 0.88rem;
        margin-top: 0.3rem;
    }

    .msg-index {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0.6rem 0 0.15rem 0.1rem;
    }

    [data-testid="stChatMessage"] {
        background-color: var(--card);
        border: 2px solid var(--ink);
        border-radius: 0;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.9rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: transparent;
        border: none;
        padding: 0.2rem 0.1rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
        color: var(--stamp);
        font-size: 0.95rem;
        font-weight: 600;
    }

    a, a:visited { color: var(--stamp); font-weight: 600; }

    .st-key-sources_fixed_top h3 {
        color: var(--ink);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0;
        padding-bottom: 0.3rem;
    }

    .stButton button {
        border-radius: 0;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        background-color: var(--ink);
        color: var(--paper);
        border: 2px solid var(--ink);
    }
    .stButton button:hover {
        background-color: var(--paper);
        border-color: var(--ink);
        color: var(--ink);
    }
    .stButton button[kind="primary"] {
        background-color: var(--stamp);
        color: #FFFFFF;
        border: 2px solid var(--stamp);
    }
    .stButton button[kind="primary"]:hover {
        background-color: var(--paper);
        border-color: var(--stamp);
        color: var(--stamp);
    }
    /* small red x next to each source, like the reference */
    [class*="st-key-remove_source_"] button {
        background: transparent !important;
        border: none !important;
        color: var(--stamp) !important;
        padding: 0 !important;
        min-height: 0 !important;
    }

    /* Chat input lives inside the chat panel, under the messages. */
    [data-testid="stChatInput"] {
        border: 3px solid var(--ink) !important;
        border-radius: 0 !important;
        background: var(--paper) !important;
    }
    [data-testid="stChatInput"]:focus-within { border-color: var(--stamp) !important; }
    [data-testid="stChatInput"] div {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        background: transparent !important;
        border-radius: 0 !important;
    }
    [data-testid="stChatInputSubmitButton"] {
        background: var(--ink) !important;
        color: #FFFFFF !important;
        border-radius: 0 !important;
    }

    input, textarea {
        background-color: var(--card) !important;
        color: var(--ink) !important;
        outline: none !important;
        box-shadow: none !important;
    }
    input:focus, textarea:focus {
        outline: none !important;
        box-shadow: none !important;
        border-color: var(--stamp) !important;
    }
    [data-baseweb="base-input"]:focus-within {
        border-color: var(--stamp) !important;
        box-shadow: none !important;
    }

    .stProgress > div > div { background-color: var(--stamp); }

    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--card);
        border: 2px solid var(--ink);
        border-radius: 0;
        padding: 0.5rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 0.2rem;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: var(--ink);
        color: var(--paper);
        border: 2px solid var(--ink);
        border-radius: 0;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.75rem;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: var(--highlight);
        color: var(--ink);
        border-color: var(--ink);
    }
    [data-testid="stFileUploaderDropzone"] small {
        font-family: 'JetBrains Mono', monospace;
        color: var(--muted);
        font-size: 0.7rem;
    }

    .line-sidebar__item {
        position: relative;
        padding: 8px 0 8px 26px;
        cursor: default;
        border-bottom: 1px solid rgba(0,0,0,0.12);
    }
    .line-sidebar__marker {
        position: absolute;
        top: 50%;
        left: 0;
        height: 12px;
        width: 2px;
        background-color: var(--ink);
        transform: translateY(-50%);
    }
    .line-sidebar__label { color: var(--ink); font-size: 0.87rem; display: inline-block; }
    .line-sidebar__index {
        font-family: 'JetBrains Mono', monospace;
        margin-right: 8px;
        opacity: 0.6;
        font-size: 0.8em;
    }

    .confidence-stamp {
        display: inline-flex;
        align-items: baseline;
        gap: 8px;
        margin-top: 1rem;
        padding: 0.3rem 0.65rem;
        border: 2px solid var(--stamp);
        color: var(--stamp);
        transform: rotate(-1.5deg);
        animation: stamp-in 0.18s ease-out;
    }
    .confidence-stamp__label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .confidence-stamp__value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        font-weight: 700;
    }
    @keyframes stamp-in {
        0%   { transform: rotate(-1.5deg) scale(1.35); opacity: 0; }
        100% { transform: rotate(-1.5deg) scale(1); opacity: 1; }
    }
    @media (prefers-reduced-motion: reduce) { .confidence-stamp { animation: none; } }

    .error-card {
        color: var(--stamp);
        font-size: 0.92rem;
        padding: 0.4rem 0;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


@st.cache_resource
def load_reranker():
    return Reranker()


model = load_model()
client = load_client()
reranker = load_reranker()

telemetry.init_db()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()  # loads from disk if present
if "hybrid_search" not in st.session_state:
    st.session_state.hybrid_search = HybridSearch(st.session_state.vector_store)

CHAT_HISTORY_PATH = "chat_history.pkl"

if "messages" not in st.session_state:
    st.session_state.messages = []
    if os.path.exists(CHAT_HISTORY_PATH):
        try:
            with open(CHAT_HISTORY_PATH, "rb") as f:
                st.session_state.messages = pickle.load(f)
        except Exception:
            pass  # corrupt/missing history file — just start fresh

# Changing this key resets the file uploader. Without it, removing a source
# left the old files in the uploader and they were re-added on the next rerun.
st.session_state.setdefault("uploader_key", 0)

if "ingested_sources" not in st.session_state:
    # O(1) membership check for dedup, instead of re-scanning every chunk
    # on every rerun (which happens on any button click, not just uploads).
    st.session_state.ingested_sources = {c["source"] for c in st.session_state.vector_store.chunks}


def add_chunks(new_chunks):
    if not new_chunks:
        return
    texts = [c["text"] for c in new_chunks]
    batch_embeddings = model.encode(texts)
    st.session_state.vector_store.add(new_chunks, batch_embeddings)


def save_messages():
    with open(CHAT_HISTORY_PATH, "wb") as f:
        pickle.dump(st.session_state.messages, f)


def extract_youtube_id(url_or_id):
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url_or_id)
    return match.group(1) if match else url_or_id.strip()


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


def render_line_sidebar():
    sources = list(dict.fromkeys(c["source"] for c in st.session_state.vector_store.chunks))
    if not sources:
        return

    for i, src in enumerate(sources):
        display_name = src if len(src) < 30 else src[:27] + "..."
        col_label, col_remove = st.columns([0.85, 0.15])
        with col_label:
            st.markdown(
                f'<div class="line-sidebar__item">'
                f'<span class="line-sidebar__marker"></span>'
                f'<span class="line-sidebar__label">'
                f'<span class="line-sidebar__index">{str(i+1).zfill(2)}</span>{display_name}'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_remove:
            if st.button("×", key=f"remove_source_{i}", help=f"Remove {src}"):
                st.session_state.vector_store.remove_source(src)
                st.session_state.ingested_sources.discard(src)
                st.session_state.uploader_key += 1
                st.rerun()


def autoscroll():
    """Keep the chat box pinned to its newest content for a few seconds.
    A single scroll at load time isn't enough: the answer's text and
    stamp finish laying out after it, so we keep re-pinning briefly. The
    value in the comment changes every run so Streamlit re-executes it.
    Also measures where the frame starts so CSS can size it to the window."""
    components.html(
        """<script>/* __N__ */
        (function () {
            var P = window.parent, D = P.document;
            function go() {
                try {
                    var f = D.querySelector('[class*="st-key-body_row"]');
                    if (f) D.documentElement.style.setProperty('--top', Math.round(f.getBoundingClientRect().top) + 'px');
                    var sc = D.querySelector('.st-key-chat_scroll');
                    if (sc) sc.scrollTop = sc.scrollHeight;
                } catch (e) {}
            }
            go();
            var n = 0;
            var t = setInterval(function () { go(); if (++n >= 30) clearInterval(t); }, 100);
        })();
        </script>""".replace("__N__", str(time.time_ns())),
        height=0,
    )


st.session_state.setdefault("show_sources", True)
show_sources = st.session_state.show_sources


def render_sources_panel():
    with st.container(key="sources_col_inner"):
        with st.container(key="sources_fixed_top"):
            st.markdown("### Sources")

            uploaded_files = st.file_uploader(
                "Upload PDF or text file",
                type=["pdf", "txt"],
                key=f"uploader_{st.session_state.uploader_key}",
                accept_multiple_files=True,
                label_visibility="collapsed"
            )

            if uploaded_files:
                new_files = [f for f in uploaded_files if f.name not in st.session_state.ingested_sources]

                if new_files:
                    total_files = len(new_files)
                    progress_bar = st.progress(0, text="Starting...")

                    for file_idx, uploaded in enumerate(new_files):
                        suffix = os.path.splitext(uploaded.name)[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded.getvalue())
                            tmp_path = tmp.name

                        try:
                            if suffix == ".pdf":
                                def page_progress(current_page, total_pages, fname=uploaded.name, idx=file_idx):
                                    file_fraction = current_page / total_pages
                                    overall = (idx + file_fraction) / total_files
                                    progress_bar.progress(
                                        overall,
                                        text=f"Processing {fname} — page {current_page}/{total_pages}"
                                    )
                                new_chunks = ingest_pdf(tmp_path, progress_callback=page_progress)
                            else:
                                progress_bar.progress(
                                    file_idx / total_files,
                                    text=f"Processing {uploaded.name}..."
                                )
                                new_chunks = ingest_text_file(tmp_path)
                        except Exception as e:
                            st.error(f"Couldn't process {uploaded.name} — skipping it. ({e})")
                            print(f"[Vantage] Ingestion failed for {uploaded.name}: {e}")
                            continue

                        for c in new_chunks:
                            c["source"] = uploaded.name

                        add_chunks(new_chunks)
                        st.session_state.ingested_sources.add(uploaded.name)
                        progress_bar.progress((file_idx + 1) / total_files, text=f"Added {uploaded.name}")

                    progress_bar.progress(1.0, text="Done!")
                    st.success(f"Added {total_files} file(s)")
                    progress_bar.empty()

            yt_input = st.text_input("YouTube URL or ID", placeholder="Paste a link...")
            if st.button("Add video", use_container_width=True, type="primary") and yt_input:
                video_id = extract_youtube_id(yt_input)
                video_source = f"youtube:{video_id}"
                if video_source not in st.session_state.ingested_sources:
                    with st.spinner("Fetching transcript..."):
                        try:
                            new_chunks = ingest_youtube(video_id)
                            if not new_chunks:
                                st.warning("No captions found for that video.")
                            else:
                                add_chunks(new_chunks)
                                st.session_state.ingested_sources.add(video_source)
                                st.success(f"Video added — {len(new_chunks)} chunks")
                        except Exception as e:
                            st.error(f"Couldn't fetch that video's transcript. ({e})")
                            print(f"[Vantage] YouTube ingestion failed for {video_id}: {e}")

            st.caption(f"{len(st.session_state.vector_store)} chunks loaded")

        # Streamlit's own scrolling for a long source list; no CSS needed.
        with st.container(key="sources_scroll", height=140, border=False):
            render_line_sidebar()

        if len(st.session_state.vector_store) > 0:
            if st.button("Clear all sources", use_container_width=True):
                st.session_state.vector_store.clear()
                st.session_state.ingested_sources = set()
                st.session_state.uploader_key += 1
                st.rerun()


with st.container(key="header_row"):
    col_title, col_toggle, col_clear = st.columns([0.62, 0.19, 0.19])
    with col_title:
        st.markdown(
            '<div class="app-header">Vantage</div>'
            '<div class="app-subtitle">Ask questions across your PDFs, text files, and YouTube lectures — grounded, cited answers.</div>',
            unsafe_allow_html=True
        )
    with col_toggle:
        if st.button("Hide sources" if show_sources else "Show sources",
                     key="toggle_sources_btn", use_container_width=True):
            st.session_state.show_sources = not show_sources
            st.rerun()
    with col_clear:
        if st.button("Clear chat", key="clear_chat_btn", use_container_width=True):
            st.session_state.messages = []
            save_messages()
            st.session_state.pop("last_query_id", None)
            st.session_state.pop("feedback_given", None)
            st.rerun()

with st.container(key="body_row" if show_sources else "body_row_full"):
    if show_sources:
        col_sources, col_chat = st.columns([0.28, 0.72])
        with col_sources:
            render_sources_panel()
    else:
        col_chat = st.container()

    with col_chat, st.container(key="chat_col_inner"):
        # Only this box scrolls. The input sits right under it, inside the frame.
        with st.container(key="chat_scroll"):
            # Two-phase send: the input (below this box) only stores the question
            # and reruns; the work happens here on the next run so the
            # question shows immediately in the right place.
            pending_question = st.session_state.pop("pending_question", None)
            history_before = st.session_state.messages.copy()

            if pending_question:
                if len(st.session_state.vector_store) == 0:
                    st.warning("Add at least one source first.")
                    pending_question = None
                else:
                    st.session_state.messages.append({"role": "user", "content": pending_question})
                    st.session_state.pop("last_query_id", None)
                    st.session_state.pop("feedback_given", None)
                    save_messages()

            for i, msg in enumerate(st.session_state.messages):
                role_label = "You" if msg["role"] == "user" else "Vantage"
                st.markdown(
                    f'<div class="msg-index">{i + 1:02d} · {role_label}</div>',
                    unsafe_allow_html=True
                )
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("confidence") is not None:
                        st.markdown(
                            f'<div class="confidence-stamp">'
                            f'<span class="confidence-stamp__label">Confidence</span>'
                            f'<span class="confidence-stamp__value">{msg["confidence"]:.2f}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            if pending_question:
                autoscroll()  # follow the new question and spinner as they appear
                question = pending_question

                contextual_query = build_contextual_query(question, history_before)
                query_embedding = model.encode(contextual_query)

                retrieval_start = time.perf_counter()
                # Separate, cheap dense-only lookup purely for a true 0-1 cosine
                # similarity score to log — the hybrid/reranked results below use
                # RRF and cross-encoder scores, which aren't on a comparable scale.
                dense_top = st.session_state.vector_store.search(query_embedding, top_k=1)
                top_similarity_score = dense_top[0][0] if dense_top else 0.0

                candidates = st.session_state.hybrid_search.search(contextual_query, query_embedding, top_k=15)
                top_chunks = reranker.rerank(contextual_query, candidates, top_k=3)
                retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

                combined_context = "\n\n".join(
                    f"[Source: {c['source']} @ {c['location']}]\n{c['text']}"
                    for score, c in top_chunks
                )

                history_text = format_history(history_before)

                prompt = f"""You are answering questions using ONLY the context provided below.

Conversation so far:
{history_text if history_text else "(no earlier messages)"}

Context from sources:
{combined_context}

Current question: {question}

Instructions: If the current question is a follow-up (like "explain more", "why", "what about that", etc.), use the conversation above to understand what it's referring to, then answer using the context. Write a clear, complete, well-explained answer in plain prose — no source citations, no brackets, no meta-commentary about where the information came from. If the answer genuinely isn't in the context, say "I don't know based on the provided notes."

Answer:"""

                with st.chat_message("assistant"):
                    with st.spinner("Searching your sources..."):
                        # gemini-3.1-flash-lite: ~3x the rate limit of 3.6-flash
                        # and far cheaper, fine for grounded QA over retrieved
                        # context. Retry covers transient 503s on any tier.
                        GENERATION_MODEL = "gemini-3.1-flash-lite"
                        max_retries = 2
                        last_error = None
                        generation_start = time.perf_counter()
                        response = None

                        for attempt in range(max_retries + 1):
                            try:
                                response = client.models.generate_content(
                                    model=GENERATION_MODEL,
                                    contents=prompt
                                )
                                last_error = None
                                break
                            except Exception as e:
                                last_error = e
                                print(f"[Vantage] Generation attempt {attempt + 1} failed: {e}")
                                if attempt < max_retries:
                                    time.sleep(1.5 * (attempt + 1))

                        generation_latency_ms = (time.perf_counter() - generation_start) * 1000

                    if last_error is not None:
                        # No st.stop() here: it would also stop the rest of the
                        # page (including the input bar) from rendering.
                        st.markdown(
                            '<div class="error-card">'
                            "Couldn't reach the AI model just now — this is usually temporary "
                            "(the API may be under heavy load). Try asking again in a moment."
                            '</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        answer = response.text

                        # Save immediately, before telemetry, which could still throw.
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "confidence": top_similarity_score,
                        })
                        save_messages()

                        prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                        completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

                        try:
                            query_id = telemetry.log_query(
                                user_query=question,
                                retrieval_latency_ms=retrieval_latency_ms,
                                generation_latency_ms=generation_latency_ms,
                                top_similarity_score=top_similarity_score,
                                retrieved_chunks_count=len(top_chunks),
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                            )
                            st.session_state.last_query_id = query_id
                            st.session_state.feedback_given = False
                        except Exception as e:
                            # Telemetry is a nice-to-have, not worth crashing an
                            # otherwise-successful answer over.
                            print(f"[Vantage] Telemetry logging failed: {e}")

                        # Rerun so the new answer renders through the normal
                        # history loop above.
                        st.rerun()

            if st.session_state.get("last_query_id") and not st.session_state.get("feedback_given"):
                fb_col1, fb_col2, _ = st.columns([1, 1, 8])
                with fb_col1:
                    if st.button("👍", key=f"fb_up_{st.session_state.last_query_id}"):
                        telemetry.update_feedback(st.session_state.last_query_id, 1)
                        st.session_state.feedback_given = True
                        st.rerun()
                with fb_col2:
                    if st.button("👎", key=f"fb_down_{st.session_state.last_query_id}"):
                        telemetry.update_feedback(st.session_state.last_query_id, -1)
                        st.session_state.feedback_given = True
                        st.rerun()
            elif st.session_state.get("feedback_given"):
                st.caption("Thanks for the feedback!")

        question = st.chat_input("Ask a question...")
        if question:
            st.session_state.pending_question = question
            st.rerun()

autoscroll()