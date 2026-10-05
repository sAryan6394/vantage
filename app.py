import streamlit as st
import streamlit.components.v1 as components
import os
import json
import html
import pickle
import urllib.error
import urllib.parse
import urllib.request
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
import rag_pipeline
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
        padding-bottom: 1.2rem;
    }
    .st-key-header_row [data-testid="stHorizontalBlock"] { align-items: center; }

    /* The frame. Divider between the two columns is a background line, so it
       always spans the full frame height whatever the columns contain. */
    [class*="st-key-body_row"] {
        border: 3px solid var(--ink);
        /* frame runs from its top edge to 18px above the window bottom */
        height: calc(100vh - 138px);
        height: calc(100dvh - 138px);
    }
    .st-key-body_row {
        background: linear-gradient(var(--ink), var(--ink)) no-repeat 28% 0 / 3px 100%;
    }

    .st-key-sources_col_inner {
        padding: 1.1rem 1.4rem;
        height: calc(100vh - 144px);
        height: calc(100dvh - 144px);
        overflow-y: auto;
    }
    .st-key-clear_all_btn { margin-top: auto; }
    .st-key-chat_col_inner { padding: 1.1rem 1.4rem; }

    /* Chat box = frame height minus frame border (6), chat padding (35),
       gap (16) and the input bar (59). Pure CSS, so no timing issues. */
    .st-key-chat_scroll {
        height: calc(100vh - 252px);
        height: calc(100dvh - 252px);
        overflow-y: auto;
        flex: 0 0 auto;
        padding-right: 10px;
        scrollbar-width: thin;
        scrollbar-color: var(--ink) transparent;
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
        font-size: 0.9rem;
        margin-top: 0.35rem;
    }

    .empty-state {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--muted);
        padding: 0.4rem 0.1rem;
    }

    .msg-index {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0.7rem 0 0.25rem 0.1rem;
    }

    [data-testid="stChatMessage"] {
        background-color: var(--card);
        border: 2px solid var(--ink);
        border-radius: 0;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.9rem;
        width: 100%;
        max-width: 100%;
    }
    [data-testid="stChatMessage"] p { line-height: 1.6; }
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
    .stButton button:active { background-color: #222; color: var(--paper); }
    /* Add video = same yellow as Upload */
    .stButton button[kind="primary"] {
        background-color: var(--highlight);
        color: var(--ink);
        border: 2px solid var(--ink);
    }
    .stButton button[kind="primary"]:hover,
    .stButton button[kind="primary"]:active {
        background-color: #E6C72F;
        border-color: var(--ink);
        color: var(--ink);
    }
    /* Destructive buttons: outlined red */
    .st-key-clear_chat_btn button,
    .st-key-clear_all_btn button {
        background-color: transparent;
        color: var(--stamp);
        border: 2px solid var(--stamp);
    }
    .st-key-clear_chat_btn button:hover,
    .st-key-clear_chat_btn button:active,
    .st-key-clear_all_btn button:hover,
    .st-key-clear_all_btn button:active {
        background-color: rgba(176, 24, 47, 0.1);
        color: var(--stamp);
        border-color: var(--stamp);
    }
    /* Feedback: small text buttons; "Not helpful" is the outlined one */
    [class*="st-key-fb_up_"] button,
    [class*="st-key-fb_down_"] button {
        min-height: 2.25rem;
        font-size: 0.75rem;
    }
    [class*="st-key-fb_down_"] button {
        background-color: transparent;
        color: var(--ink);
    }
    [class*="st-key-fb_down_"] button:hover,
    [class*="st-key-fb_down_"] button:active {
        background-color: rgba(0, 0, 0, 0.08);
        color: var(--ink);
    }
    /* Visible keyboard focus everywhere */
    .stButton button:focus-visible,
    [data-testid="stFileUploaderDropzone"] button:focus-visible {
        outline: 3px solid var(--stamp) !important;
        outline-offset: 2px;
    }
    /* small red x next to each source, like the reference */
    [class*="st-key-remove_source_"] button {
        background: transparent !important;
        border: none !important;
        color: var(--stamp) !important;
        padding: 0 !important;
        width: 32px;
        min-width: 32px;
        height: 32px;
        min-height: 32px !important;
        font-size: 1.1rem;
    }
    .st-key-sources_scroll [data-testid="stHorizontalBlock"] { align-items: center; }

    /* Chat input lives inside the chat panel, under the messages. */
    [data-testid="stChatInput"] {
        border: 3px solid var(--ink) !important;
        border-radius: 0 !important;
        background: var(--card) !important;
    }
    [data-testid="stChatInput"] textarea { background: transparent !important; }
    [data-testid="stChatInput"]:focus-within {
        outline: 3px solid var(--stamp);
        outline-offset: 2px;
    }
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
        outline: 3px solid var(--stamp);
        outline-offset: 2px;
    }
    ::placeholder { color: #6B6E66 !important; opacity: 1 !important; }
    [data-testid="stSpinner"] { font-family: 'JetBrains Mono', monospace; color: var(--stamp); }

    .stProgress > div > div { background-color: var(--stamp); }

    /* Upload = one plain yellow rectangle button (no dashed drop box, no text). */
    [data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none;
        padding: 0;
        min-height: 0;
        display: flex;
        flex-direction: column;
        align-items: stretch;
    }
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzone"] small {
        display: none;
    }
    [data-testid="stFileUploaderDropzone"] button {
        width: 100%;
        min-height: 2.5rem;
        background-color: var(--highlight);
        color: var(--ink);
        border: 2px solid var(--ink);
        border-radius: 0;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #E6C72F;
        color: var(--ink);
        border-color: var(--ink);
    }

    .line-sidebar__item {
        position: relative;
        padding: 8px 0 8px 26px;
        cursor: default;
        overflow: hidden;
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
    .line-sidebar__label {
        color: var(--ink);
        font-size: 0.87rem;
        display: block;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
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
        font-size: 0.75rem;
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

    @media (max-width: 900px) {
        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stMain"], section.main { overflow: auto !important; }
        [class*="st-key-body_row"] { height: auto; }
        .st-key-body_row { background: none; }
        .st-key-sources_col_inner { height: auto; border-bottom: 3px solid var(--ink); }
        .st-key-chat_scroll { height: 340px; }
    }

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


TITLES_PATH = "source_titles.json"


def fetch_youtube_title(video_id):
    """Returns (title, network_down). Uses YouTube's public oEmbed endpoint,
    which needs no API key. A private/removed video just has no title
    (HTTP error); anything else means we're offline, so callers stop trying."""
    url = ("https://www.youtube.com/oembed?format=json&url="
           + urllib.parse.quote(f"https://www.youtube.com/watch?v={video_id}", safe=""))
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return json.load(r).get("title"), False
    except urllib.error.HTTPError:
        return None, False
    except Exception:
        return None, True


def source_titles(sources):
    """source -> real title, fetched once per video and remembered in
    source_titles.json. Falls back to the video id if a title can't be fetched."""
    if "titles" not in st.session_state:
        try:
            with open(TITLES_PATH, encoding="utf-8") as f:
                st.session_state.titles = json.load(f)
        except Exception:
            st.session_state.titles = {}
        st.session_state.title_failed = set()
    titles, failed = st.session_state.titles, st.session_state.title_failed
    changed = False
    for src in sources:
        if src.startswith("youtube:") and src not in titles and src not in failed:
            title, offline = fetch_youtube_title(src.split(":", 1)[1])
            if title:
                titles[src] = title
                changed = True
            else:
                failed.add(src)
            if offline:
                failed.update(x for x in sources if x.startswith("youtube:"))
                break
    if changed:
        try:
            with open(TITLES_PATH, "w", encoding="utf-8") as f:
                json.dump(titles, f, ensure_ascii=False)
        except Exception:
            pass  # a cache that can't be saved just means a re-fetch next time
    return titles


def render_line_sidebar():
    sources = list(dict.fromkeys(c["source"] for c in st.session_state.vector_store.chunks))
    if not sources:
        return

    titles = source_titles(sources)
    for i, src in enumerate(sources):
        shown = html.escape(titles.get(src, src))
        col_label, col_remove = st.columns([0.85, 0.15])
        with col_label:
            st.markdown(
                f'<div class="line-sidebar__item" title="{shown}">'
                f'<span class="line-sidebar__marker"></span>'
                f'<span class="line-sidebar__label">'
                f'<span class="line-sidebar__index">{str(i+1).zfill(2)}</span>{shown}'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_remove:
            if st.button("×", key=f"remove_source_{i}", help=f"Remove {titles.get(src, src)}"):
                st.session_state.vector_store.remove_source(src)
                st.session_state.ingested_sources.discard(src)
                st.session_state.uploader_key += 1
                st.rerun()


def autoscroll():
    """Pin the chat box to its newest content while an answer is still laying
    out, then stop (and stop at once if the user scrolls). The value in the
    comment changes every run so Streamlit re-executes the script."""
    components.html(
        """<script>/* __N__ */
        (function () {
            var D = window.parent.document;
            var sc = D.querySelector('.st-key-chat_scroll');
            if (!sc) return;
            var stop = false, last = -1, same = 0, n = 0;
            // The moment you scroll yourself, stop auto-scrolling (it used to fight you).
            ['wheel', 'touchstart', 'mousedown', 'keydown'].forEach(function (ev) {
                sc.addEventListener(ev, function () { stop = true; }, { passive: true, once: true });
            });
            var t = setInterval(function () {
                if (stop || ++n > 40) { clearInterval(t); return; }
                sc.scrollTop = sc.scrollHeight;
                // content stopped growing -> we're done
                if (sc.scrollHeight === last) { if (++same >= 4) clearInterval(t); }
                else { same = 0; last = sc.scrollHeight; }
            }, 100);
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

            yt_input = st.text_input("YouTube URL", placeholder="Paste a link...")
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
        # Streamlit scrolls this box itself, so the list can never stretch the panel.
        with st.container(key="sources_scroll", height=260, border=False):
            render_line_sidebar()

        if len(st.session_state.vector_store) > 0:
            if st.button("Clear all sources", key="clear_all_btn", use_container_width=True):
                st.session_state.vector_store.clear()
                st.session_state.ingested_sources = set()
                st.session_state.uploader_key += 1
                st.rerun()


with st.container(key="header_row"):
    col_title, col_toggle, col_clear = st.columns([0.7, 0.15, 0.15])
    with col_title:
        st.markdown(
            '<div class="app-header">Vantage</div>'
            '<div class="app-subtitle">Ask questions across your sources.</div>',
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

            if not st.session_state.messages and not pending_question:
                st.markdown(
                    '<div class="empty-state">Ask any question below.</div>',
                    unsafe_allow_html=True
                )

            for i, msg in enumerate(st.session_state.messages):
                role_label = "You" if msg["role"] == "user" else "Vantage"
                st.markdown(
                    f'<div class="msg-index">{i + 1:02d} · {role_label}</div>',
                    unsafe_allow_html=True
                )
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    # Only show the match score on real answers, not on "I don't know".
                    if (msg["role"] == "assistant" and msg.get("confidence") is not None
                            and not msg["content"].strip().lower().startswith("i don't know")):
                        st.markdown(
                            f'<div class="confidence-stamp">'
                            f'<span class="confidence-stamp__label">Source match</span>'
                            f'<span class="confidence-stamp__value">{msg["confidence"]:.2f}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            if pending_question:
                autoscroll()  # follow the new question and spinner as they appear
                question = pending_question

                with st.chat_message("assistant"):
                    with st.spinner("Searching your sources..."):
                        result = rag_pipeline.answer_question(
                            question, history_before,
                            model=model, client=client,
                            vector_store=st.session_state.vector_store,
                            hybrid_search=st.session_state.hybrid_search,
                            reranker=reranker,
                        )

                    if result["error"]:
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
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "confidence": result["top_similarity_score"],
                        })
                        save_messages()
                        if result["query_id"]:
                            st.session_state.last_query_id = result["query_id"]
                            st.session_state.feedback_given = False

                        # Rerun so the new answer renders through the normal
                        # history loop above.
                        st.rerun()

            if st.session_state.get("last_query_id") and not st.session_state.get("feedback_given"):
                fb_col1, fb_col2, _ = st.columns([0.13, 0.17, 0.70])
                with fb_col1:
                    if st.button("Helpful", key=f"fb_up_{st.session_state.last_query_id}", use_container_width=True):
                        telemetry.update_feedback(st.session_state.last_query_id, 1)
                        st.session_state.feedback_given = True
                        st.rerun()
                with fb_col2:
                    if st.button("Not helpful", key=f"fb_down_{st.session_state.last_query_id}", use_container_width=True):
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
