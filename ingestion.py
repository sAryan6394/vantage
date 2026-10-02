import os
import io
import re
import platform
import pymupdf as fitz
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# --- Tesseract path: portable across Windows / Linux / Docker ---
# Set TESSERACT_CMD in your .env if it's not on PATH.
_tesseract_cmd = os.getenv("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
elif platform.system() == "Windows":
    _default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_default_win_path):
        pytesseract.pytesseract.tesseract_cmd = _default_win_path
# On Linux/Docker, pytesseract finds `tesseract` on PATH automatically
# as long as the tesseract-ocr package is installed in the image.


def format_timestamp(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


# --- Sentence-aware recursive chunking with overlap ---

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_into_sentences(text):
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s for s in sentences if s.strip()]


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Sentence-aware chunking with overlap, replacing the old fixed-width
    character slicing (which used to cut sentences in half).

    - Packs whole sentences into chunks up to `chunk_size` characters.
    - A single sentence longer than `chunk_size` is hard-split as a fallback
      (recursive character split), so nothing is ever silently dropped.
    - Each chunk after the first carries the trailing `overlap` characters
      of the previous chunk, so context/references aren't lost right at a
      chunk boundary.

    Returns a list of (chunk_text, start_offset) tuples. `start_offset` is
    the character position of the chunk's own (non-overlap) content in the
    original text — used for the "position N" location tag.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    # Find each sentence's start offset in the original text.
    sentence_spans = []
    cursor = 0
    for s in sentences:
        idx = text.find(s, cursor)
        if idx == -1:
            idx = cursor  # shouldn't happen, but don't crash on it
        sentence_spans.append((s, idx))
        cursor = idx + len(s)

    raw_chunks = []  # (text, start_offset)
    current_text = ""
    current_start = None

    for s, start in sentence_spans:
        if len(s) > chunk_size:
            # Flush whatever we've built up, then hard-split the oversized sentence.
            if current_text.strip():
                raw_chunks.append((current_text, current_start))
                current_text, current_start = "", None
            for i in range(0, len(s), chunk_size):
                raw_chunks.append((s[i:i + chunk_size], start + i))
            continue

        if not current_text:
            current_text = s
            current_start = start
        elif len(current_text) + 1 + len(s) <= chunk_size:
            current_text += " " + s
        else:
            raw_chunks.append((current_text, current_start))
            current_text = s
            current_start = start

    if current_text.strip():
        raw_chunks.append((current_text, current_start))

    # Apply overlap: prepend the tail of the previous chunk's own text.
    final_chunks = []
    for i, (chunk, start) in enumerate(raw_chunks):
        if i == 0 or overlap <= 0:
            final_chunks.append((chunk, start))
        else:
            prefix = raw_chunks[i - 1][0][-overlap:]
            final_chunks.append((f"{prefix} {chunk}", start))

    return final_chunks


def ingest_text_file(filepath, chunk_size=500, overlap=50):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = []
    for piece, offset in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
        chunks.append({
            "text": piece,
            "source": filepath,
            "source_type": "text",
            "location": f"position {offset}"
        })
    return chunks


def ingest_youtube(video_id, chunk_seconds=60, overlap_chars=50):
    """
    Auto-generated YouTube captions are typically unpunctuated running text,
    so the sentence-aware splitter (chunk_text) doesn't apply here — there
    are no sentence boundaries to find. Chunking is by time window instead,
    with the same trailing-text overlap idea as chunk_text for consistency
    across chunk boundaries.
    """
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)

    if not transcript.snippets:
        return []

    raw_chunks = []  # (text, start_time)
    current_text = ""
    current_start = transcript.snippets[0].start

    for snippet in transcript.snippets:
        if snippet.start - current_start > chunk_seconds and current_text:
            raw_chunks.append((current_text.strip(), current_start))
            current_text = ""
            current_start = snippet.start
        current_text += " " + snippet.text

    if current_text:
        raw_chunks.append((current_text.strip(), current_start))

    chunks = []
    for i, (text, start) in enumerate(raw_chunks):
        if i > 0 and overlap_chars > 0:
            text = f"{raw_chunks[i - 1][0][-overlap_chars:]} {text}"
        chunks.append({
            "text": text,
            "source": f"youtube:{video_id}",
            "source_type": "youtube",
            "location": format_timestamp(start)
        })
    return chunks


def ingest_pdf(filepath, chunk_size=500, overlap=50, progress_callback=None):
    doc = fitz.open(filepath)
    chunks = []
    total_pages = len(doc)

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text()

        if not text.strip():
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)

        for piece, offset in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append({
                "text": piece,
                "source": filepath,
                "source_type": "pdf",
                "location": f"page {page_num + 1}"
            })

        if progress_callback:
            progress_callback(page_num + 1, total_pages)

    return chunks