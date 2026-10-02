# Vantage

A multi-source RAG (Retrieval-Augmented Generation) knowledge base that lets you
ask questions across your PDFs and YouTube lecture transcripts — with answers
grounded in your actual sources, not hallucinated.

## Features
- Multi-source ingestion: PDFs (including scanned/OCR), YouTube transcripts, plain text
- Sentence-aware recursive chunking with overlap for PDFs and text files — preserves context across chunk boundaries instead of cutting sentences mid-way. YouTube transcripts use time-windowed chunking with the same overlap idea instead, since auto-generated captions are unpunctuated running text with no sentence boundaries to split on
- Hybrid retrieval: FAISS dense vector search (meaning-based) + BM25 sparse keyword search, fused via Reciprocal Rank Fusion — catches both semantic matches and exact terms (acronyms, IDs, proper nouns) that embeddings alone tend to miss
- Grounded AI answers using Google's Gemini API
- Conversational follow-ups — remembers recent context for questions like "explain more"
- Persistent storage — sources stay loaded across sessions (FAISS index + metadata on disk)
- Clean, custom-themed chat interface built with Streamlit

## How it works
1. Documents are split into sentence-aware, overlapping chunks and tagged with their source and location (page number, video timestamp, etc.)
2. Each chunk is embedded (all-MiniLM-L6-v2, 384-dim) and indexed in a FAISS HNSW index for fast approximate nearest-neighbor search; the same chunks are also indexed for BM25 keyword search
3. A question is run through both retrieval paths in parallel — dense (meaning) and sparse (keyword) — and the two ranked result lists are merged via Reciprocal Rank Fusion
4. The top fused matches are passed to an AI model, instructed to answer only from that retrieved context

## Stack
Python, Streamlit, sentence-transformers, FAISS, rank-bm25, numpy, Google Gemini API, PyMuPDF, pytesseract, youtube-transcript-api

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Requires a `.env` file with `GOOGLE_API_KEY=your_key_here`, and Tesseract OCR installed for scanned PDF support.