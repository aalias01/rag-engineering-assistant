"""
src/ingestion.py — Document ingestion pipeline for RAG Engineering Assistant.

Pipeline:
    PDF files in data/documents/
        → PyMuPDF text extraction (page-level, with metadata)
        → RecursiveCharacterTextSplitter (chunk_size=300, overlap=30)
        → OpenAI text-embedding-3-small (or local all-MiniLM-L6-v2)
        → ChromaDB persistent vector store

Usage:
    python -m src.ingestion                        # ingest all PDFs
    python -m src.ingestion --doc ASHRAE_90_1.pdf  # ingest single file
    python -m src.ingestion --reset                # wipe store + re-ingest all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config — all tunables in one place
# ---------------------------------------------------------------------------

DOCUMENTS_DIR = Path("data/documents")
CHROMA_PERSIST_PATH = Path(os.getenv("CHROMA_PERSIST_PATH", "./chroma_db"))
CHUNK_SIZE = 300          # tokens (approximated as chars / 4) — chosen by ablation:
                          # Recall@3 0.885 (300) vs 0.846 (500) vs 0.808 (800) on hybrid retrieval
CHUNK_OVERLAP = 30        # tokens (~10% of chunk size)
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "engineering_docs"
HASH_STORE_PATH = Path("data/eval/ingested_hashes.json")


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: Path) -> list[dict]:
    """
    Extract text from all pages of a PDF using PyMuPDF.

    Returns a list of dicts:
        {"text": str, "page": int, "source": str, "source_path": str}

    Tries pdfplumber as fallback if PyMuPDF yields empty text on a page
    (common for scanned PDFs — note: scanned docs require OCR, out of scope).
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            # Fallback to pdfplumber for this page
            try:
                import pdfplumber
                with pdfplumber.open(str(pdf_path)) as pdf:
                    pl_page = pdf.pages[page_num - 1]
                    text = (pl_page.extract_text() or "").strip()
            except Exception:
                pass
        if text:
            pages.append({
                "text": text,
                "page": page_num,
                "source": pdf_path.name,
                "source_path": str(pdf_path),
            })
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split page-level text into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter.

    Each chunk inherits the page metadata. When a chunk spans a page
    boundary, it carries the metadata of the starting page.

    chunk_size and overlap are in approximate token units (chars / 4).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Convert token estimates to characters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,
        chunk_overlap=overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        sub_chunks = splitter.split_text(page["text"])
        for i, chunk_text in enumerate(sub_chunks):
            chunks.append({
                "text": chunk_text,
                "page": page["page"],
                "source": page["source"],
                "chunk_index": i,
                # Unique ID: doc_name + page + chunk_index
                "chunk_id": f"{page['source']}__p{page['page']}__c{i}",
            })
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_embedding_function():
    """
    Return the embedding function for the configured provider.

    EMBEDDING_PROVIDER env var (default: "openai"):
      - "openai"  → text-embedding-3-small via OpenAI API
      - "local"   → all-MiniLM-L6-v2 via sentence-transformers (free, offline)
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    if provider == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        from langchain_openai import OpenAIEmbeddings
        # check_embedding_ctx_length=False skips tiktoken's client-side token
        # count (which requires downloading an encoding file at runtime).
        # Chunks are ~500 tokens — far below the 8191-token embedding limit —
        # so the check is unnecessary and the pipeline works offline-restricted.
        return OpenAIEmbeddings(model=EMBEDDING_MODEL, check_embedding_ctx_length=False)


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

def get_chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_PATH))


def get_or_create_collection(client, reset: bool = False):
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"Using existing collection '{COLLECTION_NAME}' ({collection.count()} chunks)")
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Created new collection '{COLLECTION_NAME}'")
    return collection


def add_chunks_to_chroma(collection, chunks: list[dict], embedding_fn) -> None:
    """
    Embed and upsert chunks into ChromaDB. Uses upsert (not add) so
    re-ingesting a file after editing is idempotent.
    """
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {"source": c["source"], "page": c["page"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]

    # Embed in batches of 100 to stay within OpenAI rate limits
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        embeddings = embedding_fn.embed_documents(batch_texts)
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        print(f"  Upserted chunks {i + 1}–{min(i + batch_size, len(texts))}/{len(texts)}")


# ---------------------------------------------------------------------------
# Hash tracking — skip already-ingested files
# ---------------------------------------------------------------------------

def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hash_store() -> dict[str, str]:
    if HASH_STORE_PATH.exists():
        return json.loads(HASH_STORE_PATH.read_text())
    return {}


def save_hash_store(store: dict[str, str]) -> None:
    HASH_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HASH_STORE_PATH.write_text(json.dumps(store, indent=2))


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest(
    doc_filter: Optional[str] = None,
    reset: bool = False,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> dict:
    """
    Ingest all PDFs in DOCUMENTS_DIR into ChromaDB.

    Args:
        doc_filter: if set, only process the file with this name
        reset: if True, wipe the ChromaDB collection first
        chunk_size: chunk size in approximate tokens
        overlap: overlap in approximate tokens

    Returns:
        summary dict with counts of files processed and chunks added
    """
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DOCUMENTS_DIR}. Add engineering documents and re-run.")
        return {"files_processed": 0, "chunks_added": 0}

    if doc_filter:
        pdf_files = [f for f in pdf_files if f.name == doc_filter]
        if not pdf_files:
            raise FileNotFoundError(f"No PDF named '{doc_filter}' in {DOCUMENTS_DIR}")

    embedding_fn = get_embedding_function()
    client = get_chroma_client()
    collection = get_or_create_collection(client, reset=reset)
    hash_store = {} if reset else load_hash_store()

    total_chunks = 0
    files_processed = 0

    for pdf_path in pdf_files:
        current_hash = file_md5(pdf_path)
        if not reset and hash_store.get(pdf_path.name) == current_hash:
            print(f"Skipping {pdf_path.name} (unchanged)")
            continue

        print(f"\nIngesting {pdf_path.name} ...")
        pages = extract_pages(pdf_path)
        if not pages:
            print(f"  WARNING: No text extracted from {pdf_path.name}. Skipping.")
            continue

        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        print(f"  {len(pages)} pages → {len(chunks)} chunks")
        add_chunks_to_chroma(collection, chunks, embedding_fn)

        hash_store[pdf_path.name] = current_hash
        total_chunks += len(chunks)
        files_processed += 1

    save_hash_store(hash_store)

    print(f"\nIngestion complete: {files_processed} file(s), {total_chunks} new chunks")
    print(f"ChromaDB collection size: {collection.count()}")
    return {"files_processed": files_processed, "chunks_added": total_chunks}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest engineering PDFs into ChromaDB")
    parser.add_argument("--doc", type=str, default=None, help="Ingest a single named PDF file")
    parser.add_argument("--reset", action="store_true", help="Wipe ChromaDB collection and re-ingest all")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size in approximate tokens")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="Chunk overlap in approximate tokens")
    args = parser.parse_args()

    ingest(doc_filter=args.doc, reset=args.reset, chunk_size=args.chunk_size, overlap=args.overlap)
