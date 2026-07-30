"""
RAGService — Standalone RAG Microservice
=========================================
REST API for indexing any document and doing semantic search.

Auto-generated interactive docs:  http://localhost:8600/docs
Alternative docs (ReDoc):         http://localhost:8600/redoc
OpenAPI schema (JSON):            http://localhost:8600/openapi.json

Collections are independent namespaces — one per project/app/use-case.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config
from file_parser import parse_file, extract_full_text, SUPPORTED_EXTENSIONS
from rag_store import RAGStore, RAGCollection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ragservice")

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAGService",
    description=__doc__,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton store (shared across requests)
_store: RAGStore | None = None

def get_store() -> RAGStore:
    global _store
    if _store is None:
        _store = RAGStore()
    return _store


# ── Request / Response schemas ─────────────────────────────────────────

class IndexTextRequest(BaseModel):
    text: str                           = Field(..., description="Raw text to index")
    source: str                         = Field("",  description="Source label (optional)")
    metadata: dict[str, Any]            = Field(default_factory=dict, description="Extra metadata to attach to all chunks")

class SearchRequest(BaseModel):
    query:   str                        = Field(..., description="Natural-language search query")
    k:       int                        = Field(10,  ge=1, le=100, description="Number of results to return")
    filters: dict[str, Any]             = Field(default_factory=dict, description="Metadata filters {key: value}")

class SearchResult(BaseModel):
    text:     str
    score:    float
    metadata: dict[str, Any]

class IndexResponse(BaseModel):
    collection: str
    added:      int
    message:    str

class CollectionStats(BaseModel):
    collection:    str
    internal_name: str
    provider:      str
    dimensions:    int
    total_chunks:  int
    files:         list[str]


# ── Utility ────────────────────────────────────────────────────────────

def _get_collection(name: str) -> RAGCollection:
    return get_store().collection(name)


# ══════════════════════════════════════════════════════════════════════
#  SYSTEM ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health():
    """Liveness check — returns OK if the service is running."""
    return {"status": "ok", "version": "1.0.0", "embedding_provider": config.EMBEDDING_PROVIDER}


@app.get("/supported-types", tags=["System"])
def supported_types():
    """List of file extensions this service can parse."""
    return {"supported": sorted(SUPPORTED_EXTENSIONS)}


# ══════════════════════════════════════════════════════════════════════
#  COLLECTION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

@app.get("/v1/collections", tags=["Collections"])
def list_collections():
    """List all existing collections (namespaces)."""
    return {"collections": get_store().list_collections()}


@app.get("/v1/collections/{collection}", response_model=CollectionStats, tags=["Collections"])
def get_collection_stats(collection: str):
    """Get stats for a collection: chunk count, indexed files, embedding info."""
    return _get_collection(collection).stats()


@app.delete("/v1/collections/{collection}", tags=["Collections"])
def delete_collection(collection: str):
    """Delete an entire collection and all its indexed data. This is irreversible."""
    get_store().delete_collection(collection)
    return {"message": f"Collection '{collection}' deleted."}


# ══════════════════════════════════════════════════════════════════════
#  INDEXING
# ══════════════════════════════════════════════════════════════════════

@app.post("/v1/collections/{collection}/index/file",
          response_model=IndexResponse, tags=["Indexing"])
async def index_file(
    collection: str,
    file:   UploadFile = File(..., description="File to index (.pptx .docx .pdf .xlsx .csv .txt .md)"),
    source: str        = Form("",  description="Source label, e.g. department or project name"),
):
    """
    Upload and index a file into a collection.

    The file is parsed into text chunks, embedded locally, and stored in ChromaDB.
    Re-uploading the same file is safe — duplicate chunks are skipped automatically.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    content = await file.read()
    try:
        chunks = parse_file(content, file.filename, source=source)
    except Exception as e:
        raise HTTPException(422, f"Failed to parse '{file.filename}': {e}")

    coll  = _get_collection(collection)
    added = coll.upsert(chunks)
    log.info("Indexed %s → collection=%s chunks=%d new=%d", file.filename, collection, len(chunks), added)
    return IndexResponse(
        collection=collection,
        added=added,
        message=f"Indexed '{file.filename}': {added} new chunks added ({len(chunks) - added} already existed).",
    )


@app.post("/v1/collections/{collection}/index/files",
          response_model=list[IndexResponse], tags=["Indexing"])
async def index_multiple_files(
    collection: str,
    files:  list[UploadFile] = File(..., description="Multiple files to index"),
    source: str              = Form("",  description="Source label applied to all files"),
):
    """Upload and index multiple files at once."""
    results = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append(IndexResponse(collection=collection, added=0,
                                          message=f"SKIPPED '{f.filename}': unsupported type."))
            continue
        content = await f.read()
        try:
            chunks = parse_file(content, f.filename, source=source)
            added  = _get_collection(collection).upsert(chunks)
            results.append(IndexResponse(collection=collection, added=added,
                                          message=f"'{f.filename}': {added} new chunks."))
        except Exception as e:
            results.append(IndexResponse(collection=collection, added=0,
                                          message=f"ERROR '{f.filename}': {e}"))
    return results


@app.post("/v1/collections/{collection}/index/text",
          response_model=IndexResponse, tags=["Indexing"])
def index_text(collection: str, body: IndexTextRequest):
    """
    Index a raw text string directly (no file upload needed).

    Useful for programmatic indexing from pipelines, APIs, or databases.
    """
    coll  = _get_collection(collection)
    added = coll.upsert_text(body.text, metadata={"source": body.source, **body.metadata})
    return IndexResponse(
        collection=collection,
        added=added,
        message=f"Text indexed: {added} new chunks.",
    )


# ══════════════════════════════════════════════════════════════════════
#  FILE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

@app.get("/v1/collections/{collection}/files", tags=["File Management"])
def list_files(collection: str):
    """List all filenames currently indexed in a collection."""
    return {"collection": collection, "files": _get_collection(collection).list_files()}


@app.delete("/v1/collections/{collection}/files/{filename}", tags=["File Management"])
def delete_file(collection: str, filename: str):
    """
    Remove all chunks from a specific file.

    Use this before re-uploading an updated version of the same file.
    """
    _get_collection(collection).delete_by_filename(filename)
    return {"message": f"All chunks for '{filename}' removed from '{collection}'."}


# ══════════════════════════════════════════════════════════════════════
#  SEARCH
# ══════════════════════════════════════════════════════════════════════

@app.post("/v1/collections/{collection}/search",
          response_model=list[SearchResult], tags=["Search"])
def search(collection: str, body: SearchRequest):
    """
    Semantic similarity search within a collection.

    Returns the top-k most relevant chunks with their text, metadata, and similarity score (0–1).
    Optionally filter by metadata fields using the `filters` dict.

    **Example filters:**
    ```json
    {"source": "Finance", "filename": "Q3_report.pptx"}
    ```
    """
    results = _get_collection(collection).search(
        query=body.query, k=body.k,
        filters=body.filters or None,
    )
    return [SearchResult(**r) for r in results]


@app.get("/v1/collections/{collection}/search", tags=["Search"])
def search_get(
    collection: str,
    q:  str = Query(..., description="Search query"),
    k:  int = Query(10,  ge=1, le=100, description="Number of results"),
):
    """
    Quick GET-based semantic search (no request body needed).

    Useful for testing directly from the browser or curl.

    ```
    GET /v1/collections/my-project/search?q=budget+risks&k=5
    ```
    """
    results = _get_collection(collection).search(query=q, k=k)
    return [SearchResult(**r) for r in results]


# ══════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=False,
    )
