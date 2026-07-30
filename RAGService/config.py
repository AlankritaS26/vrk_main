from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR        = Path(__file__).parent
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
UPLOADS_DIR     = BASE_DIR / "data" / "uploads"

for _d in (VECTORSTORE_DIR, UPLOADS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Embedding provider: "local" | "azure_openai" ──────────────────────
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
BGE_MODEL_NAME     = "BAAI/bge-large-en-v1.5"
BGE_DEVICE         = os.getenv("BGE_DEVICE", "cpu")

# Azure OpenAI (only needed when EMBEDDING_PROVIDER=azure_openai)
AZURE_OPENAI_ENDPOINT         = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY          = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-ada-002")
AZURE_OPENAI_API_VERSION      = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# ── Chunking ───────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
RAG_TOP_K     = int(os.getenv("RAG_TOP_K", "10"))

# ── API ────────────────────────────────────────────────────────────────
API_HOST    = os.getenv("API_HOST", "0.0.0.0")
API_PORT    = int(os.getenv("API_PORT", "8600"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))
