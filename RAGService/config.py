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

# Device-aware model selection — same pattern as the kiosk's STT (small.en
# on CPU, a turbo model when GPU-connected). The large model is NOT removed:
# it's still used automatically whenever a GPU is available, so quality is
# preserved wherever the hardware supports it; CPU gets a lighter model
# purely for latency, appropriate for this project's focused ~90-chunk
# campus knowledge base (not a huge diverse corpus where the accuracy gap
# would matter more).
BGE_DEVICE     = os.getenv("BGE_DEVICE", "cpu")
BGE_MODEL_CPU  = os.getenv("BGE_MODEL_CPU", "BAAI/bge-base-en-v1.5")
BGE_MODEL_GPU  = os.getenv("BGE_MODEL_GPU", "BAAI/bge-large-en-v1.5")
# Explicit override always wins if set; otherwise choose by device.
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "").strip() or (
    BGE_MODEL_GPU if BGE_DEVICE.strip().lower() in ("cuda", "gpu") else BGE_MODEL_CPU
)

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