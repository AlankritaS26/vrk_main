from __future__ import annotations
import hashlib
from typing import List
import chromadb
from chromadb.config import Settings
from embeddings import get_embedding_provider, EmbeddingProvider
import config


class RAGCollection:
    """
    One named collection in the vector store.
    Each collection is independent — different apps/projects use different collections.
    """

    def __init__(self, collection_name: str):
        self._name     = collection_name
        self._provider: EmbeddingProvider = get_embedding_provider()
        self._client   = chromadb.PersistentClient(
            path=str(config.VECTORSTORE_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        # Collection name encodes provider+dims to prevent dimension collisions
        self._coll_name = f"{collection_name}__{self._provider.collection_suffix()}"
        self._coll = self._client.get_or_create_collection(
            name=self._coll_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ──────────────────────────────────────────────────────────

    def upsert(self, documents: List[dict]) -> int:
        """Index chunks. Deduplicates by content hash. Returns count of new chunks added."""
        if not documents:
            return 0
        texts     = [d["text"] for d in documents]
        metadatas = [d["metadata"] for d in documents]
        ids       = [_chunk_id(d["metadata"].get("filename", ""), d["metadata"].get("chunk_idx", 0), d["text"]) for d in documents]

        existing  = set(self._coll.get(ids=ids)["ids"])
        new_items = [(t, m, i) for t, m, i in zip(texts, metadatas, ids) if i not in existing]
        if not new_items:
            return 0

        nt, nm, ni = zip(*new_items)
        embeddings = self._provider.embed_documents(list(nt))
        self._coll.add(embeddings=embeddings, documents=list(nt), metadatas=list(nm), ids=list(ni))
        return len(ni)

    def upsert_text(self, text: str, metadata: dict | None = None) -> int:
        """Index a raw text string directly."""
        from file_parser import _chunk
        chunks = _chunk([{"text": text, "loc": {}}], metadata or {"filename": "raw_text"})
        return self.upsert(chunks)

    def delete_by_filename(self, filename: str) -> None:
        self._coll.delete(where={"filename": filename})

    def delete_all(self) -> None:
        self._client.delete_collection(self._coll_name)

    # ── Read ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int | None = None,
        filters: dict | None = None,
    ) -> List[dict]:
        """Semantic similarity search. Returns [{text, metadata, score}]."""
        k = k or config.RAG_TOP_K
        if self._coll.count() == 0:
            return []
        where = _build_where(filters) if filters else None
        results = self._coll.query(
            query_embeddings=[self._provider.embed_query(query)],
            n_results=min(k, self._coll.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": t, "metadata": m, "score": round(1.0 - d, 4)}
            for t, m, d in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ]

    # ── Stats ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "collection":    self._name,
            "internal_name": self._coll_name,
            "provider":      self._provider.provider_name,
            "dimensions":    self._provider.dimensions,
            "total_chunks":  self._coll.count(),
            "files":         self.list_files(),
        }

    def list_files(self) -> List[str]:
        if self._coll.count() == 0:
            return []
        metas = self._coll.get(include=["metadatas"])["metadatas"]
        return sorted({m.get("filename", "") for m in metas if m.get("filename")})


class RAGStore:
    """Registry of all named collections in the persistent store."""

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(config.VECTORSTORE_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

    def list_collections(self) -> List[str]:
        """Return logical collection names (strip the provider suffix)."""
        all_cols = self._client.list_collections()
        names = []
        for c in all_cols:
            n = c.name if hasattr(c, 'name') else str(c)
            # strip provider suffix __local_1024 or __azure_openai_1536
            base = n.rsplit("__", 1)[0] if "__" in n else n
            if base not in names:
                names.append(base)
        return sorted(names)

    def collection(self, name: str) -> RAGCollection:
        return RAGCollection(name)

    def delete_collection(self, name: str) -> None:
        provider = get_embedding_provider()
        internal = f"{name}__{provider.collection_suffix()}"
        try:
            self._client.delete_collection(internal)
        except Exception:
            pass


# ── Helpers ────────────────────────────────────────────────────────────

def _chunk_id(filename: str, chunk_idx: int, text: str) -> str:
    key = f"{filename}::{chunk_idx}::{text[:128]}"
    return hashlib.sha256(key.encode()).hexdigest()


def _build_where(filters: dict) -> dict:
    """Convert a flat {key: value} dict to ChromaDB $eq where clause."""
    if len(filters) == 1:
        k, v = next(iter(filters.items()))
        return {k: {"$eq": v}}
    return {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}
