from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import config


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]: ...

    def collection_suffix(self) -> str:
        return f"{self.provider_name}_{self.dimensions}"


class LocalBGEEmbedding(EmbeddingProvider):
    _model = None

    @property
    def dimensions(self) -> int:
        return 1024

    @property
    def provider_name(self) -> str:
        return "local"

    def _load(self):
        if LocalBGEEmbedding._model is None:
            from sentence_transformers import SentenceTransformer
            LocalBGEEmbedding._model = SentenceTransformer(
                config.BGE_MODEL_NAME, device=config.BGE_DEVICE
            )
        return LocalBGEEmbedding._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._load().encode(
            texts,
            normalize_embeddings=True, batch_size=32,
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._load().encode(
            f"Represent this sentence for searching relevant passages: {text}",
            normalize_embeddings=True,
        ).tolist()


class AzureOpenAIEmbedding(EmbeddingProvider):
    _client = None

    @property
    def dimensions(self) -> int:
        return 1536

    @property
    def provider_name(self) -> str:
        return "azure_openai"

    def _get_client(self):
        if AzureOpenAIEmbedding._client is None:
            from openai import AzureOpenAI
            AzureOpenAIEmbedding._client = AzureOpenAI(
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                api_key=config.AZURE_OPENAI_API_KEY,
                api_version=config.AZURE_OPENAI_API_VERSION,
            )
        return AzureOpenAIEmbedding._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        all_emb: List[List[float]] = []
        for i in range(0, len(texts), 256):
            resp = client.embeddings.create(
                input=texts[i:i+256], model=config.AZURE_OPENAI_EMBED_DEPLOYMENT
            )
            all_emb.extend(e.embedding for e in resp.data)
        return all_emb

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_embedding_provider() -> EmbeddingProvider:
    p = config.EMBEDDING_PROVIDER.strip().lower()
    if p == "local":
        return LocalBGEEmbedding()
    if p == "azure_openai":
        return AzureOpenAIEmbedding()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER='{p}'. Valid: 'local', 'azure_openai'")
