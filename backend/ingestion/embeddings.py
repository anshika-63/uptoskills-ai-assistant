"""
Embedding generation service for the UpToSkills RAG ingestion pipeline.

Converts processed text chunks into dense numerical vectors using locally
runnable sentence-transformers models (e.g. all-MiniLM-L6-v2) by default.
Ensures zero dependency on paid API keys, high throughput on CPU/CUDA,
and full compatibility with the downstream query-processing team's vector retrieval.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from backend.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_NORMALIZE,
)

logger = logging.getLogger("rag.embeddings")


class EmbeddingModelManager:
    """
    Lazy-loading singleton cache for local sentence-transformer models.
    Avoids expensive model reloads on every ingestion or embedding request.
    """

    _instance: Optional["EmbeddingModelManager"] = None
    _models: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "EmbeddingModelManager":
        if cls._instance is None:
            cls._instance = EmbeddingModelManager()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clears cached model instances (useful in unit testing)."""
        cls._models.clear()
        cls._instance = None

    @classmethod
    def is_loaded(cls, model_name: Optional[str] = None, device: Optional[str] = None) -> bool:
        """Checks if a model is currently loaded in memory."""
        target_name = model_name or EMBEDDING_MODEL_NAME
        target_device = device or EMBEDDING_DEVICE
        cache_key = f"{target_name}_{target_device}"
        return cache_key in cls._models

    @classmethod
    def get_dimension(cls, model_name: Optional[str] = None, device: Optional[str] = None) -> int:
        """Returns the embedding vector dimension for the loaded or configured model."""
        model = cls.get_model(model_name=model_name, device=device)
        get_dim = getattr(
            model, "get_embedding_dimension", getattr(model, "get_sentence_embedding_dimension", None)
        )
        dim = get_dim() if callable(get_dim) else 384
        return int(dim or 384)

    @classmethod
    def get_model(
        cls,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Any:
        """
        Retrieves or loads a sentence-transformers SentenceTransformer instance.
        Callable as both classmethod and instance method.
        """
        target_name = model_name or EMBEDDING_MODEL_NAME
        target_device = device or EMBEDDING_DEVICE
        cache_key = f"{target_name}_{target_device}"

        if cache_key not in cls._models:
            logger.info(
                "Loading embedding model '%s' on device '%s'...",
                target_name,
                target_device,
            )
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(target_name, device=target_device)
                cls._models[cache_key] = model
                get_dim = getattr(
                    model, "get_embedding_dimension", getattr(model, "get_sentence_embedding_dimension", None)
                )
                dim = get_dim() if callable(get_dim) else 384
                logger.info(
                    "Embedding model '%s' loaded successfully (dimension=%d).",
                    target_name,
                    dim or 384,
                )
            except Exception as exc:
                logger.error("Failed to load embedding model '%s': %s", target_name, exc)
                raise RuntimeError(
                    f"Could not load local embedding model '{target_name}': {exc}"
                ) from exc

        return cls._models[cache_key]


class EmbeddingService:
    """
    Service responsible for converting single text strings and preprocessed
    document chunks into normalized numerical vectors.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize_embeddings: Optional[bool] = None,
        normalize: Optional[bool] = None,
    ):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.device = device or EMBEDDING_DEVICE
        self.batch_size = batch_size or EMBEDDING_BATCH_SIZE
        norm_val = normalize_embeddings if normalize_embeddings is not None else normalize
        self.normalize_embeddings = (
            norm_val
            if norm_val is not None
            else EMBEDDING_NORMALIZE
        )
        self._model_manager = EmbeddingModelManager.get_instance()
        self._cached_dimension: Optional[int] = None

    @property
    def normalize(self) -> bool:
        """Alias for normalize_embeddings."""
        return self.normalize_embeddings

    @property
    def model(self) -> Any:
        """Returns the loaded SentenceTransformer model instance."""
        return self._model_manager.get_model(self.model_name, self.device)

    @property
    def dimension(self) -> int:
        """Returns the output vector dimensionality of the embedding model."""
        if self._cached_dimension is None:
            try:
                get_dim = getattr(
                    self.model, "get_embedding_dimension", getattr(self.model, "get_sentence_embedding_dimension", None)
                )
                dim = get_dim() if callable(get_dim) else 384
                self._cached_dimension = int(dim) if dim else 384
            except Exception:
                self._cached_dimension = 384
        return self._cached_dimension

    def embed_text(self, text: str) -> List[float]:
        """
        Generates a vector embedding for a single text string.

        Args:
            text: Plain text string to embed.

        Returns:
            List[float]: Numerical vector.
        """
        if not isinstance(text, str):
            raise TypeError("Text input must be a string.")

        clean = text.strip()
        if not clean:
            # Return zero-vector of model dimension for empty text
            return [0.0] * self.dimension

        embeddings = self.embed_texts([clean])
        return embeddings[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a list of text strings in batches.

        Args:
            texts: List of strings to encode.

        Returns:
            List[List[float]]: List of numerical vectors matching the input order.
        """
        if not texts:
            return []

        # Handle empty/whitespace strings by temporarily replacing them with placeholder
        # and zeroing out the resulting vector
        cleaned_texts: List[str] = []
        empty_mask: List[bool] = []
        for t in texts:
            val = t.strip() if isinstance(t, str) else ""
            if val:
                cleaned_texts.append(val)
                empty_mask.append(False)
            else:
                cleaned_texts.append(" ")
                empty_mask.append(True)

        try:
            raw_vectors = self.model.encode(
                cleaned_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            )
        except Exception as exc:
            logger.error("Embedding generation failed: %s", exc)
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

        dim = self.dimension
        result: List[List[float]] = []
        for idx, is_empty in enumerate(empty_mask):
            if is_empty:
                result.append([0.0] * dim)
            else:
                result.append([float(x) for x in raw_vectors[idx]])

        return result

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        text_key: str = "text",
    ) -> List[List[float]]:
        """
        Extracts chunk text from processed chunk dictionaries and returns vectors.

        Args:
            chunks: List of chunk dictionaries (as produced by preprocessing).
            text_key: Dictionary key containing the chunk's text (default 'text').

        Returns:
            List[List[float]]: Generated embedding vectors.
        """
        if not chunks:
            return []

        texts = [c.get(text_key, "") if isinstance(c, dict) else str(c) for c in chunks]
        return self.embed_texts(texts)

    def attach_embeddings_to_chunks(
        self,
        chunks: List[Dict[str, Any]],
        text_key: str = "text",
        embedding_key: str = "embedding",
    ) -> List[Dict[str, Any]]:
        """
        Computes embeddings and attaches them in-place to copies of the chunk dictionaries.

        Returns:
            List[Dict[str, Any]]: Chunks enriched with embedding vector and metadata.
        """
        if not chunks:
            return []

        embeddings = self.embed_chunks(chunks, text_key=text_key)
        enriched: List[Dict[str, Any]] = []

        for chunk, vec in zip(chunks, embeddings):
            c_copy = dict(chunk)
            c_copy[embedding_key] = vec
            c_copy["embedding_model"] = self.model_name
            c_copy["embedding_dimension"] = len(vec)
            enriched.append(c_copy)

        return enriched
