"""
Unit tests for EmbeddingModelManager and EmbeddingService (Phase 4).
Tests embedding model loading, singleton caching, text embedding generation,
L2 normalization, chunk embedding enrichment, empty string edge cases, and dimension compliance.
"""

import math
import pytest
from unittest.mock import MagicMock, patch

from backend.config import EMBEDDING_MODEL_NAME
from backend.ingestion.embeddings import EmbeddingModelManager, EmbeddingService


@pytest.fixture(autouse=True)
def clean_model_singleton():
    """Ensure singleton is reset before and after each test."""
    EmbeddingModelManager.reset()
    yield
    EmbeddingModelManager.reset()


def test_embedding_model_singleton():
    """EmbeddingModelManager returns the same model instance across repeated calls."""
    m1 = EmbeddingModelManager.get_model()
    m2 = EmbeddingModelManager.get_model()
    assert m1 is m2
    assert EmbeddingModelManager.is_loaded()


def test_embedding_model_device_info():
    """Model manager provides accurate device and dimension metadata."""
    model = EmbeddingModelManager.get_model()
    dim = EmbeddingModelManager.get_dimension()
    assert dim == 384  # all-MiniLM-L6-v2 default dimension
    assert isinstance(dim, int)


def test_embedding_service_initialization():
    """EmbeddingService initializes with default or custom configuration."""
    service = EmbeddingService(model_name=EMBEDDING_MODEL_NAME, normalize=True)
    assert service.dimension == 384
    assert service.model_name == EMBEDDING_MODEL_NAME
    assert service.normalize is True


def test_embed_single_text():
    """Embedding a single string produces a 384-dimensional unit vector."""
    service = EmbeddingService()
    text = "UpToSkills Artificial Intelligence RAG document ingestion pipeline."
    vector = service.embed_text(text)

    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(val, float) for val in vector)

    # Verify L2 normalization: sum of squares should be approximately 1.0
    l2_norm = math.sqrt(sum(v * v for v in vector))
    assert pytest.approx(l2_norm, abs=1e-3) == 1.0


def test_embed_empty_or_whitespace_text():
    """Embedding empty or whitespace-only text returns a zero vector without error."""
    service = EmbeddingService()
    zero_vec_empty = service.embed_text("")
    zero_vec_ws = service.embed_text("   \n\t  ")

    assert len(zero_vec_empty) == 384
    assert all(v == 0.0 for v in zero_vec_empty)
    assert len(zero_vec_ws) == 384
    assert all(v == 0.0 for v in zero_vec_ws)


def test_embed_texts_batch():
    """Embedding a batch of texts returns matching list of vectors."""
    service = EmbeddingService()
    texts = [
        "First chunk discussing machine learning concepts.",
        "Second chunk discussing database vector indexing.",
        "Third chunk about speech-to-text transcription.",
    ]
    vectors = service.embed_texts(texts)

    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == 384
        l2_norm = math.sqrt(sum(v * v for v in vec))
        assert pytest.approx(l2_norm, abs=1e-3) == 1.0

    # Vectors for different texts should not be identical
    assert vectors[0] != vectors[1]


def test_embed_chunks_dict():
    """Embedding chunk dictionaries returns vectors corresponding to chunk text."""
    service = EmbeddingService()
    chunks = [
        {"chunk_id": "doc1_c0", "text": "This is chunk 0 about software engineering."},
        {"chunk_id": "doc1_c1", "text": "This is chunk 1 about system architecture."},
    ]
    vectors = service.embed_chunks(chunks)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def test_attach_embeddings_to_chunks():
    """attach_embeddings_to_chunks mutates/enriches chunk dicts with 'embedding' key."""
    service = EmbeddingService()
    chunks = [
        {"chunk_id": "doc1_c0", "text": "Deep learning models and neural networks."},
        {"chunk_id": "doc1_c1", "text": "ChromaDB vector database for embeddings."},
    ]
    enriched = service.attach_embeddings_to_chunks(chunks)

    assert len(enriched) == 2
    for c in enriched:
        assert "embedding" in c
        assert isinstance(c["embedding"], list)
        assert len(c["embedding"]) == 384


def test_attach_embeddings_empty_list():
    """Attaching embeddings to an empty chunk list returns empty list safely."""
    service = EmbeddingService()
    res = service.attach_embeddings_to_chunks([])
    assert res == []
