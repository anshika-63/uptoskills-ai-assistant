"""
Unit and functional tests for VectorStoreManager (Phase 4).
Tests ChromaDB initialization, persistence across restarts, chunk addition,
document-level atomic updates, cascading deletion, metadata sanitization,
and semantic vector similarity queries.
"""

import pytest
from pathlib import Path
from backend.ingestion.embeddings import EmbeddingService
from backend.ingestion.vector_store import VectorStoreManager


@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Provides an isolated temporary directory for ChromaDB storage."""
    chroma_dir = tmp_path / "test_chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir


@pytest.fixture
def embedding_service():
    """Provides an EmbeddingService instance."""
    return EmbeddingService()


@pytest.fixture
def vector_store(temp_chroma_dir, embedding_service):
    """Provides a fresh VectorStoreManager with isolated storage and test collection."""
    return VectorStoreManager(
        persist_dir=temp_chroma_dir,
        collection_name="test_uptoskills_rag",
        embedding_service=embedding_service,
    )


def test_vector_store_initialization(vector_store, temp_chroma_dir):
    """VectorStoreManager initializes persistent ChromaDB and collection."""
    assert vector_store.collection_name == "test_uptoskills_rag"
    assert vector_store.persist_dir == temp_chroma_dir
    assert vector_store.count_vectors() == 0
    assert vector_store.list_indexed_document_ids() == []


def test_sanitize_metadata(vector_store):
    """VectorStoreManager.sanitize_metadata cleanses non-conforming types for ChromaDB."""
    raw_meta = {
        "doc_id": "doc_123",
        "chunk_index": 0,
        "score": 0.95,
        "is_active": True,
        "none_value": None,
        "nested_dict": {"inner_key": 42},
        "nested_list": ["tag1", "tag2"],
    }
    sanitized = vector_store.sanitize_metadata(raw_meta)

    # Valid primitives preserved
    assert sanitized["doc_id"] == "doc_123"
    assert sanitized["chunk_index"] == 0
    assert sanitized["score"] == 0.95
    assert sanitized["is_active"] is True

    # None removed
    assert "none_value" not in sanitized

    # Complex types stringified
    assert sanitized["nested_dict"] == '{"inner_key": 42}'
    assert sanitized["nested_list"] == '["tag1", "tag2"]'


def test_add_chunks_and_count(vector_store, embedding_service):
    """Adding embedded chunks indexes them into ChromaDB and reflects in vector counts."""
    doc_id = "doc_test_001"
    chunks = [
        {
            "chunk_id": f"{doc_id}_chunk_0",
            "document_id": doc_id,
            "text": "First chunk discussing computer science principles.",
            "chunk_index": 0,
            "page_number": 1,
            "filename": "cs_notes.pdf",
        },
        {
            "chunk_id": f"{doc_id}_chunk_1",
            "document_id": doc_id,
            "text": "Second chunk discussing data structures and algorithms.",
            "chunk_index": 1,
            "page_number": 2,
            "filename": "cs_notes.pdf",
        },
    ]

    # Generate embeddings and attach
    chunks = embedding_service.attach_embeddings_to_chunks(chunks)
    added = vector_store.add_chunks(chunks)

    assert added == 2
    assert vector_store.count_vectors() == 2
    assert vector_store.count_by_document(doc_id) == 2
    assert vector_store.count_by_document("nonexistent") == 0
    assert vector_store.list_indexed_document_ids() == [doc_id]


def test_get_document_chunks(vector_store, embedding_service):
    """get_document_chunks retrieves all stored chunks for a specific document."""
    doc_id = "doc_test_002"
    chunks = [
        {
            "chunk_id": f"{doc_id}_chunk_0",
            "document_id": doc_id,
            "text": "Content of chunk 0.",
            "chunk_index": 0,
        },
        {
            "chunk_id": f"{doc_id}_chunk_1",
            "document_id": doc_id,
            "text": "Content of chunk 1.",
            "chunk_index": 1,
        },
    ]
    chunks = embedding_service.attach_embeddings_to_chunks(chunks)
    vector_store.add_chunks(chunks)

    stored = vector_store.get_document_chunks(doc_id)
    assert len(stored) == 2

    # Verify order and structure
    ids = [c["chunk_id"] for c in stored]
    assert f"{doc_id}_chunk_0" in ids
    assert f"{doc_id}_chunk_1" in ids


def test_update_document_replaces_without_duplicates(vector_store, embedding_service):
    """update_document atomically deletes old vectors and inserts new ones."""
    doc_id = "doc_test_003"

    # Initial 3 chunks
    initial_chunks = [
        {"chunk_id": f"{doc_id}_chunk_{i}", "document_id": doc_id, "text": f"Old version chunk {i}"}
        for i in range(3)
    ]
    initial_chunks = embedding_service.attach_embeddings_to_chunks(initial_chunks)
    vector_store.add_chunks(initial_chunks)
    assert vector_store.count_by_document(doc_id) == 3

    # Update with 2 new chunks
    updated_chunks = [
        {"chunk_id": f"{doc_id}_chunk_{i}", "document_id": doc_id, "text": f"New version chunk {i}"}
        for i in range(2)
    ]
    updated_chunks = embedding_service.attach_embeddings_to_chunks(updated_chunks)
    indexed_count = vector_store.update_document(doc_id, updated_chunks)

    assert indexed_count == 2
    assert vector_store.count_by_document(doc_id) == 2
    assert vector_store.count_vectors() == 2

    # Verify text was updated
    stored = vector_store.get_document_chunks(doc_id)
    assert len(stored) == 2
    assert all("New version" in c["text"] for c in stored)


def test_delete_document(vector_store, embedding_service):
    """delete_document removes all vectors associated with a document ID."""
    doc1 = "doc_keep"
    doc2 = "doc_delete"

    c1 = embedding_service.attach_embeddings_to_chunks([
        {"chunk_id": f"{doc1}_chunk_0", "document_id": doc1, "text": "Keep this document."}
    ])
    c2 = embedding_service.attach_embeddings_to_chunks([
        {"chunk_id": f"{doc2}_chunk_0", "document_id": doc2, "text": "Delete this document."},
        {"chunk_id": f"{doc2}_chunk_1", "document_id": doc2, "text": "Delete this chunk too."},
    ])

    vector_store.add_chunks(c1)
    vector_store.add_chunks(c2)
    assert vector_store.count_vectors() == 3

    deleted_count = vector_store.delete_document(doc2)
    assert deleted_count == 2
    assert vector_store.count_vectors() == 1
    assert vector_store.count_by_document(doc2) == 0
    assert vector_store.count_by_document(doc1) == 1
    assert vector_store.list_indexed_document_ids() == [doc1]


def test_vector_persistence_across_instances(temp_chroma_dir, embedding_service):
    """Data stored in ChromaDB persists across different VectorStoreManager instances."""
    # First instance: add vector
    store1 = VectorStoreManager(
        persist_dir=temp_chroma_dir,
        collection_name="persist_test",
        embedding_service=embedding_service,
    )
    chunk = embedding_service.attach_embeddings_to_chunks([
        {"chunk_id": "p_chunk_0", "document_id": "p_doc", "text": "Persistent knowledge chunk."}
    ])
    store1.add_chunks(chunk)
    assert store1.count_vectors() == 1

    # Second instance: re-open same directory and collection
    store2 = VectorStoreManager(
        persist_dir=temp_chroma_dir,
        collection_name="persist_test",
        embedding_service=embedding_service,
    )
    assert store2.count_vectors() == 1
    stored = store2.get_document_chunks("p_doc")
    assert len(stored) == 1
    assert stored[0]["text"] == "Persistent knowledge chunk."


def test_similarity_search_and_query_by_text(vector_store, embedding_service):
    """Semantic vector query returns the most relevant chunk with highest similarity."""
    chunks = [
        {
            "chunk_id": "ml_chunk",
            "document_id": "doc_ai",
            "text": "Convolutional neural networks and deep learning models for image recognition.",
            "topic": "ai",
        },
        {
            "chunk_id": "finance_chunk",
            "document_id": "doc_fin",
            "text": "Stock market trading strategies, corporate balance sheets, and dividends.",
            "topic": "finance",
        },
    ]
    chunks = embedding_service.attach_embeddings_to_chunks(chunks)
    vector_store.add_chunks(chunks)

    # Query for AI topic
    results = vector_store.query_by_text("neural networks and vision architectures", top_k=2)
    assert len(results) == 2

    # Top result should be the AI chunk
    top_hit = results[0]
    assert top_hit["chunk_id"] == "ml_chunk"
    assert top_hit["document_id"] == "doc_ai"
    assert "score" in top_hit
    assert top_hit["score"] > 0.0
    # Top hit score should exceed second hit score
    assert top_hit["score"] > results[1]["score"]
