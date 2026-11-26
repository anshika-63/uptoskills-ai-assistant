"""
Phase 4 End-to-End Integration Tests.
Verifies the complete RAG ingestion pipeline:
Upload -> Validation -> Storage -> Extraction -> Preprocessing -> Chunking -> Embeddings -> ChromaDB Indexing,
including duplicate uploads, document reprocessing, cascading deletion, and query-team retrieval interface.
"""

import pytest
from pathlib import Path

from backend.ingestion.embeddings import EmbeddingService
from backend.ingestion.metadata import MetadataManager
from backend.ingestion.pipeline import IngestionPipeline
from backend.ingestion.storage import StorageManager
from backend.ingestion.validator import FileValidator
from backend.ingestion.vector_store import VectorStoreManager
from backend.tests.conftest import (
    create_sample_pdf,
    create_sample_png,
    create_sample_txt,
    create_real_docx,
)


@pytest.fixture
def p4_pipeline(tmp_path):
    """Provides an isolated end-to-end IngestionPipeline with dedicated storage, DB, and ChromaDB."""
    storage_dir = tmp_path / "storage"
    db_path = tmp_path / "metadata.db"
    chroma_dir = tmp_path / "chroma"

    storage_mgr = StorageManager(base_dir=storage_dir)
    metadata_mgr = MetadataManager(db_path=db_path)
    validator = FileValidator()
    embedding_svc = EmbeddingService()
    vector_store = VectorStoreManager(
        persist_dir=chroma_dir,
        collection_name="p4_test_collection",
        embedding_service=embedding_svc,
    )

    return IngestionPipeline(
        validator=validator,
        storage_manager=storage_mgr,
        metadata_manager=metadata_mgr,
        embedding_service=embedding_svc,
        vector_store=vector_store,
    )


def test_pipeline_e2e_txt_embedding_and_indexing(p4_pipeline):
    """A new text file is ingested, chunked, embedded, and indexed into both SQLite and ChromaDB."""
    txt_content = (
        "UpToSkills Artificial Intelligence Course Syllabus.\n\n"
        "Module 1 covers Machine Learning basics, supervised vs unsupervised learning.\n"
        "Module 2 covers Deep Learning, Convolutional Neural Networks, and Transformers.\n"
        "Module 3 covers Retrieval-Augmented Generation (RAG) and ChromaDB vector indexing."
    ).encode("utf-8")

    result = p4_pipeline.process_file("ai_syllabus.txt", txt_content)

    assert result.success is True
    assert result.upload_success is True
    assert result.document_id is not None
    assert result.extraction_status == "success"
    assert result.embedding_status == "indexed"
    assert result.vector_count > 0

    doc_id = result.document_id

    # 1. Verify SQLite metadata record status is 'indexed'
    meta = p4_pipeline.metadata.get_by_id(doc_id)
    assert meta is not None
    assert meta.status == "indexed"

    # 2. Verify SQLite document_chunks persistence
    sqlite_chunks = p4_pipeline.metadata.get_chunks(doc_id)
    assert len(sqlite_chunks) == result.vector_count
    for c in sqlite_chunks:
        assert c["document_id"] == doc_id
        assert "chunk_id" in c
        assert "text" in c
        assert len(c["text"]) > 0

    # 3. Verify ChromaDB vector store persistence
    chroma_chunks = p4_pipeline.vector_store.get_document_chunks(doc_id)
    assert len(chroma_chunks) == result.vector_count
    assert p4_pipeline.vector_store.count_by_document(doc_id) == result.vector_count


def test_pipeline_duplicate_upload_vector_retrieval(p4_pipeline):
    """Duplicate upload retrieves existing chunk and vector records without duplicating in ChromaDB."""
    content = "Deterministic duplicate test content for vector store validation.".encode("utf-8")

    # First upload
    res1 = p4_pipeline.process_file("original.txt", content)
    assert res1.success is True
    assert res1.embedding_status == "indexed"
    initial_vectors = res1.vector_count
    assert initial_vectors > 0
    total_vectors_before = p4_pipeline.vector_store.count_vectors()

    # Second upload with identical content
    res2 = p4_pipeline.process_file("duplicate_copy.txt", content)
    assert res2.success is False
    assert res2.validation_result.is_duplicate is True
    assert res2.embedding_status == "retrieved"
    assert res2.vector_count == initial_vectors

    # Ensure total vector count in ChromaDB did NOT increase
    total_vectors_after = p4_pipeline.vector_store.count_vectors()
    assert total_vectors_after == total_vectors_before


def test_pipeline_reprocess_document_updates_vectors(p4_pipeline):
    """reprocess_document updates chunks and vectors cleanly without leaving orphans or duplicates."""
    content = "Initial document text for reprocessing test.".encode("utf-8")
    res = p4_pipeline.process_file("doc_to_reprocess.txt", content)
    doc_id = res.document_id
    orig_count = res.vector_count

    # Reprocess the document
    re_res = p4_pipeline.reprocess_document(doc_id)
    assert re_res is not None
    assert re_res.success is True
    assert re_res.embedding_status == "indexed"
    assert re_res.vector_count == orig_count

    # Vector store count should not have doubled
    assert p4_pipeline.vector_store.count_by_document(doc_id) == orig_count


def test_pipeline_delete_document_cascades_all_stores(p4_pipeline):
    """delete_document cascades across storage, SQLite metadata, extraction, chunks, and ChromaDB."""
    content = "Document to be permanently removed from all RAG storage layers.".encode("utf-8")
    res = p4_pipeline.process_file("to_delete.txt", content)
    doc_id = res.document_id
    storage_path = res.metadata.storage_path

    # Verify presence before deletion
    assert p4_pipeline.storage.file_exists(storage_path)
    assert p4_pipeline.metadata.get_by_id(doc_id) is not None
    assert len(p4_pipeline.metadata.get_chunks(doc_id)) > 0
    assert p4_pipeline.vector_store.count_by_document(doc_id) > 0

    # Execute cascading deletion
    deleted = p4_pipeline.delete_document(doc_id)
    assert deleted is True

    # Verify absence after deletion across all layers
    assert not p4_pipeline.storage.file_exists(storage_path)
    assert p4_pipeline.metadata.get_by_id(doc_id) is None
    assert p4_pipeline.metadata.get_extraction(doc_id) is None
    assert len(p4_pipeline.metadata.get_chunks(doc_id)) == 0
    assert p4_pipeline.vector_store.count_by_document(doc_id) == 0


def test_pipeline_non_text_files_skip_embedding(p4_pipeline):
    """Image and video files are stored and validated, but bypass text chunking and embedding."""
    png_bytes = create_sample_png()
    res = p4_pipeline.process_file("photo.png", png_bytes)

    assert res.success is True
    assert res.upload_success is True
    assert res.extraction_status == "not_applicable"
    assert res.embedding_status == "not_applicable"
    assert res.vector_count == 0
    assert p4_pipeline.vector_store.count_vectors() == 0


def test_query_team_access_interface(p4_pipeline):
    """
    Demonstrates how the downstream query-processing team interacts with the vector store.
    Tests semantic query retrieval, score ranking, and metadata contract.
    """
    course_text = (
        "Retrieval-Augmented Generation (RAG) architecture combines dense vector retrieval "
        "with large language model generation to produce grounded and hallucination-free answers."
    ).encode("utf-8")
    res = p4_pipeline.process_file("rag_overview.txt", course_text)
    doc_id = res.document_id

    # Downstream team accesses vector_store directly
    query = "How does RAG prevent hallucinations using vector retrieval?"
    hits = p4_pipeline.vector_store.query_by_text(query=query, top_k=1)

    assert len(hits) == 1
    top_hit = hits[0]

    # Verify query response contract for query-processing team
    assert "chunk_id" in top_hit
    assert "document_id" in top_hit
    assert top_hit["document_id"] == doc_id
    assert "text" in top_hit
    assert "Retrieval-Augmented Generation" in top_hit["text"]
    assert "metadata" in top_hit
    assert "score" in top_hit
    assert isinstance(top_hit["score"], float)
    assert top_hit["score"] > 0.0
