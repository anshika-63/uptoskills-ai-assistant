"""
Part B: Document Lifecycle Verification Suite for UpToSkills RAG Ingestion Pipeline.
Tests:
1. Duplicate detection: upload the same document twice, verify duplicate status and no extra vectors created.
2. Document reprocessing: reprocess an existing document and verify old chunks and vectors are replaced atomically.
3. Content update: verify updated content is indexed and immediately searchable.
4. Document deletion: delete a document and verify physical file, transcript JSON, ChromaDB vectors, and SQLite records are removed.
5. Metadata & ChromaDB synchronization: verify chunk count in SQLite equals vector count in ChromaDB.
6. Application restart persistence: simulate application restart and confirm vectors and metadata persist.
7. Failure consistency: verify invalid/corrupted uploads or extraction errors do not create orphan records.
"""

from pathlib import Path
import pytest

from backend.ingestion.embeddings import EmbeddingService
from backend.ingestion.metadata import MetadataManager
from backend.ingestion.models import ValidationStatus
from backend.ingestion.pipeline import IngestionPipeline
from backend.ingestion.storage import StorageManager
from backend.ingestion.validator import FileValidator
from backend.ingestion.vector_store import VectorStoreManager
from backend.tests.conftest import create_real_docx, create_sample_png, create_sample_txt


def test_lifecycle_duplicate_upload_no_extra_vectors(multimodal_pipeline):
    """
    1. Upload the same document twice:
       - First upload indexes chunks into ChromaDB.
       - Second upload identifies SHA-256 duplicate, marks is_duplicate=True, upload_success=False.
       - Verifies that zero extra vectors are added to ChromaDB.
    """
    content = create_real_docx(paragraphs=["Unique research text for duplicate verification."])
    filename = "unique_study.docx"

    # First upload
    res1 = multimodal_pipeline.process_file(filename, content)
    assert res1.success is True
    assert res1.upload_success is True
    doc_id = res1.document_id
    initial_vector_count = res1.vector_count
    assert initial_vector_count >= 1

    total_vectors_before = multimodal_pipeline.vector_store.count_vectors()

    # Second upload of identical file
    res2 = multimodal_pipeline.process_file("copy_of_unique_study.docx", content)
    assert res2.success is False
    assert res2.upload_success is False
    assert res2.validation_result.is_duplicate is True
    assert res2.validation_result.status == ValidationStatus.DUPLICATE
    assert res2.document_id == doc_id
    assert res2.embedding_status == "retrieved"

    # Vector store count must NOT have increased
    total_vectors_after = multimodal_pipeline.vector_store.count_vectors()
    assert total_vectors_after == total_vectors_before
    assert multimodal_pipeline.vector_store.count_by_document(doc_id) == initial_vector_count


def test_lifecycle_reprocess_document_replaces_old_chunks(multimodal_pipeline):
    """
    2. Reprocess an existing document:
       - Calls pipeline.reprocess_document(doc_id).
       - Verifies old chunks in SQLite and vectors in ChromaDB are replaced atomically
         without creating duplicate records.
    """
    content = create_real_docx(paragraphs=["Initial version of document text for reprocessing test."])
    res = multimodal_pipeline.process_file("doc_to_reprocess.docx", content)
    assert res.success is True
    doc_id = res.document_id

    initial_chunks = multimodal_pipeline.metadata.get_chunks(doc_id)
    initial_count = len(initial_chunks)
    assert initial_count >= 1

    # Execute reprocess
    rep_res = multimodal_pipeline.reprocess_document(doc_id)
    assert rep_res is not None
    assert rep_res.success is True

    # Verify chunk count matches exactly (no duplicates accumulated)
    new_chunks = multimodal_pipeline.metadata.get_chunks(doc_id)
    assert len(new_chunks) == initial_count

    # Verify ChromaDB vector count matches exactly
    vector_cnt = multimodal_pipeline.vector_store.count_by_document(doc_id)
    assert vector_cnt == initial_count


def test_lifecycle_updated_document_content_becomes_searchable(multimodal_pipeline):
    """
    3. Update document content and reprocess:
       - Modifies physical file content on disk with a distinctive new keyword.
       - Reprocesses document.
       - Queries ChromaDB with the new keyword and verifies top retrieval.
    """
    content_v1 = create_real_docx(paragraphs=["Standard robotics control systems and kinematics."])
    res = multimodal_pipeline.process_file("robotics.docx", content_v1)
    doc_id = res.document_id

    # Overwrite stored file on disk with updated content containing new keyword
    meta = multimodal_pipeline.metadata.get_by_id(doc_id)
    stored_path = multimodal_pipeline.storage.base_dir / meta.storage_path
    content_v2 = create_real_docx(paragraphs=[
        "CRISPR Cas9 genomic sequence modification and molecular editing techniques."
    ])
    stored_path.write_bytes(content_v2)

    # Force cache refresh and reprocess
    multimodal_pipeline.metadata.delete_extraction(doc_id)
    rep_res = multimodal_pipeline.reprocess_document(doc_id)
    assert rep_res.success is True

    # Query ChromaDB for the updated topic
    results = multimodal_pipeline.vector_store.query_by_text(
        "CRISPR Cas9 genomic sequence editing", top_k=1
    )
    assert len(results) == 1
    assert results[0]["document_id"] == doc_id
    assert "CRISPR" in results[0]["text"]


def test_lifecycle_delete_document_removes_all_artifacts(multimodal_pipeline):
    """
    4. Delete a document:
       - Calls pipeline.delete_document(doc_id).
       - Verifies physical file on disk is deleted.
       - Verifies ChromaDB vectors are deleted.
       - Verifies SQLite metadata, extraction, and chunks are deleted.
    """
    content = create_sample_txt()
    res = multimodal_pipeline.process_file("to_delete.txt", content)
    doc_id = res.document_id

    # Verify initially present
    meta = multimodal_pipeline.metadata.get_by_id(doc_id)
    assert meta is not None
    assert (multimodal_pipeline.storage.base_dir / meta.storage_path).exists()
    assert multimodal_pipeline.vector_store.count_by_document(doc_id) >= 1
    assert len(multimodal_pipeline.metadata.get_chunks(doc_id)) >= 1

    # Delete
    deleted = multimodal_pipeline.delete_document(doc_id)
    assert deleted is True

    # Verify physical file removed
    assert not (multimodal_pipeline.storage.base_dir / meta.storage_path).exists()

    # Verify ChromaDB vectors removed
    assert multimodal_pipeline.vector_store.count_by_document(doc_id) == 0

    # Verify SQLite records removed
    assert multimodal_pipeline.metadata.get_by_id(doc_id) is None
    assert multimodal_pipeline.metadata.get_extraction(doc_id) is None
    assert len(multimodal_pipeline.metadata.get_chunks(doc_id)) == 0


def test_lifecycle_metadata_and_chromadb_synchronization(multimodal_pipeline):
    """
    5. Ingest multiple diverse documents and verify that for every cataloged document,
       the chunk count in SQLite is exactly synchronized with the vector count in ChromaDB.
    """
    doc1 = create_real_docx(paragraphs=["Document alpha content about distributed systems."])
    doc2 = create_sample_txt()

    res1 = multimodal_pipeline.process_file("alpha.docx", doc1)
    res2 = multimodal_pipeline.process_file("beta.txt", doc2)

    all_docs = multimodal_pipeline.metadata.list_documents()
    assert len(all_docs) >= 2

    for d in all_docs:
        sqlite_chunks = multimodal_pipeline.metadata.get_chunks(d.document_id)
        chroma_count = multimodal_pipeline.vector_store.count_by_document(d.document_id)
        assert len(sqlite_chunks) == chroma_count, f"Sync mismatch for {d.document_id}"


def test_lifecycle_application_restart_persistence(tmp_path, validator):
    """
    6. Simulate application restart:
       - Instance 1 ingests documents, chunks, and vectors into persistent storage.
       - Destroy instance 1.
       - Instance 2 initializes pointing to the same storage and ChromaDB directories.
       - Confirms all documents, chunks, and vector embeddings are intact and searchable.
    """
    storage_dir = tmp_path / "shared_storage"
    db_path = tmp_path / "shared_metadata.db"
    chroma_dir = tmp_path / "shared_chroma"

    emb_svc1 = EmbeddingService()
    v_store1 = VectorStoreManager(persist_dir=chroma_dir, collection_name="restart_test", embedding_service=emb_svc1)
    pipeline1 = IngestionPipeline(
        validator=validator,
        storage_manager=StorageManager(base_dir=storage_dir),
        metadata_manager=MetadataManager(db_path=db_path),
        embedding_service=emb_svc1,
        vector_store=v_store1,
        enable_multimodal=True,
    )

    docx_bytes = create_real_docx(paragraphs=["Knowledge retrieval across application restarts and cold reboots."])
    res = pipeline1.process_file("cold_restart.docx", docx_bytes)
    assert res.success is True
    doc_id = res.document_id
    initial_vectors = v_store1.count_vectors()

    # Destroy pipeline1 and vector store connection
    del pipeline1
    del v_store1

    # Instantiate pipeline2 (simulating cold application restart)
    emb_svc2 = EmbeddingService()
    v_store2 = VectorStoreManager(persist_dir=chroma_dir, collection_name="restart_test", embedding_service=emb_svc2)
    pipeline2 = IngestionPipeline(
        validator=validator,
        storage_manager=StorageManager(base_dir=storage_dir),
        metadata_manager=MetadataManager(db_path=db_path),
        embedding_service=emb_svc2,
        vector_store=v_store2,
        enable_multimodal=True,
    )

    # Verify documents and chunks persisted in SQLite
    restarted_meta = pipeline2.metadata.get_by_id(doc_id)
    assert restarted_meta is not None
    assert restarted_meta.original_filename == "cold_restart.docx"

    restarted_chunks = pipeline2.metadata.get_chunks(doc_id)
    assert len(restarted_chunks) >= 1

    # Verify ChromaDB vectors persisted
    assert v_store2.count_vectors() == initial_vectors
    assert v_store2.count_by_document(doc_id) == len(restarted_chunks)

    # Verify semantic search works on persisted vectors without re-ingesting
    search_hit = v_store2.query_by_text("cold reboots and application restarts", top_k=1)
    assert len(search_hit) == 1
    assert search_hit[0]["document_id"] == doc_id


def test_lifecycle_failure_consistency_no_orphan_records(multimodal_pipeline):
    """
    7. Test failure scenarios to verify inconsistent or orphan records are never created:
       - Corrupted file failing validation: 0 files stored, 0 SQLite rows, 0 vectors.
       - Extraction failure: status marked 'failed', 0 chunks in SQLite, 0 vectors in ChromaDB.
    """
    # Scenario A: Corrupted file failing validation
    corrupt_bytes = b"CORRUPTED_NON_PDF_PAYLOAD"
    res_val_fail = multimodal_pipeline.process_file("bad_doc.pdf", corrupt_bytes)
    assert res_val_fail.success is False
    assert res_val_fail.upload_success is False
    assert res_val_fail.document_id is None
    assert res_val_fail.vector_count == 0

    # Verify no records entered SQLite or ChromaDB
    all_docs = multimodal_pipeline.metadata.list_documents()
    assert not any(d.original_filename == "bad_doc.pdf" for d in all_docs)
    assert multimodal_pipeline.vector_store.count_vectors() == 0

    # Scenario B: File passing validation header but failing deeper extraction
    corrupt_pdf_body = b"%PDF-1.4\nCorrupted internal stream without xref or EOF"
    res_ext_fail = multimodal_pipeline.process_file("bad_stream.pdf", corrupt_pdf_body)
    # File is stored but extraction fails
    assert res_ext_fail.extraction_status == "failed"
    assert res_ext_fail.extraction_error is not None
    assert res_ext_fail.vector_count == 0
    assert len(res_ext_fail.preprocessed_content or []) == 0

    # Verify zero chunks or vectors were created in SQLite or ChromaDB for this document
    doc_id = res_ext_fail.document_id
    assert len(multimodal_pipeline.metadata.get_chunks(doc_id)) == 0
    assert multimodal_pipeline.vector_store.count_by_document(doc_id) == 0
