"""
Unit tests for the MetadataManager component.
Verifies SQLite persistence, retrieval, SHA-256 duplicate index lookup,
filtering, text search, and aggregations.
"""

from backend.ingestion.models import DocumentMetadata


def make_sample_metadata(
    doc_id: str = "doc_001",
    filename: str = "curriculum.pdf",
    category: str = "documents",
    file_type: str = ".pdf",
    size: int = 1024,
    sha256: str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=doc_id,
        original_filename=filename,
        safe_filename=filename,
        category=category,
        file_type=file_type,
        mime_type="application/pdf" if category == "documents" else "image/png",
        file_size_bytes=size,
        sha256_hash=sha256,
        storage_path=f"{category}/{doc_id}_{filename}",
        upload_timestamp="2026-11-17T21:00:00Z",
        status="stored",
    )


def test_metadata_save_and_get_by_id(metadata_manager):
    meta = make_sample_metadata(doc_id="doc_test_1")
    assert metadata_manager.save(meta)

    retrieved = metadata_manager.get_by_id("doc_test_1")
    assert retrieved is not None
    assert retrieved.document_id == "doc_test_1"
    assert retrieved.original_filename == "curriculum.pdf"
    assert retrieved.file_size_bytes == 1024
    assert retrieved.category == "documents"


def test_metadata_get_by_hash(metadata_manager):
    hash_val = "1111222233334444555566667777888811112222333344445555666677778888"
    meta = make_sample_metadata(doc_id="doc_hash_1", sha256=hash_val)
    metadata_manager.save(meta)

    # Lookup by identical hash
    found = metadata_manager.get_by_hash(hash_val)
    assert found is not None
    assert found.document_id == "doc_hash_1"

    # Non-existent hash
    assert metadata_manager.get_by_hash("nonexistent_hash") is None


def test_metadata_list_and_category_filter(metadata_manager):
    m1 = make_sample_metadata(doc_id="doc_1", filename="doc1.pdf", category="documents")
    m2 = make_sample_metadata(doc_id="doc_2", filename="doc2.docx", category="documents", file_type=".docx")
    m3 = make_sample_metadata(doc_id="img_1", filename="pic.png", category="images", file_type=".png")

    metadata_manager.save(m1)
    metadata_manager.save(m2)
    metadata_manager.save(m3)

    all_docs = metadata_manager.list_documents(category="All")
    assert len(all_docs) == 3

    doc_only = metadata_manager.list_documents(category="documents")
    assert len(doc_only) == 2
    assert all(d.category == "documents" for d in doc_only)

    img_only = metadata_manager.list_documents(category="images")
    assert len(img_only) == 1
    assert img_only[0].document_id == "img_1"


def test_metadata_search(metadata_manager):
    m1 = make_sample_metadata(doc_id="doc_search_alpha", filename="machine_learning_syllabus.pdf")
    m2 = make_sample_metadata(doc_id="doc_search_beta", filename="internship_guide.pdf")

    metadata_manager.save(m1)
    metadata_manager.save(m2)

    # Search by filename keyword
    results_ml = metadata_manager.list_documents(search="syllabus")
    assert len(results_ml) == 1
    assert results_ml[0].document_id == "doc_search_alpha"

    # Search by document ID
    results_id = metadata_manager.list_documents(search="beta")
    assert len(results_id) == 1
    assert results_id[0].document_id == "doc_search_beta"


def test_metadata_aggregations(metadata_manager):
    m1 = make_sample_metadata(doc_id="d1", category="documents", size=1000)
    m2 = make_sample_metadata(doc_id="d2", category="documents", size=2000)
    m3 = make_sample_metadata(doc_id="i1", category="images", size=5000)

    metadata_manager.save(m1)
    metadata_manager.save(m2)
    metadata_manager.save(m3)

    assert metadata_manager.count_total() == 3
    counts = metadata_manager.count_by_category()
    assert counts["documents"] == 2
    assert counts["images"] == 1
    assert counts["videos"] == 0

    assert metadata_manager.get_total_storage_bytes() == 8000


def test_metadata_delete(metadata_manager):
    m1 = make_sample_metadata(doc_id="del_me")
    metadata_manager.save(m1)
    assert metadata_manager.get_by_id("del_me") is not None

    deleted = metadata_manager.delete("del_me")
    assert deleted
    assert metadata_manager.get_by_id("del_me") is None
