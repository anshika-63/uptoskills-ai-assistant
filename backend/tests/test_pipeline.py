"""
Integration tests for the complete IngestionPipeline.
Verifies the end-to-end flow: upload, validation, duplicate detection,
persistent categorized storage, and metadata recording.
"""

from backend.ingestion.models import ValidationStatus
from backend.tests.conftest import (
    create_sample_docx,
    create_sample_jpeg,
    create_sample_mp4,
    create_sample_pdf,
    create_sample_png,
    create_sample_txt,
)


def test_pipeline_ingest_document(pipeline, storage_manager, metadata_manager):
    pdf_bytes = create_sample_pdf()
    result = pipeline.process_file("syllabus.pdf", pdf_bytes)

    assert result.success
    assert result.document_id is not None
    assert result.metadata is not None
    assert result.metadata.category == "documents"
    assert result.metadata.original_filename == "syllabus.pdf"

    # Verify physical file on disk
    assert storage_manager.file_exists(result.metadata.storage_path)

    # Verify SQLite record
    meta_in_db = metadata_manager.get_by_id(result.document_id)
    assert meta_in_db is not None
    assert meta_in_db.sha256_hash == result.validation_result.sha256_hash


def test_pipeline_ingest_image(pipeline, storage_manager):
    png_bytes = create_sample_png()
    result = pipeline.process_file("diagram.png", png_bytes)

    assert result.success
    assert result.metadata.category == "images"
    assert storage_manager.file_exists(result.metadata.storage_path)


def test_pipeline_ingest_video(pipeline, storage_manager):
    mp4_bytes = create_sample_mp4()
    result = pipeline.process_file("lecture.mp4", mp4_bytes)

    assert result.success
    assert result.metadata.category == "videos"
    assert storage_manager.file_exists(result.metadata.storage_path)


def test_pipeline_duplicate_detection(pipeline):
    txt_bytes = create_sample_txt()

    # First upload: should succeed
    res1 = pipeline.process_file("notes.txt", txt_bytes)
    assert res1.success
    assert res1.document_id is not None

    # Second upload with identical content (even with a different name): should detect duplicate
    res2 = pipeline.process_file("notes_renamed.txt", txt_bytes)
    assert not res2.success
    assert res2.validation_result.is_duplicate
    assert res2.validation_result.status == ValidationStatus.DUPLICATE
    assert res2.validation_result.existing_document_id == res1.document_id
    assert "duplicate" in res2.message.lower()


def test_pipeline_reject_corrupted_file(pipeline):
    corrupted_pdf = b"NOT_A_REAL_PDF"
    res = pipeline.process_file("corrupted.pdf", corrupted_pdf)

    assert not res.success
    assert not res.validation_result.is_valid
    assert "corrupted" in res.message.lower() or "invalid" in res.message.lower()


def test_pipeline_reject_empty_file(pipeline):
    res = pipeline.process_file("empty.docx", b"")

    assert not res.success
    assert "empty" in res.message.lower()


def test_pipeline_reject_unsupported_extension(pipeline):
    res = pipeline.process_file("script.py", b"print('hello world')")

    assert not res.success
    assert "not supported" in res.message.lower()


def test_pipeline_batch_processing(pipeline):
    files = [
        ("doc1.pdf", create_sample_pdf()),
        ("img1.png", create_sample_png()),
        ("notes.txt", create_sample_txt()),
        ("empty.txt", b""),
        ("bad.pdf", b"CORRUPTED"),
    ]

    results = pipeline.process_batch(files)
    assert len(results) == 5

    # First three succeed
    assert results[0].success
    assert results[1].success
    assert results[2].success

    # Last two fail validation
    assert not results[3].success
    assert not results[4].success


def test_pipeline_prevents_file_overwrite(pipeline, storage_manager):
    # Two different files with identical original filename
    content1 = create_sample_txt()
    content2 = "Different content for document 2".encode("utf-8")

    res1 = pipeline.process_file("document.txt", content1)
    res2 = pipeline.process_file("document.txt", content2)

    assert res1.success
    assert res2.success
    assert res1.document_id != res2.document_id
    assert res1.metadata.storage_path != res2.metadata.storage_path

    # Both files must coexist on disk without overwriting
    assert storage_manager.file_exists(res1.metadata.storage_path)
    assert storage_manager.file_exists(res2.metadata.storage_path)
    assert storage_manager.read_file(res1.metadata.storage_path) == content1
    assert storage_manager.read_file(res2.metadata.storage_path) == content2
