"""
Integration tests for the complete IngestionPipeline.
Verifies the end-to-end flow: upload, validation, duplicate detection,
persistent categorized storage, metadata recording, and text extraction persistence.
"""

import json

from backend.ingestion.models import ValidationStatus
from backend.tests.conftest import (
    create_real_docx,
    create_sample_docx,
    create_sample_jpeg,
    create_sample_mp3,
    create_sample_mp4,
    create_sample_pdf,
    create_sample_png,
    create_sample_txt,
    create_sample_wav,
)


# ─────────────────────────────────────────────
# Original Phase 1 integration tests (preserved)
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# New extraction persistence tests
# ─────────────────────────────────────────────

def test_new_upload_extraction_persisted_pdf(pipeline, metadata_manager):
    """Uploading a new PDF must persist extraction data to document_extraction table."""
    pdf_bytes = create_sample_pdf()
    result = pipeline.process_file("lecture_notes.pdf", pdf_bytes)

    assert result.success
    assert result.upload_success
    assert result.extraction_status == "success"
    assert result.extracted_content is not None
    assert result.extracted_content["file_type"] == "pdf"

    # Verify extraction record is in SQLite
    assert metadata_manager.extraction_exists(result.document_id)
    stored = metadata_manager.get_extraction(result.document_id)
    assert stored is not None
    assert stored["file_type"] == "pdf"
    assert "raw_text" in stored


def test_new_upload_extraction_persisted_txt(pipeline, metadata_manager):
    """Uploading a TXT must persist extraction with correct word/char counts."""
    txt_bytes = create_sample_txt()
    result = pipeline.process_file("readme.txt", txt_bytes)

    assert result.success
    assert result.extraction_status == "success"
    assert result.extracted_content is not None
    assert "raw_text" in result.extracted_content

    stored = metadata_manager.get_extraction(result.document_id)
    assert stored is not None
    assert stored["word_count"] > 0
    assert stored["char_count"] == len(stored["raw_text"])


def test_new_upload_extraction_persisted_real_docx(pipeline, metadata_manager):
    """Uploading a real DOCX must persist paragraphs and 3-D tables."""
    paragraphs = ["Introduction", "Methods"]
    tables = [["Header", "Value"], ["Row1A", "Row1B"]]
    docx_bytes = create_real_docx(paragraphs=paragraphs, tables=[tables])
    result = pipeline.process_file("report.docx", docx_bytes)

    assert result.success
    assert result.extraction_status == "success"
    assert result.extracted_content is not None
    assert "Introduction" in result.extracted_content["paragraphs"]

    # Verify tables are stored as 3-D list (not flat rows)
    stored = metadata_manager.get_extraction(result.document_id)
    assert stored is not None
    tbl = stored["tables"]
    assert isinstance(tbl, list)
    assert len(tbl) == 1          # one table
    assert isinstance(tbl[0], list)   # table is list of rows
    assert isinstance(tbl[0][0], list)  # row is list of cells


def test_extraction_not_applicable_for_images(pipeline, metadata_manager):
    """Images must have extraction_status='not_applicable' and no extraction record."""
    png_bytes = create_sample_png()
    result = pipeline.process_file("photo.png", png_bytes)

    assert result.success
    assert result.extraction_status == "not_applicable"
    assert result.extracted_content is None
    assert not metadata_manager.extraction_exists(result.document_id)


def test_extraction_not_applicable_for_videos(pipeline, metadata_manager):
    """Videos must have extraction_status='not_applicable' and no extraction record."""
    mp4_bytes = create_sample_mp4()
    result = pipeline.process_file("video.mp4", mp4_bytes)

    assert result.success
    assert result.extraction_status == "not_applicable"
    assert result.extracted_content is None


def test_duplicate_upload_retrieves_existing_extraction(pipeline, metadata_manager):
    """
    Re-uploading an identical file (duplicate) must return the already-persisted
    extraction data instead of None, and report extraction_status='retrieved'.
    """
    txt_bytes = create_sample_txt()

    # First upload — stores file and persists extraction
    res1 = pipeline.process_file("notes.txt", txt_bytes)
    assert res1.success
    assert res1.extraction_status == "success"
    orig_extraction = res1.extracted_content
    assert orig_extraction is not None

    # Second upload — identical content, different name
    res2 = pipeline.process_file("notes_v2.txt", txt_bytes)
    assert not res2.success                          # duplicate → not stored again
    assert res2.validation_result.is_duplicate
    assert res2.extraction_status == "retrieved"      # retrieved from DB
    assert res2.extracted_content is not None         # NOT None — has content
    assert res2.extracted_content["raw_text"] == orig_extraction["raw_text"]
    assert res2.extracted_content["file_type"] == "txt"


def test_duplicate_upload_message_contains_extraction_info(pipeline):
    """Duplicate upload message must indicate existing extraction was returned."""
    pdf_bytes = create_sample_pdf()
    pipeline.process_file("doc.pdf", pdf_bytes)

    res2 = pipeline.process_file("doc_copy.pdf", pdf_bytes)
    assert res2.validation_result.is_duplicate
    assert "existing extraction" in res2.message.lower() or "extraction" in res2.message.lower()


def test_get_or_reextract_returns_cached_extraction(pipeline, metadata_manager):
    """get_or_reextract must return existing DB record on the first call."""
    txt_bytes = create_sample_txt()
    result = pipeline.process_file("cached.txt", txt_bytes)
    assert result.extraction_status == "success"

    # Call get_or_reextract — must hit DB cache
    retrieved, error = pipeline.get_or_reextract(result.document_id)
    assert error is None
    assert retrieved is not None
    assert retrieved["file_type"] == "txt"
    assert retrieved["raw_text"] == result.extracted_content["raw_text"]


def test_get_or_reextract_reruns_when_db_record_missing(pipeline, metadata_manager):
    """
    get_or_reextract must run extraction from the stored file when no
    extraction record exists in the DB (simulates missing record scenario).
    """
    txt_bytes = create_sample_txt()
    result = pipeline.process_file("missing_ext.txt", txt_bytes)
    assert result.extraction_status == "success"

    # Manually delete the extraction record to simulate missing data
    metadata_manager.delete_extraction(result.document_id)
    assert not metadata_manager.extraction_exists(result.document_id)

    # get_or_reextract must re-extract and re-persist
    retrieved, error = pipeline.get_or_reextract(result.document_id)
    assert error is None
    assert retrieved is not None
    assert retrieved["file_type"] == "txt"
    # Verify the record is now back in DB
    assert metadata_manager.extraction_exists(result.document_id)


def test_extraction_stats_recorded(pipeline, metadata_manager):
    """Extraction stats (char_count, word_count) must be accessible via get_extraction_stats."""
    txt_bytes = create_sample_txt()
    result = pipeline.process_file("stats_test.txt", txt_bytes)

    stats = metadata_manager.get_extraction_stats(result.document_id)
    assert stats is not None
    assert stats["char_count"] > 0
    assert stats["word_count"] > 0
    assert stats["extraction_status"] == "success"
    assert "extracted_at" in stats


def test_upload_success_flag_true_for_new_files(pipeline):
    """upload_success must be True only for genuinely new files."""
    txt_bytes = create_sample_txt()
    res1 = pipeline.process_file("new_file.txt", txt_bytes)
    assert res1.upload_success is True

    # Duplicate: upload_success must be False
    res2 = pipeline.process_file("new_file_copy.txt", txt_bytes)
    assert res2.upload_success is False


def test_upload_success_false_for_invalid_files(pipeline):
    """upload_success must be False for validation failures."""
    res = pipeline.process_file("bad.pdf", b"NOT_PDF")
    assert res.upload_success is False
    assert res.success is False


# ─────────────────────────────────────────────
# Audio Transcription & Persistence Tests
# ─────────────────────────────────────────────

def test_pipeline_ingest_audio_wav(pipeline, storage_manager, metadata_manager):
    """
    Ingesting a valid WAV audio file must:
    1. Store audio file in uploads/audio/
    2. Transcribe via faster-whisper with timestamps and segments
    3. Persist transcript JSON to storage/transcripts/{document_id}.json
    4. Store extraction and metadata records in SQLite with status='transcribed'
    """
    wav_bytes = create_sample_wav(duration_s=1.0, freq=440.0, sample_rate=16000)
    result = pipeline.process_file("speech.wav", wav_bytes)

    assert result.success
    assert result.upload_success
    assert result.document_id is not None
    assert result.metadata is not None
    assert result.metadata.category == "audio"
    assert result.metadata.file_type == ".wav"
    assert result.extraction_status == "success"
    assert result.extracted_content is not None
    assert result.extracted_content["category"] == "audio"
    assert "segments" in result.extracted_content
    assert "duration" in result.extracted_content

    # 1. Verify physical audio file in audio/ storage
    assert storage_manager.file_exists(result.metadata.storage_path)
    assert "audio" in result.metadata.storage_path

    # 2. Verify transcript JSON on disk
    transcript_json_path = storage_manager.transcripts_dir / f"{result.document_id}.json"
    assert transcript_json_path.exists()
    disk_data = storage_manager.get_transcript_json(result.document_id)
    assert disk_data is not None
    assert disk_data["category"] == "audio"

    # 3. Verify SQLite metadata and extraction table
    meta_row = metadata_manager.get_by_id(result.document_id)
    assert meta_row is not None
    assert meta_row.status == "transcribed"
    meta_extra = json.loads(meta_row.metadata_json)
    assert meta_extra.get("transcription_status") == "success"
    assert "transcript_path" in meta_extra

    assert metadata_manager.extraction_exists(result.document_id)
    db_ext = metadata_manager.get_extraction(result.document_id)
    assert db_ext is not None
    assert db_ext["category"] == "audio"


def test_pipeline_duplicate_audio_retrieves_transcript(pipeline, storage_manager):
    """
    Re-uploading an identical audio file must detect the SHA-256 duplicate,
    prevent duplicate file storage, and retrieve the existing transcript.
    """
    wav_bytes = create_sample_wav(duration_s=0.5, freq=500.0)

    # First upload
    res1 = pipeline.process_file("meeting.wav", wav_bytes)
    assert res1.success
    assert res1.upload_success
    assert res1.extraction_status == "success"
    orig_transcript = res1.extracted_content
    assert orig_transcript is not None

    # Second upload with different filename
    res2 = pipeline.process_file("meeting_copy.wav", wav_bytes)
    assert not res2.success
    assert res2.validation_result.is_duplicate
    assert res2.validation_result.existing_document_id == res1.document_id
    assert res2.extraction_status == "retrieved"
    assert res2.extracted_content is not None
    assert res2.extracted_content["category"] == "audio"
    assert res2.extracted_content["duration"] == orig_transcript["duration"]


def test_pipeline_audio_missing_extraction_triggers_retranscription(pipeline, metadata_manager, storage_manager):
    """
    When extraction record and transcript JSON are missing for an ingested audio file,
    get_or_reextract must re-transcribe from the stored audio file and persist.
    """
    wav_bytes = create_sample_wav(duration_s=0.5, freq=300.0)
    result = pipeline.process_file("podcast.wav", wav_bytes)
    assert result.extraction_status == "success"

    # Manually delete SQLite extraction record and transcript JSON file
    metadata_manager.delete_extraction(result.document_id)
    transcript_file = storage_manager.transcripts_dir / f"{result.document_id}.json"
    if transcript_file.exists():
        transcript_file.unlink()

    assert not metadata_manager.extraction_exists(result.document_id)
    assert not transcript_file.exists()

    # get_or_reextract must re-transcribe from the stored audio file
    reextracted, error = pipeline.get_or_reextract(result.document_id)
    assert error is None
    assert reextracted is not None
    assert reextracted["category"] == "audio"

    # Verify both DB and disk transcript are re-saved
    assert metadata_manager.extraction_exists(result.document_id)
    assert transcript_file.exists()


def test_pipeline_reject_corrupted_audio(pipeline):
    """Corrupted audio files must fail validation without running Whisper."""
    bad_mp3 = b"NOT_A_VALID_MP3_STREAM"
    res = pipeline.process_file("corrupted.mp3", bad_mp3)

    assert not res.success
    assert not res.upload_success
    assert "corrupted mp3" in res.message.lower()


