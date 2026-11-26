"""
Phase 5 Multimodal End-to-End Integration Tests.
Verifies the complete multimodal ingestion pipeline:
- Upload -> Validation -> Categorized Storage -> Multimodal Extraction -> Preprocessing -> Timestamped Chunking -> Embeddings -> Persistent ChromaDB Indexing.
- Covers Image OCR, Scanned PDF OCR fallback, Video audio extraction + transcription,
  timestamp and page preservation, duplicate handling, and cascading deletion.
"""

from unittest.mock import MagicMock, patch
import pytest

from backend.ingestion.text_extractor import set_ocr_engine, reset_ocr_engine
from backend.tests.conftest import (
    create_real_video_mp4,
    create_sample_png,
    create_scanned_pdf,
)


@pytest.fixture(autouse=True)
def cleanup_ocr():
    """Reset OCR engine before and after each test."""
    reset_ocr_engine()
    yield
    reset_ocr_engine()


def test_e2e_image_ocr_ingestion_and_vector_indexing(multimodal_pipeline):
    """
    Uploading an image executes OCR, preprocessed chunking, embedding generation,
    and ChromaDB vector indexing.
    """
    # Register mock OCR text recognition
    set_ocr_engine(lambda img: "Flowchart showing user authentication and authorization workflow.")

    png_bytes = create_sample_png()
    result = multimodal_pipeline.process_file("architecture_diagram.png", png_bytes)

    assert result.success is True
    assert result.upload_success is True
    assert result.extraction_status == "success"
    assert result.embedding_status == "indexed"
    assert result.vector_count > 0

    doc_id = result.document_id

    # 1. Verify SQLite metadata record
    meta = multimodal_pipeline.metadata.get_by_id(doc_id)
    assert meta is not None
    assert meta.status == "indexed"

    # 2. Verify SQLite document_chunks table
    chunks = multimodal_pipeline.metadata.get_chunks(doc_id)
    assert len(chunks) == result.vector_count
    for c in chunks:
        assert c["category"] == "images"
        assert c["page_number"] == 1
        assert "authentication" in c["text"]

    # 3. Verify ChromaDB vector store
    chroma_chunks = multimodal_pipeline.vector_store.get_document_chunks(doc_id)
    assert len(chroma_chunks) == result.vector_count


def test_e2e_scanned_pdf_ocr_fallback_and_indexing(multimodal_pipeline):
    """
    Uploading a scanned (image-only) PDF detects scanned pages, triggers OCR fallback,
    chunks text with is_scanned=True metadata, and indexes in ChromaDB.
    """
    set_ocr_engine(lambda img: "Scanned textbook chapter covering algorithmic complexity and Big-O notation.")

    pdf_bytes = create_scanned_pdf()
    result = multimodal_pipeline.process_file("scanned_textbook.pdf", pdf_bytes)

    assert result.success is True
    assert result.extraction_status == "success"
    assert result.embedding_status == "indexed"
    assert result.vector_count > 0

    doc_id = result.document_id

    # Verify extraction record noted scanned PDF and OCR fallback
    ext = multimodal_pipeline.metadata.get_extraction(doc_id)
    assert ext is not None
    assert ext["file_type"] == "pdf"

    # Verify SQLite chunks retain page number and is_scanned flag
    chunks = multimodal_pipeline.metadata.get_chunks(doc_id)
    assert len(chunks) > 0
    assert chunks[0]["page_number"] == 1
    assert chunks[0].get("is_scanned") is True
    assert "algorithmic complexity" in chunks[0]["text"]


def test_e2e_video_ingestion_transcription_and_vector_indexing(multimodal_pipeline):
    """
    Uploading a video extracts the audio track with FFmpeg, transcribes it with Whisper,
    generates chunks preserving timestamp bounds, and indexes vectors into ChromaDB.
    """
    video_bytes = create_real_video_mp4(duration_s=1.0)

    # Mock Whisper transcription engine output
    mock_seg1 = MagicMock()
    mock_seg1.id = 0
    mock_seg1.start = 0.0
    mock_seg1.end = 0.5
    mock_seg1.text = "In this section, we discuss vector databases and similarity search."

    mock_seg2 = MagicMock()
    mock_seg2.id = 1
    mock_seg2.start = 0.5
    mock_seg2.end = 1.0
    mock_seg2.text = "Cosine distance metric enables fast semantic retrieval of knowledge chunks."

    mock_info = MagicMock()
    mock_info.duration = 1.0
    mock_info.language = "en"
    mock_info.language_probability = 0.99

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_seg1, mock_seg2], mock_info)

    with patch("backend.ingestion.audio_transcriber.WhisperModelManager.get_instance") as mock_mgr:
        inst = MagicMock()
        inst.get_model.return_value = mock_model
        mock_mgr.return_value = inst

        result = multimodal_pipeline.process_file("lecture_video.mp4", video_bytes)

    assert result.success is True
    assert result.upload_success is True
    assert result.extraction_status == "success"
    assert result.embedding_status == "indexed"
    assert result.vector_count > 0

    doc_id = result.document_id

    # 1. Verify transcript JSON is persisted
    disk_transcript = multimodal_pipeline.storage.get_transcript_json(doc_id)
    assert disk_transcript is not None
    assert disk_transcript["category"] == "videos"
    assert len(disk_transcript["segments"]) == 2

    # 2. Verify chunks preserve video timestamps
    chunks = multimodal_pipeline.metadata.get_chunks(doc_id)
    assert len(chunks) > 0
    first_chunk = chunks[0]
    assert "start_time" in first_chunk
    assert "end_time" in first_chunk
    assert "timestamp" in first_chunk
    assert first_chunk["category"] == "videos"

    # 3. Verify ChromaDB indexing and timestamp metadata preservation
    chroma_chunks = multimodal_pipeline.vector_store.get_document_chunks(doc_id)
    assert len(chroma_chunks) == result.vector_count
    assert "timestamp" in chroma_chunks[0]


def test_multimodal_semantic_search_with_timestamps(multimodal_pipeline):
    """
    Semantic search across multimodal vectors returns matching chunks with media timestamps.
    """
    video_bytes = create_real_video_mp4(duration_s=0.5)

    mock_seg = MagicMock()
    mock_seg.id = 0
    mock_seg.start = 12.0
    mock_seg.end = 25.0
    mock_seg.text = "Deep learning backpropagation and gradient descent optimization algorithms."

    mock_info = MagicMock()
    mock_info.duration = 25.0
    mock_info.language = "en"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_seg], mock_info)

    with patch("backend.ingestion.audio_transcriber.WhisperModelManager.get_instance") as mock_mgr:
        inst = MagicMock()
        inst.get_model.return_value = mock_model
        mock_mgr.return_value = inst

        res = multimodal_pipeline.process_file("deep_learning_clip.mp4", video_bytes)

    # Search for gradient descent
    query = "gradient descent and optimization algorithms"
    hits = multimodal_pipeline.vector_store.query_by_text(query=query, top_k=1)

    assert len(hits) == 1
    top_hit = hits[0]
    assert top_hit["document_id"] == res.document_id
    assert "start_time" in top_hit["metadata"]
    assert "timestamp" in top_hit["metadata"]
    assert top_hit["metadata"]["timestamp"] == "[00:12.00 -> 00:25.00]"


def test_multimodal_duplicate_detection(multimodal_pipeline):
    """Duplicate video upload retrieves existing transcript and vectors without creating duplicates."""
    video_bytes = create_real_video_mp4(duration_s=0.5)

    mock_seg = MagicMock()
    mock_seg.id = 0
    mock_seg.start = 0.0
    mock_seg.end = 0.5
    mock_seg.text = "Deduplication test for video transcription pipeline."

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_seg], MagicMock(duration=0.5, language="en"))

    with patch("backend.ingestion.audio_transcriber.WhisperModelManager.get_instance") as mock_mgr:
        inst = MagicMock()
        inst.get_model.return_value = mock_model
        mock_mgr.return_value = inst

        res1 = multimodal_pipeline.process_file("first_video.mp4", video_bytes)
        assert res1.success is True
        assert res1.embedding_status == "indexed"
        vectors_before = multimodal_pipeline.vector_store.count_vectors()

        # Second upload with identical bytes
        res2 = multimodal_pipeline.process_file("second_video.mp4", video_bytes)
        assert res2.success is False
        assert res2.validation_result.is_duplicate is True
        assert res2.embedding_status == "retrieved"
        assert res2.vector_count == res1.vector_count

        # Total vectors should not have increased
        vectors_after = multimodal_pipeline.vector_store.count_vectors()
        assert vectors_after == vectors_before


def test_multimodal_cascading_deletion(multimodal_pipeline):
    """Deleting a multimodal document purges physical file, transcript JSON, SQLite, and ChromaDB."""
    video_bytes = create_real_video_mp4(duration_s=0.5)

    mock_seg = MagicMock()
    mock_seg.id = 0
    mock_seg.start = 0.0
    mock_seg.end = 0.5
    mock_seg.text = "Content scheduled for cascading deletion test."

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_seg], MagicMock(duration=0.5, language="en"))

    with patch("backend.ingestion.audio_transcriber.WhisperModelManager.get_instance") as mock_mgr:
        inst = MagicMock()
        inst.get_model.return_value = mock_model
        mock_mgr.return_value = inst

        res = multimodal_pipeline.process_file("video_to_delete.mp4", video_bytes)

    doc_id = res.document_id

    # Verify presence across all layers
    assert multimodal_pipeline.storage.file_exists(res.metadata.storage_path)
    assert multimodal_pipeline.storage.get_transcript_json(doc_id) is not None
    assert multimodal_pipeline.metadata.get_by_id(doc_id) is not None
    assert len(multimodal_pipeline.metadata.get_chunks(doc_id)) > 0
    assert multimodal_pipeline.vector_store.count_by_document(doc_id) > 0

    # Delete
    deleted = multimodal_pipeline.delete_document(doc_id)
    assert deleted is True

    # Verify absence across all layers
    assert not multimodal_pipeline.storage.file_exists(res.metadata.storage_path)
    assert multimodal_pipeline.storage.get_transcript_json(doc_id) is None
    assert multimodal_pipeline.metadata.get_by_id(doc_id) is None
    assert multimodal_pipeline.metadata.get_extraction(doc_id) is None
    assert len(multimodal_pipeline.metadata.get_chunks(doc_id)) == 0
    assert multimodal_pipeline.vector_store.count_by_document(doc_id) == 0
