"""
Regression and Integration Tests for Tesseract OCR Integration in UpToSkills RAG Pipeline.
Verifies .env loading, Tesseract path resolution across working directories,
pytesseract command configuration, real image OCR extraction, error logging,
and downstream routing through preprocessing, chunking, embeddings, and ChromaDB.
"""

import io
import logging
import os
from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image, ImageDraw

from backend.config import (
    BACKEND_DIR,
    PROJECT_ROOT,
    TESSERACT_CMD,
    load_environment,
    resolve_tesseract_cmd,
)
from backend.ingestion.text_extractor import (
    configure_tesseract,
    extract_image,
    get_tesseract_cmd,
    is_ocr_available,
    reset_ocr_engine,
    run_ocr,
)
from backend.ingestion.pipeline import IngestionPipeline


@pytest.fixture(autouse=True)
def reset_ocr():
    """Ensure custom OCR engine is cleaned up before and after each test."""
    reset_ocr_engine()
    yield
    reset_ocr_engine()


def create_ocr_test_png(text: str = "UPTOSKILLS OCR TEST 2026") -> bytes:
    """Helper to generate a clean, readable PNG with given text for OCR verification."""
    img = Image.new("RGB", (450, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 35), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_env_loading_and_tesseract_path_resolution():
    """
    Verifies that load_environment discovers .env and resolve_tesseract_cmd resolves
    the executable path to tesseract.exe regardless of working directory.
    """
    loaded = load_environment()
    assert len(loaded) > 0, "Expected at least one .env file to be discovered"

    resolved = resolve_tesseract_cmd()
    assert resolved != "", "Expected resolve_tesseract_cmd to resolve a non-empty path"
    assert Path(resolved).exists(), f"Resolved path '{resolved}' does not exist on disk"
    assert resolved.lower().endswith("tesseract.exe")


def test_tesseract_configuration_and_pytesseract_sync():
    """
    Verifies configure_tesseract synchronizes pytesseract.pytesseract.tesseract_cmd
    and is_ocr_available returns True for the live binary.
    """
    import pytesseract

    success = configure_tesseract()
    assert success is True
    assert is_ocr_available() is True

    resolved = get_tesseract_cmd()
    assert resolved is not None
    assert pytesseract.pytesseract.tesseract_cmd == str(resolved)

    version = pytesseract.get_tesseract_version()
    assert str(version).startswith("5.")


def test_real_image_ocr_extraction(tmp_path):
    """
    Tests extract_image using an actual PNG containing text and the live Tesseract binary.
    Verifies raw_text, char_count, word_count, and extraction_method.
    """
    test_phrase = "UPTOSKILLS OCR VERIFICATION"
    img_bytes = create_ocr_test_png(test_phrase)
    img_path = tmp_path / "actual_test.png"
    img_path.write_bytes(img_bytes)

    result = extract_image(img_path)

    assert result["file_type"] == "image"
    assert result["category"] == "images"
    assert result["extraction_method"] == "tesseract_ocr"
    assert result["error"] is None
    assert result["char_count"] > 0
    assert result["word_count"] > 0
    # OCR recognition check
    raw_upper = result["raw_text"].upper()
    assert "UPTOSKILLS" in raw_upper or "OCR" in raw_upper


def test_invalid_tesseract_path_logging_and_degradation(tmp_path, caplog):
    """
    Verifies that configuring an invalid Tesseract path logs an appropriate warning/error
    and extract_image degrades gracefully to ocr_unavailable with informative message.
    """
    img_bytes = create_ocr_test_png("DUMMY")
    img_path = tmp_path / "dummy.png"
    img_path.write_bytes(img_bytes)

    invalid_cmd = str(tmp_path / "missing_tesseract.exe")

    with caplog.at_level(logging.WARNING):
        configured = configure_tesseract(custom_cmd=invalid_cmd)
        assert configured is False
        assert any("Tesseract binary not found" in r.message or "executable not found" in r.message for r in caplog.records)

    with patch("backend.ingestion.text_extractor.get_tesseract_cmd", return_value=None):
        result = extract_image(img_path)
        assert result["extraction_method"] == "ocr_unavailable"
        assert result["raw_text"] == ""
        assert "Tesseract OCR engine is not available" in result["error"]


def test_end_to_end_image_pipeline_chromadb(multimodal_pipeline):
    """
    Verifies that an image uploaded to IngestionPipeline:
    1. Extracts text using real Tesseract OCR.
    2. Builds preprocessed chunks with page_number=1 and media_type='image'.
    3. Generates embeddings using sentence-transformers.
    4. Indexes chunks into ChromaDB.
    5. Retrieves the chunk via ChromaDB similarity query.
    """
    img_text = "Artificial Intelligence in Medical Healthcare Diagnosis"
    img_bytes = create_ocr_test_png(img_text)
    filename = "medical_ai_research.png"

    result = multimodal_pipeline.process_file(filename, img_bytes)

    assert result.success is True
    assert result.extraction_status == "success"
    assert result.extracted_content is not None
    assert result.extracted_content["extraction_method"] == "tesseract_ocr"
    assert result.extracted_content["char_count"] > 0

    # Chunks verification
    assert result.preprocessed_content is not None
    assert len(result.preprocessed_content) >= 1
    first_chunk = result.preprocessed_content[0]
    assert first_chunk.get("page_number") == 1
    assert first_chunk.get("media_type") == "image"

    # Vector store verification
    assert result.embedding_status == "indexed"
    assert result.vector_count >= 1

    # Similarity query against ChromaDB
    query_results = multimodal_pipeline.vector_store.query_by_text(
        "Medical Healthcare Diagnosis",
        top_k=1,
    )
    assert len(query_results) >= 1
    retrieved_chunk = query_results[0]
    assert retrieved_chunk["document_id"] == result.document_id
    assert "metadata" in retrieved_chunk
    assert retrieved_chunk["metadata"]["category"] == "images"
