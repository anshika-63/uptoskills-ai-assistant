"""
Unit tests for Multimodal OCR and Scanned PDF Detection (Phase 5).
Tests OCR engine resolution, custom OCR engine registration, image extraction,
scanned page/PDF heuristic detection, and PDF OCR fallback behavior.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.ingestion.text_extractor import (
    extract_image,
    extract_pdf,
    extract_text,
    get_tesseract_cmd,
    is_ocr_available,
    is_scanned_page,
    is_scanned_pdf,
    reset_ocr_engine,
    run_ocr,
    set_ocr_engine,
)
from backend.tests.conftest import (
    create_sample_png,
    create_sample_pdf,
    create_scanned_pdf,
)


@pytest.fixture(autouse=True)
def cleanup_ocr_engine():
    """Ensure custom OCR engine is cleaned up before and after each test."""
    reset_ocr_engine()
    yield
    reset_ocr_engine()


def test_ocr_availability_and_custom_engine():
    """Registering a custom OCR engine marks OCR as available and executes run_ocr."""
    assert reset_ocr_engine() is None
    # Register mock OCR
    set_ocr_engine(lambda img: "Mocked OCR Text Recognition")
    assert is_ocr_available() is True

    # Test run_ocr with registered engine
    text = run_ocr(create_sample_png())
    assert text == "Mocked OCR Text Recognition"


def test_extract_image_with_custom_ocr(tmp_path):
    """extract_image returns unified schema when OCR extracts text from an image."""
    img_bytes = create_sample_png()
    img_path = tmp_path / "diagram.png"
    img_path.write_bytes(img_bytes)

    set_ocr_engine(lambda img: "Diagram showing neural network architecture.")

    result = extract_image(img_path)

    assert result["file_type"] == "image"
    assert result["category"] == "images"
    assert "neural network" in result["raw_text"]
    assert result["char_count"] == len(result["raw_text"])
    assert result["word_count"] > 0
    assert result["extraction_method"] == "custom_ocr"
    assert result["error"] is None


def test_extract_image_ocr_unavailable(tmp_path):
    """extract_image returns ocr_unavailable status gracefully when OCR is not present."""
    img_bytes = create_sample_png()
    img_path = tmp_path / "photo.png"
    img_path.write_bytes(img_bytes)

    # Force is_ocr_available to False
    with patch("backend.ingestion.text_extractor.is_ocr_available", return_value=False):
        result = extract_image(img_path)

    assert result["file_type"] == "image"
    assert result["extraction_method"] == "ocr_unavailable"
    assert result["raw_text"] == ""
    assert result["char_count"] == 0


def test_extract_image_missing_file_raises():
    """extract_image raises FileNotFoundError for nonexistent paths."""
    with pytest.raises(FileNotFoundError):
        extract_image("nonexistent_path/image.png")


def test_extract_image_corrupted_file(tmp_path):
    """extract_image handles corrupted non-image bytes gracefully with failed status."""
    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"INVALID_IMAGE_BYTES_XYZ")

    set_ocr_engine(lambda img: "should not be called")
    result = extract_image(corrupt_path)

    assert result["file_type"] == "image"
    assert result["extraction_method"] == "failed"
    assert result["error"] is not None


def test_is_scanned_page_detection():
    """Detects pages with zero text and raster images as scanned pages."""
    import pymupdf
    # 1. Text PDF page
    doc_text = pymupdf.open()
    p_text = doc_text.new_page()
    p_text.insert_text(pymupdf.Point(50, 50), "This is real digital text inside a PDF.")
    assert is_scanned_page(p_text) is False

    # 2. Scanned PDF page (image only, 0 digital text)
    doc_scan = pymupdf.open()
    p_scan = doc_scan.new_page(width=100, height=100)
    p_scan.insert_image(pymupdf.Rect(0, 0, 100, 100), stream=create_sample_png())
    assert is_scanned_page(p_scan) is True


def test_is_scanned_pdf_detection(tmp_path):
    """is_scanned_pdf accurately detects scanned PDFs and normal text PDFs."""
    # Normal digital PDF
    pdf_text_path = tmp_path / "digital.pdf"
    pdf_text_path.write_bytes(create_sample_pdf())
    # create_sample_pdf has no raster images
    assert is_scanned_pdf(pdf_text_path) is False

    # Scanned image-only PDF
    pdf_scan_path = tmp_path / "scanned.pdf"
    pdf_scan_path.write_bytes(create_scanned_pdf())
    assert is_scanned_pdf(pdf_scan_path) is True


def test_extract_pdf_ocr_fallback(tmp_path):
    """extract_pdf detects scanned page, applies OCR fallback, and records is_scanned flag."""
    pdf_path = tmp_path / "scanned_doc.pdf"
    pdf_path.write_bytes(create_scanned_pdf())

    # Register custom OCR engine
    set_ocr_engine(lambda img: "Extracted text from scanned receipt via OCR.")

    result = extract_pdf(pdf_path)

    assert result["file_type"] == "pdf"
    assert result["is_scanned"] is True
    assert result["ocr_applied"] is True
    assert "scanned receipt" in result["raw_text"]
    assert len(result["pages"]) == 1
    page0 = result["pages"][0]
    assert page0["extraction_method"] == "ocr_fallback"
    assert page0["is_scanned"] is True


def test_extract_pdf_ocr_unavailable(tmp_path):
    """When OCR is unavailable, extract_pdf gracefully notes ocr_unavailable on scanned pages."""
    pdf_path = tmp_path / "scanned_doc.pdf"
    pdf_path.write_bytes(create_scanned_pdf())

    with patch("backend.ingestion.text_extractor.is_ocr_available", return_value=False):
        result = extract_pdf(pdf_path)

    assert result["is_scanned"] is True
    assert result["ocr_applied"] is False
    assert result["pages"][0]["extraction_method"] == "ocr_unavailable"
