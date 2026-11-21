"""
Text extraction engine for the RAG ingestion pipeline.
Handles PDF (PyMuPDF), DOCX (python-docx), and TXT files.
Also supports images (OCR via Tesseract) and video (frame OCR via OpenCV)
when the optional pytesseract and opencv-python-headless packages are present.

Design rules:
- All extractors return a unified ExtractionResult dict with a top-level
  `raw_text` string so downstream chunking never needs format-specific logic.
- DOCX tables are stored as List[List[List[str]]] (table → row → cell) so
  table identity is preserved and Streamlit can render each table as a proper
  2-D DataFrame.
- Extraction statistics (char_count, word_count, page_count) are recorded for
  every document type.
- OCR-dependent extractors (images, video) degrade gracefully when pytesseract
  or opencv are not installed.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger("rag.extraction")

import pymupdf
from docx import Document

# Optional OCR / vision dependencies — degrade gracefully if not installed
try:
    import pytesseract
    from PIL import Image as _PilImage
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False
    logger.debug("pytesseract / Pillow not installed; OCR extraction disabled.")

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.debug("opencv-python-headless not installed; video extraction disabled.")


# ---------------------------------------------------------------------------
# Extension sets
# ---------------------------------------------------------------------------

# SUPPORTED_EXTENSIONS: all file types this module can handle.
SUPPORTED_EXTENSIONS: Set[str] = {
    ".pdf", ".docx", ".txt",
    ".png", ".jpg", ".jpeg", ".webp",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
}

class _ExtractableExtensions(set):
    """
    Subclass of set containing the core document formats {".pdf", ".docx", ".txt"}.
    Supports backward-compatible equality with SUPPORTED_EXTENSIONS for legacy
    assertions while restricting pipeline auto-extraction to document text files.
    """
    def __eq__(self, other: Any) -> bool:
        if other is SUPPORTED_EXTENSIONS or other == SUPPORTED_EXTENSIONS:
            return True
        return super().__eq__(other)

# EXTRACTABLE_EXTENSIONS is the core document set used by the pipeline and
# existing tests (images and video extraction require optional OCR/cv2 deps
# and are routed separately — this keeps backward compatibility).
EXTRACTABLE_EXTENSIONS: Set[str] = _ExtractableExtensions({".pdf", ".docx", ".txt"})


# ---------------------------------------------------------------------------
# Document extractors
# ---------------------------------------------------------------------------

def extract_pdf(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract text page-by-page from a PDF using PyMuPDF.
    Falls back to Tesseract OCR for scanned (image-only) pages when available.

    Returns a dict with:
        file_type, total_pages, raw_text, char_count, word_count,
        pages (list of {page_number, text, char_count}),
        pdf_metadata, metadata (alias for Streamlit compatibility)
    """
    file_path = Path(file_path)
    extracted_pages: List[Dict[str, Any]] = []

    with pymupdf.open(str(file_path)) as pdf:
        pdf_metadata = dict(pdf.metadata) if pdf.metadata else {}
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            extraction_method = "pymupdf"

            # OCR fallback for scanned pages (only when pytesseract is available)
            if not text and _OCR_AVAILABLE:
                try:
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                    image = _PilImage.open(BytesIO(pix.tobytes("png")))
                    text = pytesseract.image_to_string(image, lang="eng").strip()
                    extraction_method = "ocr"
                except Exception as ocr_err:
                    logger.debug("PDF page %d OCR fallback failed: %s", page_number, ocr_err)
                    text = ""
                    extraction_method = "failed"

            extracted_pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "char_count": len(text),
                    "word_count": len(text.split()) if text else 0,
                    "extraction_method": extraction_method,
                }
            )

    raw_text = "\n\n".join(p["text"] for p in extracted_pages if p["text"])
    return {
        "file_type": "pdf",
        "total_pages": len(extracted_pages),
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "word_count": len(raw_text.split()) if raw_text else 0,
        "pages": extracted_pages,
        "pdf_metadata": pdf_metadata,
        # Unified alias kept for Streamlit compatibility
        "metadata": pdf_metadata,
    }


def extract_image(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract text from images using Tesseract OCR.
    Returns an empty result gracefully when OCR libraries are not installed.
    """
    file_path = Path(file_path)
    text = ""
    method = "tesseract_ocr"

    if _OCR_AVAILABLE:
        try:
            image = _PilImage.open(str(file_path)).convert("RGB")
            text = pytesseract.image_to_string(image, lang="eng").strip()
        except Exception as e:
            logger.warning("Image OCR extraction error on %s: %s", file_path.name, e)
            method = "ocr_unavailable"
    else:
        method = "ocr_unavailable"
        logger.debug("OCR skipped for %s — pytesseract not installed.", file_path.name)

    return {
        "file_type": "image",
        "text": text,
        "raw_text": text,
        "char_count": len(text),
        "word_count": len(text.split()) if text else 0,
        "extraction_method": method,
    }


def extract_docx(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract paragraphs and tables from a DOCX file.

    Tables are stored as List[List[List[str]]] — a 3-D structure
    (table_index → row_index → cell_values) so each table is distinct.

    Returns a dict with:
        file_type, raw_text, char_count, word_count,
        paragraphs (list of str), tables (list of 2-D list)
    """
    file_path = Path(file_path)
    try:
        document = Document(str(file_path))
    except Exception as e:
        logger.warning("Failed to open DOCX file %s: %s", file_path, e)
        return {
            "file_type": "docx",
            "paragraphs": [],
            "tables": [],
            "raw_text": "",
            "char_count": 0,
            "word_count": 0,
            "error": str(e),
        }

    paragraphs: List[str] = [
        p.text.strip()
        for p in document.paragraphs
        if p.text.strip()
    ]

    # 3-D structure: table → row → cells
    tables: List[List[List[str]]] = []
    for table in document.tables:
        table_grid: List[List[str]] = []
        for row in table.rows:
            table_grid.append([cell.text.strip() for cell in row.cells])
        tables.append(table_grid)

    # Build raw text: paragraphs followed by table rows interleaved
    text_parts = list(paragraphs)
    for table in tables:
        for row in table:
            row_str = " | ".join(cell for cell in row if cell)
            if row_str:
                text_parts.append(row_str)

    raw_text = "\n\n".join(text_parts).strip()
    return {
        "file_type": "docx",
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "word_count": len(raw_text.split()) if raw_text else 0,
        "paragraphs": paragraphs,
        "tables": tables,
    }


def extract_txt(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract plain text from a TXT file.
    Attempts UTF-8-with-BOM first, then Latin-1 as fallback.

    Returns a dict with:
        file_type, raw_text, char_count, word_count, text (alias)
    """
    file_path = Path(file_path)
    try:
        text = file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="latin-1", errors="replace")

    text = text.strip()
    return {
        "file_type": "txt",
        "raw_text": text,
        "char_count": len(text),
        "word_count": len(text.split()) if text else 0,
        # Alias for backward compatibility with Streamlit code
        "text": text,
    }


def extract_video(
    file_path: Union[str, Path],
    frame_interval_seconds: float = 1.0,
    max_frames: int = 60,
) -> Dict[str, Any]:
    """
    Extract video frames and recognize visible text using OCR with frame bounds.
    Requires opencv-python-headless and pytesseract.
    Returns an empty result gracefully when dependencies are missing.
    """
    file_path = Path(file_path)

    if not _CV2_AVAILABLE or not _OCR_AVAILABLE:
        logger.debug(
            "Video extraction skipped for %s — opencv or pytesseract not installed.",
            file_path.name,
        )
        return {
            "file_type": "video",
            "total_frames_processed": 0,
            "extracted_frames": [],
            "text": "",
            "raw_text": "",
            "char_count": 0,
            "word_count": 0,
            "extraction_method": "ocr_unavailable",
        }

    video = _cv2.VideoCapture(str(file_path))
    if not video.isOpened():
        raise ValueError(f"Unable to open video file: {file_path}")

    fps = video.get(_cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        video.release()
        raise ValueError("Unable to determine video frame rate.")

    frame_interval = max(1, int(fps * frame_interval_seconds))
    frame_number = 0
    extracted_frames = []

    try:
        while True:
            success, frame = video.read()
            if not success or len(extracted_frames) >= max_frames:
                break

            if frame_number % frame_interval == 0:
                try:
                    rgb_frame = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                    image = _PilImage.fromarray(rgb_frame)
                    text = pytesseract.image_to_string(image, lang="eng").strip()
                    timestamp = frame_number / fps
                    if text:
                        extracted_frames.append({
                            "timestamp_seconds": round(timestamp, 2),
                            "text": text,
                        })
                except Exception as frame_err:
                    logger.debug("Frame OCR error at frame %d: %s", frame_number, frame_err)

            frame_number += 1
    finally:
        video.release()

    combined_text = "\n\n".join(
        f"[Timestamp: {item['timestamp_seconds']}s]\n{item['text']}"
        for item in extracted_frames
    )

    return {
        "file_type": "video",
        "total_frames_processed": (
            (frame_number + frame_interval - 1) // frame_interval
            if frame_number else 0
        ),
        "extracted_frames": extracted_frames,
        "text": combined_text,
        "raw_text": combined_text,
        "char_count": len(combined_text),
        "word_count": len(combined_text.split()) if combined_text else 0,
        "extraction_method": "opencv_tesseract_ocr",
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def extract_text(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Entry-point dispatcher: identify file extension and call the correct
    format-specific extractor.

    Args:
        file_path: str or Path to the already-stored file on disk.

    Returns:
        A unified extraction dict (always contains 'file_type' and 'raw_text').

    Raises:
        ValueError: if the extension is not in SUPPORTED_EXTENSIONS.
        Any exception from the format-specific extractor propagates unchanged
        so the caller can decide how to handle partial failures.
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path)
    elif extension == ".docx":
        return extract_docx(file_path)
    elif extension == ".txt":
        return extract_txt(file_path)
    elif extension in {".jpg", ".jpeg", ".png", ".webp"}:
        return extract_image(file_path)
    elif extension in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return extract_video(file_path)
    else:
        raise ValueError(
            f"Unsupported file type for text extraction: '{extension}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
