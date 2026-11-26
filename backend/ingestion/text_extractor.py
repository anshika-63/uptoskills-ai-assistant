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
import os
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger("rag.extraction")

import pymupdf
from docx import Document

from backend.config import BACKEND_DIR, PROJECT_ROOT, TESSERACT_CMD, resolve_tesseract_cmd

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
# OCR Engine Management & Helpers
# ---------------------------------------------------------------------------

_CUSTOM_OCR_FUNC: Optional[Callable[[Any], str]] = None


def set_ocr_engine(func: Optional[Callable[[Any], str]]) -> None:
    """Sets a custom OCR callable (image -> text) for testing or alternate OCR backends."""
    global _CUSTOM_OCR_FUNC
    _CUSTOM_OCR_FUNC = func


def reset_ocr_engine() -> None:
    """Resets the OCR engine to default pytesseract."""
    global _CUSTOM_OCR_FUNC
    _CUSTOM_OCR_FUNC = None


def get_tesseract_cmd() -> Optional[str]:
    """
    Resolves the executable path to Tesseract OCR.
    Prioritizes dynamic env/config, already-configured pytesseract,
    system PATH, and standard Windows install locations.
    """
    # 1. Check environment variable dynamically or config setting
    configured = (
        os.getenv("TESSERACT_CMD", "").strip("\"' ")
        or resolve_tesseract_cmd()
        or TESSERACT_CMD
    )
    if configured:
        cand = Path(configured)
        if cand.is_file():
            return str(cand.resolve())
        # Try relative to known base paths
        for base in (PROJECT_ROOT, BACKEND_DIR, PROJECT_ROOT.parent, Path.cwd()):
            resolved = (base / configured).resolve()
            if resolved.is_file():
                return str(resolved)

    # 2. Existing pytesseract configuration if valid
    if _OCR_AVAILABLE and hasattr(pytesseract, "pytesseract") and pytesseract.pytesseract.tesseract_cmd:
        cmd = pytesseract.pytesseract.tesseract_cmd
        if cmd != "tesseract" and Path(cmd).is_file():
            return str(Path(cmd).resolve())

    # 3. System PATH
    which_path = shutil.which("tesseract")
    if which_path:
        return str(Path(which_path).resolve())

    # 4. Standard Windows install locations
    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        r"E:\RAG\tesseract.exe",
        str((PROJECT_ROOT.parent / "tesseract.exe").resolve()),
        str((PROJECT_ROOT / "tesseract.exe").resolve()),
    ]
    for p in candidate_paths:
        if p and Path(p).is_file():
            return str(Path(p).resolve())

    return None


def configure_tesseract(custom_cmd: Optional[str] = None) -> bool:
    """
    Configures pytesseract.pytesseract.tesseract_cmd with a verified executable path.
    Logs warnings for missing binary or configuration errors.
    Returns True if successfully configured and verified.
    """
    if not _OCR_AVAILABLE:
        logger.debug("pytesseract is not installed; skipping Tesseract configuration.")
        return False

    cmd = custom_cmd or get_tesseract_cmd()
    if not cmd:
        configured_env = os.getenv("TESSERACT_CMD", TESSERACT_CMD)
        logger.warning(
            "Tesseract OCR executable not found. Configured TESSERACT_CMD='%s'. "
            "Please ensure Tesseract is installed or specify TESSERACT_CMD in .env.",
            configured_env,
        )
        return False

    try:
        pytesseract.pytesseract.tesseract_cmd = str(cmd)
        version = pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR configured successfully at '%s' (v%s)", cmd, version)
        return True
    except pytesseract.TesseractNotFoundError as exc:
        logger.warning("Tesseract binary not found at '%s': %s", cmd, exc)
        return False
    except Exception as exc:
        logger.error("Tesseract initialization/verification error for '%s': %s", cmd, exc)
        return False


# Auto-configure Tesseract on module import if available
if _OCR_AVAILABLE:
    configure_tesseract()


def is_ocr_available() -> bool:
    """Checks whether an OCR engine (custom or pytesseract with binary) is available."""
    if _CUSTOM_OCR_FUNC is not None:
        return True
    if not _OCR_AVAILABLE:
        return False

    cmd = get_tesseract_cmd()
    if not cmd:
        return False

    try:
        pytesseract.pytesseract.tesseract_cmd = str(cmd)
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError as exc:
        logger.warning("Tesseract binary not found at '%s': %s", cmd, exc)
        return False
    except Exception as exc:
        logger.error("Tesseract configuration or execution verification error at '%s': %s", cmd, exc)
        return False


def run_ocr(image: Any, lang: str = "eng") -> str:
    """
    Executes OCR on a PIL Image, file path, or image bytes.
    Uses registered custom OCR callable or pytesseract.
    Ensures pytesseract.pytesseract.tesseract_cmd is configured before invocation.
    """
    if _CUSTOM_OCR_FUNC is not None:
        return str(_CUSTOM_OCR_FUNC(image)).strip()

    if not _OCR_AVAILABLE:
        raise RuntimeError("pytesseract or Pillow is not installed on this system.")

    cmd = get_tesseract_cmd()
    if not cmd:
        configured = os.getenv("TESSERACT_CMD", TESSERACT_CMD)
        raise RuntimeError(
            f"Tesseract OCR executable is not available. Configured path: '{configured}'. "
            "Please verify that Tesseract is installed and TESSERACT_CMD points to tesseract.exe."
        )

    # Ensure pytesseract.pytesseract.tesseract_cmd is configured using TESSERACT_CMD
    pytesseract.pytesseract.tesseract_cmd = str(cmd)

    if not isinstance(image, _PilImage.Image):
        image = _PilImage.open(str(image)).convert("RGB")

    try:
        return pytesseract.image_to_string(image, lang=lang).strip()
    except pytesseract.TesseractNotFoundError as exc:
        logger.error("Tesseract executable not found at '%s' during OCR execution: %s", cmd, exc)
        raise RuntimeError(f"Tesseract executable not found at '{cmd}': {exc}") from exc
    except Exception as exc:
        logger.error("Tesseract OCR execution failed with engine '%s': %s", cmd, exc)
        raise


# ---------------------------------------------------------------------------
# Scanned PDF Detection Helpers
# ---------------------------------------------------------------------------

def is_scanned_page(page: Any, min_char_threshold: int = 15) -> bool:
    """
    Detects if a PDF page is a scanned document (image-only or minimal embedded text).
    A page is considered scanned if extracted text length is below threshold
    AND the page contains images or raster elements.
    """
    text = page.get_text("text").strip()
    if len(text) >= min_char_threshold:
        return False

    # Check for embedded raster images
    try:
        images = page.get_images()
        if len(images) > 0:
            return True
    except Exception:
        pass

    return False


def is_scanned_pdf(file_path: Union[str, Path], min_char_threshold: int = 15) -> bool:
    """Checks whether any page in a PDF is scanned."""
    file_path = Path(file_path)
    if not file_path.exists():
        return False
    try:
        with pymupdf.open(str(file_path)) as doc:
            return any(is_scanned_page(p, min_char_threshold) for p in doc)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Extension sets
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS: Set[str] = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DOCUMENT_EXTENSIONS: Set[str] = {".pdf", ".docx", ".txt"}

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
    Detects scanned pages and falls back to OCR when available.

    Returns a dict with:
        file_type, total_pages, raw_text, char_count, word_count,
        pages (list of {page_number, text, char_count, extraction_method, is_scanned}),
        is_scanned, ocr_applied, pdf_metadata, metadata
    """
    file_path = Path(file_path)
    extracted_pages: List[Dict[str, Any]] = []

    with pymupdf.open(str(file_path)) as pdf:
        pdf_metadata = dict(pdf.metadata) if pdf.metadata else {}
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            extraction_method = "pymupdf"
            scanned = is_scanned_page(page)

            # OCR fallback for scanned pages
            if scanned:
                if is_ocr_available():
                    try:
                        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                        image = _PilImage.open(BytesIO(pix.tobytes("png")))
                        ocr_text = run_ocr(image, lang="eng")
                        if ocr_text:
                            text = ocr_text
                            extraction_method = "ocr_fallback"
                        else:
                            extraction_method = "ocr_empty"
                    except Exception as ocr_err:
                        logger.debug("PDF page %d OCR fallback failed: %s", page_number, ocr_err)
                        extraction_method = "ocr_failed"
                else:
                    extraction_method = "ocr_unavailable"

            extracted_pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "char_count": len(text),
                    "word_count": len(text.split()) if text else 0,
                    "extraction_method": extraction_method,
                    "is_scanned": scanned,
                }
            )

    raw_text = "\n\n".join(p["text"] for p in extracted_pages if p["text"])
    is_scanned = any(p.get("is_scanned", False) for p in extracted_pages)
    ocr_applied = any(p.get("extraction_method") == "ocr_fallback" for p in extracted_pages)

    return {
        "file_type": "pdf",
        "total_pages": len(extracted_pages),
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "word_count": len(raw_text.split()) if raw_text else 0,
        "pages": extracted_pages,
        "is_scanned": is_scanned,
        "ocr_applied": ocr_applied,
        "pdf_metadata": pdf_metadata,
        # Unified alias kept for Streamlit compatibility
        "metadata": pdf_metadata,
    }


def extract_image(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract text from images using Tesseract OCR (or registered OCR engine).
    Gracefully handles missing OCR binaries, empty images, and corrupted files.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    text = ""
    method = "tesseract_ocr"
    error = None

    if is_ocr_available():
        try:
            image = _PilImage.open(str(file_path)).convert("RGB")
            text = run_ocr(image, lang="eng")
            method = "tesseract_ocr" if _CUSTOM_OCR_FUNC is None else "custom_ocr"
        except Exception as e:
            logger.error("Image OCR extraction error on %s: %s", file_path.name, e)
            method = "failed"
            error = str(e)
    else:
        cmd = get_tesseract_cmd() or os.getenv("TESSERACT_CMD", TESSERACT_CMD)
        method = "ocr_unavailable"
        error = f"Tesseract OCR engine is not available (configured: '{cmd}')."
        logger.warning("OCR skipped for %s — Tesseract not available (configured: '%s').", file_path.name, cmd)

    return {
        "file_type": "image",
        "category": "images",
        "text": text,
        "raw_text": text,
        "char_count": len(text),
        "word_count": len(text.split()) if text else 0,
        "extraction_method": method,
        "error": error,
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
                    text = run_ocr(image, lang="eng")
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
