"""
Text extraction module for UpToSkills AI Assistant.
Extracts structured text, tables, and metadata from documents (PDF, DOCX, TXT),
images (OCR), and videos (frame sampling + OCR).
"""

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import logging

import cv2
import pymupdf
import pytesseract
from PIL import Image
from docx import Document

logger = logging.getLogger(__name__)

# Supported extractable formats
SUPPORTED_EXTENSIONS: Set[str] = {
    ".pdf",
    ".docx",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}

EXTRACTABLE_EXTENSIONS: Set[str] = SUPPORTED_EXTENSIONS


def extract_pdf(file_path: Union[str, Path]) -> Dict:
    """Extract PDF text using PyMuPDF with OCR fallback for scanned pages."""
    file_path = Path(file_path)
    extracted_pages = []

    with pymupdf.open(str(file_path)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            extraction_method = "pymupdf"

            if not text:
                try:
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                    image = Image.open(BytesIO(pix.tobytes("png")))
                    text = pytesseract.image_to_string(image, lang="eng").strip()
                    extraction_method = "ocr"
                except Exception as ocr_err:
                    logger.debug(f"PDF page {page_number} OCR fallback failed: {ocr_err}")
                    text = ""
                    extraction_method = "failed"

            extracted_pages.append({
                "page_number": page_number,
                "text": text,
                "char_count": len(text),
                "word_count": len(text.split()) if text else 0,
                "extraction_method": extraction_method,
            })

        pdf_metadata = dict(pdf.metadata) if pdf.metadata else {}

    raw_text = "\n\n".join(p["text"] for p in extracted_pages if p.get("text")).strip()

    return {
        "file_type": "pdf",
        "total_pages": len(extracted_pages),
        "metadata": pdf_metadata,
        "pages": extracted_pages,
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "word_count": len(raw_text.split()) if raw_text else 0,
    }


def extract_image(file_path: Union[str, Path]) -> Dict:
    """Extract text from images using Tesseract OCR."""
    file_path = Path(file_path)
    text = ""
    method = "tesseract_ocr"

    try:
        image = Image.open(str(file_path)).convert("RGB")
        text = pytesseract.image_to_string(image, lang="eng").strip()
    except Exception as e:
        logger.warning(f"Image OCR extraction error on {file_path.name}: {e}")
        method = "ocr_unavailable"

    return {
        "file_type": "image",
        "text": text,
        "raw_text": text,
        "char_count": len(text),
        "word_count": len(text.split()) if text else 0,
        "extraction_method": method,
    }


def extract_docx(file_path: Union[str, Path]) -> Dict:
    """Extract paragraphs and structured tables from a DOCX file."""
    file_path = Path(file_path)
    try:
        document = Document(str(file_path))
    except Exception as e:
        logger.warning(f"Failed to open DOCX file {file_path}: {e}")
        return {
            "file_type": "docx",
            "paragraphs": [],
            "tables": [],
            "raw_text": "",
            "char_count": 0,
            "word_count": 0,
            "error": str(e),
        }

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    tables: List[List[List[str]]] = []
    for table in document.tables:
        table_grid: List[List[str]] = []
        for row in table.rows:
            table_grid.append([cell.text.strip() for cell in row.cells])
        tables.append(table_grid)

    # Build raw text representation
    text_parts = list(paragraphs)
    for t_idx, table in enumerate(tables, start=1):
        for row in table:
            row_str = " | ".join(cell for cell in row if cell)
            if row_str:
                text_parts.append(row_str)

    raw_text = "\n\n".join(text_parts).strip()

    return {
        "file_type": "docx",
        "paragraphs": paragraphs,
        "tables": tables,
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "word_count": len(raw_text.split()) if raw_text else 0,
    }


def extract_txt(file_path: Union[str, Path]) -> Dict:
    """Extract text from a TXT file."""
    file_path = Path(file_path)

    try:
        text = file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="latin-1", errors="replace")

    clean = text.strip()
    return {
        "file_type": "txt",
        "text": clean,
        "raw_text": clean,
        "char_count": len(clean),
        "word_count": len(clean.split()) if clean else 0,
    }


def extract_video(
    file_path: Union[str, Path],
    frame_interval_seconds: float = 1.0,
    max_frames: int = 60,
) -> Dict:
    """Extract video frames and recognize visible text using OCR with frame bounds."""
    file_path = Path(file_path)
    video = cv2.VideoCapture(str(file_path))

    if not video.isOpened():
        raise ValueError(f"Unable to open video file: {file_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
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
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb_frame)
                    text = pytesseract.image_to_string(image, lang="eng").strip()
                    timestamp = frame_number / fps
                    if text:
                        extracted_frames.append({
                            "timestamp_seconds": round(timestamp, 2),
                            "text": text,
                        })
                except Exception as frame_err:
                    logger.debug(f"Frame OCR error at frame {frame_number}: {frame_err}")

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


def extract_text(file_path: Union[str, Path]) -> Dict:
    """Identify file extension and call the correct extractor."""
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
        raise ValueError(f"Unsupported file type: {extension}")
