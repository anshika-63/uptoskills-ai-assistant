
from pathlib import Path
from io import BytesIO

import cv2
import pymupdf
import pytesseract
from PIL import Image
from docx import Document


def extract_pdf(file_path):
    """Extract PDF text using PyMuPDF with OCR fallback."""

    extracted_pages = []

    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            extraction_method = "pymupdf"

            if not text:
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )
                image = Image.open(BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(
                    image, lang="eng"
                ).strip()
                extraction_method = "ocr"

            extracted_pages.append({
                "page_number": page_number,
                "text": text,
                "extraction_method": extraction_method
            })

        pdf_metadata = pdf.metadata

    return {
        "file_type": "pdf",
        "total_pages": len(extracted_pages),
        "metadata": pdf_metadata,
        "pages": extracted_pages
    }


def extract_image(file_path):
    """Extract text from images using Tesseract OCR."""

    image = Image.open(file_path).convert("RGB")
    text = pytesseract.image_to_string(
        image, lang="eng"
    ).strip()

    return {
        "file_type": "image",
        "text": text,
        "extraction_method": "tesseract_ocr"
    }


def extract_docx(file_path):
    """Extract paragraphs and tables from a DOCX file."""

    document = Document(file_path)

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    tables = []

    for table in document.tables:
        for row in table.rows:
            tables.append([
                cell.text.strip() for cell in row.cells
            ])

    return {
        "file_type": "docx",
        "paragraphs": paragraphs,
        "tables": tables
    }


def extract_txt(file_path):
    """Extract text from a TXT file."""

    file_path = Path(file_path)

    try:
        text = file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="latin-1")

    return {
        "file_type": "txt",
        "text": text.strip()
    }


def extract_video(file_path, frame_interval_seconds=1):
    """Extract video frames and recognize visible text using OCR."""

    video = cv2.VideoCapture(str(file_path))

    if not video.isOpened():
        raise ValueError("Unable to open video file.")

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

            if not success:
                break

            if frame_number % frame_interval == 0:
                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )
                image = Image.fromarray(rgb_frame)

                text = pytesseract.image_to_string(
                    image, lang="eng"
                ).strip()

                timestamp = frame_number / fps

                if text:
                    extracted_frames.append({
                        "timestamp_seconds": round(timestamp, 2),
                        "text": text
                    })

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
        "extraction_method": "opencv_tesseract_ocr"
    }


def extract_text(file_path):
    """Identify file extension and call the correct extractor."""

    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path)

    elif extension == ".docx":
        return extract_docx(file_path)

    elif extension == ".txt":
        return extract_txt(file_path)

    elif extension in [".jpg", ".jpeg", ".png", ".webp"]:
        return extract_image(file_path)

    elif extension in [
        ".mp4", ".mov", ".avi", ".mkv", ".webm"
    ]:
        return extract_video(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )
