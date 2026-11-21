"""
Text extraction engine for the RAG ingestion pipeline.
Handles PDF (PyMuPDF), DOCX (python-docx), and TXT files.

Design rules:
- All extractors return a unified ExtractionResult dict with a top-level
  `raw_text` string so downstream chunking never needs format-specific logic.
- DOCX tables are stored as List[List[List[str]]] (table → row → cell) so
  table identity is preserved and Streamlit can render each table as a proper
  2-D DataFrame.
- Extraction statistics (char_count, word_count, page_count) are recorded for
  every document type.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pymupdf
from docx import Document

logger = logging.getLogger("rag.extraction")

# EXTRACTABLE_EXTENSIONS is used by the pipeline and tests to decide
# whether to attempt text extraction for a given file type.
EXTRACTABLE_EXTENSIONS = {".pdf", ".docx", ".txt"}


def extract_pdf(file_path: Path) -> Dict[str, Any]:
    """
    Extract text page-by-page from a PDF using PyMuPDF.

    Returns a dict with:
        file_type, total_pages, raw_text, char_count, word_count,
        pages (list of {page_number, text, char_count}),
        pdf_metadata
    """
    extracted_pages: List[Dict[str, Any]] = []

    with pymupdf.open(str(file_path)) as pdf:
        pdf_metadata = dict(pdf.metadata)
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            extracted_pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "char_count": len(text),
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


def extract_docx(file_path: Path) -> Dict[str, Any]:
    """
    Extract paragraphs and tables from a DOCX file.

    Tables are stored as List[List[List[str]]] — a 3-D structure
    (table_index → row_index → cell_values) so each table is distinct.

    Returns a dict with:
        file_type, raw_text, char_count, word_count,
        paragraphs (list of str), tables (list of 2-D list)
    """
    document = Document(str(file_path))

    paragraphs: List[str] = [
        p.text.strip()
        for p in document.paragraphs
        if p.text.strip()
    ]

    # 3-D structure: table → row → cells
    tables: List[List[List[str]]] = []
    for table in document.tables:
        table_rows: List[List[str]] = [
            [cell.text.strip() for cell in row.cells]
            for row in table.rows
        ]
        tables.append(table_rows)

    # Flatten tables into plain text for raw_text
    table_texts: List[str] = []
    for t in tables:
        for row in t:
            table_texts.append(" | ".join(row))

    raw_text = "\n".join(paragraphs + table_texts)
    return {
        "file_type": "docx",
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "word_count": len(raw_text.split()) if raw_text else 0,
        "paragraphs": paragraphs,
        "tables": tables,
    }


def extract_txt(file_path: Path) -> Dict[str, Any]:
    """
    Extract plain text from a TXT file.
    Attempts UTF-8-with-BOM first, then Latin-1 as fallback.

    Returns a dict with:
        file_type, raw_text, char_count, word_count
    """
    file_path = Path(file_path)
    try:
        text = file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="latin-1")

    text = text.strip()
    return {
        "file_type": "txt",
        "raw_text": text,
        "char_count": len(text),
        "word_count": len(text.split()) if text else 0,
        # Alias for backward compatibility with Streamlit code
        "text": text,
    }


def extract_text(file_path) -> Dict[str, Any]:
    """
    Entry-point dispatcher: identify file extension and call the correct
    format-specific extractor.

    Args:
        file_path: str or Path to the already-stored file on disk.

    Returns:
        A unified extraction dict (always contains 'file_type' and 'raw_text').

    Raises:
        ValueError: if the extension is not in EXTRACTABLE_EXTENSIONS.
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
    else:
        raise ValueError(
            f"Unsupported file type for text extraction: '{extension}'. "
            f"Supported: {sorted(EXTRACTABLE_EXTENSIONS)}"
        )

