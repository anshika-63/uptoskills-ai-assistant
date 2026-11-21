"""
Unit tests for backend.ingestion.text_extractor.
Covers:
  - PDF extraction (digital text, page counts, raw_text presence)
  - DOCX extraction (paragraphs, 3-D table structure, raw_text)
  - TXT extraction (UTF-8, BOM, Latin-1 fallback, raw_text)
  - Unsupported extension raises ValueError
  - EXTRACTABLE_EXTENSIONS constant
"""

import io
import os
import tempfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from backend.ingestion.text_extractor import (
    EXTRACTABLE_EXTENSIONS,
    extract_docx,
    extract_pdf,
    extract_text,
    extract_txt,
)
from backend.tests.conftest import create_sample_pdf


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _write_tmp(suffix: str, data: bytes) -> Path:
    """Write bytes to a named temp file and return its Path."""
    tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tf.write(data)
    tf.close()
    return Path(tf.name)


def _make_real_docx(paragraphs=None, tables=None) -> bytes:
    """
    Build a genuine DOCX using python-docx so it exercises real extraction.
    paragraphs: list[str]
    tables: list[list[list[str]]]  — table → rows → cells
    """
    from docx import Document as DocxDoc

    doc = DocxDoc()
    for p in (paragraphs or []):
        doc.add_paragraph(p)
    for tbl in (tables or []):
        if not tbl:
            continue
        rows, cols = len(tbl), len(tbl[0])
        word_table = doc.add_table(rows=rows, cols=cols)
        for r_i, row_data in enumerate(tbl):
            for c_i, cell_text in enumerate(row_data):
                word_table.cell(r_i, c_i).text = cell_text
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


# ─────────────────────────────────────────────
# EXTRACTABLE_EXTENSIONS
# ─────────────────────────────────────────────

def test_extractable_extensions_contains_expected():
    assert ".pdf"  in EXTRACTABLE_EXTENSIONS
    assert ".docx" in EXTRACTABLE_EXTENSIONS
    assert ".txt"  in EXTRACTABLE_EXTENSIONS
    assert ".png"  not in EXTRACTABLE_EXTENSIONS
    assert ".mp4"  not in EXTRACTABLE_EXTENSIONS


# ─────────────────────────────────────────────
# PDF extraction
# ─────────────────────────────────────────────

def test_extract_pdf_returns_required_keys():
    path = _write_tmp(".pdf", create_sample_pdf())
    try:
        result = extract_pdf(path)
    finally:
        os.unlink(path)

    assert result["file_type"] == "pdf"
    assert "pages" in result
    assert "raw_text" in result
    assert "char_count" in result
    assert "word_count" in result
    assert "total_pages" in result
    assert isinstance(result["total_pages"], int)
    assert result["total_pages"] >= 1


def test_extract_pdf_pages_have_page_number_and_text():
    path = _write_tmp(".pdf", create_sample_pdf())
    try:
        result = extract_pdf(path)
    finally:
        os.unlink(path)

    for page in result["pages"]:
        assert "page_number" in page
        assert "text" in page
        assert isinstance(page["page_number"], int)
        assert isinstance(page["text"], str)


def test_extract_pdf_char_count_matches_raw_text():
    path = _write_tmp(".pdf", create_sample_pdf())
    try:
        result = extract_pdf(path)
    finally:
        os.unlink(path)

    assert result["char_count"] == len(result["raw_text"])


def test_extract_text_dispatcher_pdf():
    path = _write_tmp(".pdf", create_sample_pdf())
    try:
        result = extract_text(path)
    finally:
        os.unlink(path)
    assert result["file_type"] == "pdf"


# ─────────────────────────────────────────────
# DOCX extraction
# ─────────────────────────────────────────────

def test_extract_docx_paragraphs():
    data = _make_real_docx(paragraphs=["First paragraph", "Second paragraph"])
    path = _write_tmp(".docx", data)
    try:
        result = extract_docx(path)
    finally:
        os.unlink(path)

    assert result["file_type"] == "docx"
    assert "First paragraph" in result["paragraphs"]
    assert "Second paragraph" in result["paragraphs"]
    assert "raw_text" in result
    assert "First paragraph" in result["raw_text"]


def test_extract_docx_tables_are_3d():
    """
    Tables must be List[List[List[str]]] — table → row → cell.
    Two separate tables must NOT be merged into one flat list.
    """
    t1 = [["A1", "B1"], ["A2", "B2"]]
    t2 = [["X1", "Y1"]]
    data = _make_real_docx(tables=[t1, t2])
    path = _write_tmp(".docx", data)
    try:
        result = extract_docx(path)
    finally:
        os.unlink(path)

    tables = result["tables"]
    # Must have exactly 2 separate tables
    assert len(tables) == 2, f"Expected 2 tables, got {len(tables)}: {tables}"

    # Table 1: 2 rows × 2 cols
    assert len(tables[0]) == 2
    assert tables[0][0] == ["A1", "B1"]
    assert tables[0][1] == ["A2", "B2"]

    # Table 2: 1 row × 2 cols
    assert len(tables[1]) == 1
    assert tables[1][0] == ["X1", "Y1"]


def test_extract_docx_table_content_in_raw_text():
    t = [["ColA", "ColB"], ["v1", "v2"]]
    data = _make_real_docx(tables=[t])
    path = _write_tmp(".docx", data)
    try:
        result = extract_docx(path)
    finally:
        os.unlink(path)

    assert "ColA" in result["raw_text"]
    assert "v2"   in result["raw_text"]


def test_extract_docx_char_and_word_counts():
    data = _make_real_docx(paragraphs=["hello world"])
    path = _write_tmp(".docx", data)
    try:
        result = extract_docx(path)
    finally:
        os.unlink(path)

    assert result["char_count"] == len(result["raw_text"])
    assert result["word_count"] >= 2  # at least "hello world"


def test_extract_text_dispatcher_docx():
    data = _make_real_docx(paragraphs=["Dispatcher test"])
    path = _write_tmp(".docx", data)
    try:
        result = extract_text(path)
    finally:
        os.unlink(path)
    assert result["file_type"] == "docx"


# ─────────────────────────────────────────────
# TXT extraction
# ─────────────────────────────────────────────

def test_extract_txt_utf8():
    content = "UpToSkills RAG pipeline — text extraction test."
    path = _write_tmp(".txt", content.encode("utf-8"))
    try:
        result = extract_txt(path)
    finally:
        os.unlink(path)

    assert result["file_type"] == "txt"
    assert content.strip() in result["raw_text"]
    # backward-compat alias
    assert "text" in result
    assert result["text"] == result["raw_text"]


def test_extract_txt_utf8_bom():
    content = "BOM text"
    # utf-8-sig adds BOM
    path = _write_tmp(".txt", content.encode("utf-8-sig"))
    try:
        result = extract_txt(path)
    finally:
        os.unlink(path)
    assert result["raw_text"] == content


def test_extract_txt_latin1_fallback():
    # latin-1 encoded string that is NOT valid utf-8
    content = "caf\xe9 and r\xe9sum\xe9"
    path = _write_tmp(".txt", content.encode("latin-1"))
    try:
        result = extract_txt(path)
    finally:
        os.unlink(path)
    assert len(result["raw_text"]) > 0


def test_extract_txt_char_and_word_counts():
    text = "five words here are present"
    path = _write_tmp(".txt", text.encode("utf-8"))
    try:
        result = extract_txt(path)
    finally:
        os.unlink(path)

    assert result["char_count"] == len(text)
    assert result["word_count"] == 5


def test_extract_text_dispatcher_txt():
    path = _write_tmp(".txt", b"dispatcher test")
    try:
        result = extract_text(path)
    finally:
        os.unlink(path)
    assert result["file_type"] == "txt"


# ─────────────────────────────────────────────
# Unsupported extension
# ─────────────────────────────────────────────

def test_extract_text_unsupported_raises():
    path = _write_tmp(".exe", b"MZ binary content")
    try:
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text(path)
    finally:
        os.unlink(path)
