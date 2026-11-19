
from pathlib import Path

import pymupdf
from docx import Document


def extract_pdf(file_path):
    """Extract text page by page from a PDF."""

    extracted_pages = []

    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")

            extracted_pages.append({
                "page_number": page_number,
                "text": text.strip()
            })

        pdf_metadata = pdf.metadata

    return {
        "file_type": "pdf",
        "total_pages": len(extracted_pages),
        "metadata": pdf_metadata,
        "pages": extracted_pages
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

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )
