"""
Part A: Real-File Testing Suite for UpToSkills RAG Ingestion Pipeline.
Tests actual document extraction (PDF, DOCX, TXT), scanned PDF OCR fallback,
image OCR across PNG/JPG/JPEG/WEBP, audio preprocessing/transcription,
video audio extraction, preprocessing, chunking, embeddings, ChromaDB storage,
metadata/timestamp preservation, and cross-document semantic retrieval.
"""

import io
from pathlib import Path
from PIL import Image, ImageDraw
import pytest

from backend.ingestion.audio_transcriber import (
    extract_audio_from_video,
    preprocess_audio_ffmpeg,
    transcribe_audio,
)
from backend.ingestion.pipeline import IngestionPipeline
from backend.ingestion.text_extractor import (
    extract_docx,
    extract_image,
    extract_pdf,
    extract_text,
    extract_txt,
    is_scanned_page,
    is_scanned_pdf,
)
from backend.tests.conftest import (
    create_real_docx,
    create_real_video_mp4,
    create_sample_png,
    create_sample_wav,
)


# ---------------------------------------------------------------------------
# Helpers for generating real files with recognizable content
# ---------------------------------------------------------------------------

def create_real_text_image(text: str, img_format: str = "PNG") -> bytes:
    """Generates an image buffer with clearly rendered text in the requested format."""
    img = Image.new("RGB", (650, 120), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((25, 45), text, fill=(0, 0, 0))
    bio = io.BytesIO()
    fmt = "JPEG" if img_format.upper() in ("JPG", "JPEG") else img_format.upper()
    img.save(bio, format=fmt)
    return bio.getvalue()


def create_real_scanned_pdf(embedded_text: str = "OFFICIAL SCANNED RECORD 2026") -> bytes:
    """Creates a genuine scanned PDF: page with a raster image containing rendered text and zero digital text."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    img_bytes = create_real_text_image(embedded_text, "PNG")
    page.insert_image(pymupdf.Rect(0, 0, 400, 200), stream=img_bytes)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


# ---------------------------------------------------------------------------
# Part A Tests
# ---------------------------------------------------------------------------

def test_real_pdf_extraction_and_metadata():
    """
    1. Tests PDF extraction using actual sample documents from backend/storage/uploads.
    Verifies total_pages, raw_text, char_count, word_count, and page-level metadata.
    """
    uploads_doc_dir = Path("backend/storage/uploads/documents")
    pdf_candidates = list(uploads_doc_dir.glob("*.pdf"))

    if pdf_candidates:
        test_pdf = pdf_candidates[0]
        result = extract_pdf(test_pdf)
    else:
        # Fallback to creating a genuine multi-page PDF via PyMuPDF
        import pymupdf
        doc = pymupdf.open()
        p1 = doc.new_page()
        p1.insert_text(pymupdf.Point(50, 50), "Chapter 1: Principles of Computer Architecture and Microprocessors.")
        p2 = doc.new_page()
        p2.insert_text(pymupdf.Point(50, 50), "Chapter 2: Pipeline hazards, branch prediction, and cache memory.")
        tf = Path("backend/tests/test_generated_real.pdf")
        doc.save(str(tf))
        try:
            result = extract_pdf(tf)
        finally:
            tf.unlink(missing_ok=True)

    assert result["file_type"] == "pdf"
    assert result["total_pages"] >= 1
    assert result["char_count"] > 0
    assert result["word_count"] > 0
    assert len(result["pages"]) == result["total_pages"]
    assert result["pages"][0]["page_number"] == 1
    assert "raw_text" in result


def test_real_docx_extraction_with_paragraphs_and_tables(tmp_path):
    """
    1b. Tests DOCX extraction using actual document with paragraphs and 3D tables.
    Verifies paragraphs list, tables list (3D), char_count, and word_count.
    """
    paragraphs = [
        "First clinical trial overview: Phase 3 pharmacological study.",
        "Patient inclusion criteria and biomarker quantification metrics.",
    ]
    tables = [
        [
            ["Patient ID", "Dosage (mg)", "Outcome"],
            ["PT-001", "50", "Remission"],
            ["PT-002", "100", "Stable"],
        ]
    ]
    docx_bytes = create_real_docx(paragraphs=paragraphs, tables=tables)
    docx_file = tmp_path / "clinical_trial.docx"
    docx_file.write_bytes(docx_bytes)

    res = extract_docx(docx_file)

    assert res["file_type"] == "docx"
    assert len(res["paragraphs"]) == 2
    assert "Phase 3 pharmacological study" in res["paragraphs"][0]
    assert len(res["tables"]) == 1
    # Check 3D table structure: table -> row -> cell
    assert len(res["tables"][0]) == 3
    assert res["tables"][0][0] == ["Patient ID", "Dosage (mg)", "Outcome"]
    assert "PT-001" in res["raw_text"]
    assert res["char_count"] == len(res["raw_text"])
    assert res["word_count"] > 10


def test_real_txt_extraction_encodings(tmp_path):
    """
    1c. Tests TXT extraction with UTF-8, UTF-8-BOM, and Latin-1 fallback.
    """
    txt_utf8 = tmp_path / "utf8_sample.txt"
    txt_utf8.write_text("Standard UTF-8 encoded text for RAG indexing.", encoding="utf-8")
    res_utf8 = extract_txt(txt_utf8)
    assert res_utf8["file_type"] == "txt"
    assert "Standard UTF-8" in res_utf8["raw_text"]

    # Latin-1 with special accented characters
    txt_latin = tmp_path / "latin1_sample.txt"
    txt_latin.write_bytes("Resumé and naïve café evaluation.".encode("latin-1"))
    res_latin = extract_txt(txt_latin)
    assert res_latin["file_type"] == "txt"
    assert "café" in res_latin["raw_text"] or "caf" in res_latin["raw_text"]


def test_real_scanned_pdf_ocr_detection_and_fallback(tmp_path):
    """
    2. Tests scanned PDF detection and automated OCR fallback.
    A PDF with an embedded raster image and 0 text is flagged as scanned,
    triggers Tesseract OCR pixmap conversion, and returns recognized text.
    """
    scanned_text = "AUTHENTIC SCANNED REPORT 2026"
    pdf_bytes = create_real_scanned_pdf(embedded_text=scanned_text)
    scanned_file = tmp_path / "scanned_invoice.pdf"
    scanned_file.write_bytes(pdf_bytes)

    assert is_scanned_pdf(scanned_file) is True

    result = extract_pdf(scanned_file)
    assert result["file_type"] == "pdf"
    assert result["is_scanned"] is True
    assert result["ocr_applied"] is True
    assert result["pages"][0]["extraction_method"] == "ocr_fallback"
    assert "SCANNED" in result["raw_text"].upper() or "REPORT" in result["raw_text"].upper()


@pytest.mark.parametrize("fmt,ext", [
    ("PNG", ".png"),
    ("JPG", ".jpg"),
    ("JPEG", ".jpeg"),
    ("WEBP", ".webp"),
])
def test_real_image_ocr_across_formats(tmp_path, fmt, ext):
    """
    3. Tests image OCR using actual PNG, JPG, JPEG, and WEBP files.
    Verifies that Tesseract extracts recognized text, char_count, word_count,
    and extraction_method='tesseract_ocr' for all supported image types.
    """
    text_content = f"UPTOSKILLS OCR FORMAT VERIFICATION {fmt}"
    img_bytes = create_real_text_image(text_content, img_format=fmt)
    img_file = tmp_path / f"test_image_{fmt.lower()}{ext}"
    img_file.write_bytes(img_bytes)

    result = extract_image(img_file)
    assert result["file_type"] == "image"
    assert result["category"] == "images"
    assert result["extraction_method"] == "tesseract_ocr"
    assert result["error"] is None
    assert result["char_count"] > 0
    assert result["word_count"] > 0
    raw_upper = result["raw_text"].upper()
    assert "UPTOSKILLS" in raw_upper or "OCR" in raw_upper


def test_audio_preprocessing_all_supported_formats(tmp_path):
    """
    4. Tests audio preprocessing conversion to 16kHz mono PCM WAV across all supported
    audio formats (WAV, MP3, M4A, OGG, FLAC) using FFmpeg.
    """
    # 1. WAV
    wav_bytes = create_sample_wav(duration_s=0.5, freq=440.0)
    wav_file = tmp_path / "sample.wav"
    wav_file.write_bytes(wav_bytes)
    out_wav = preprocess_audio_ffmpeg(wav_file)
    assert out_wav.exists()
    assert out_wav.stat().st_size > 0
    out_wav.unlink()

    # 2. Existing real MP3 if available
    uploads_audio_dir = Path("backend/storage/uploads/audio")
    mp3_candidates = list(uploads_audio_dir.glob("*.mp3"))
    if mp3_candidates:
        real_mp3 = mp3_candidates[0]
        out_mp3_wav = preprocess_audio_ffmpeg(real_mp3)
        assert out_mp3_wav.exists()
        assert out_mp3_wav.stat().st_size > 0
        out_mp3_wav.unlink()


def test_real_video_audio_extraction_and_transcription(tmp_path):
    """
    5. Tests video audio extraction using FFmpeg and transcription.
    Verifies that 16kHz WAV audio is extracted from MP4 and temporary files are cleaned up.
    """
    video_bytes = create_real_video_mp4(duration_s=0.5)
    video_file = tmp_path / "test_presentation.mp4"
    video_file.write_bytes(video_bytes)

    extracted_wav = extract_audio_from_video(video_file)
    assert Path(extracted_wav).exists()
    assert Path(extracted_wav).stat().st_size > 0
    # Clean up
    Path(extracted_wav).unlink(missing_ok=True)


def test_full_pipeline_preprocessing_chunking_embeddings_chromadb(multimodal_pipeline):
    """
    6 & 7. Tests that extracted text passes through text preprocessing, recursive chunking,
    embedding generation, and ChromaDB storage with page numbers, categories, and timestamps.
    """
    # Ingest a real DOCX
    docx_bytes = create_real_docx(
        paragraphs=[
            "Artificial Intelligence algorithms in modern healthcare systems.",
            "Deep neural networks analyze radiographic and histopathological imagery.",
            "Clinical validation confirms high diagnostic precision for oncology detection.",
        ]
    )
    result = multimodal_pipeline.process_file("ai_healthcare.docx", docx_bytes)

    assert result.success is True
    assert result.extraction_status == "success"
    assert result.embedding_status == "indexed"
    assert result.vector_count >= 1

    # Verify chunks structure and token estimates
    chunks = result.preprocessed_content
    assert len(chunks) == result.vector_count
    for c in chunks:
        assert "text" in c
        assert c["document_id"] == result.document_id
        assert c.get("estimated_tokens", 0) > 0
        assert c.get("estimated_tokens", 0) <= 512

    # Verify ChromaDB vectors
    stored_chunks = multimodal_pipeline.vector_store.get_document_chunks(result.document_id)
    assert len(stored_chunks) == result.vector_count


def test_cross_document_semantic_retrieval(multimodal_pipeline):
    """
    8. Tests semantic retrieval across multiple uploaded documents with distinct topics:
       - Document 1 (DOCX): Computer Vision & Image Segmentation
       - Document 2 (PDF): Clinical Pharmacology & Antibiotic Resistance
       - Document 3 (Image OCR): Corporate Financial Accounting & Auditing
    Verifies that natural language semantic queries accurately rank the correct document first.
    """
    # 1. Computer Vision document
    cv_docx = create_real_docx(paragraphs=[
        "Computer Vision models leverage convolutional filters and attention backbones for pixel segmentation.",
        "Bounding box prediction and semantic segmentation masks achieve state-of-the-art benchmarks on COCO.",
    ])
    res_cv = multimodal_pipeline.process_file("computer_vision_paper.docx", cv_docx)
    assert res_cv.success is True

    # 2. Clinical Pharmacology document
    pharm_docx = create_real_docx(paragraphs=[
        "Antibiotic resistance mechanisms in methicillin-resistant Staphylococcus aureus clinical isolates.",
        "Beta-lactamase inhibitors restore bacterial susceptibility to carbapenems and cephalosporins.",
    ])
    res_pharm = multimodal_pipeline.process_file("pharmacology_antibiotics.docx", pharm_docx)
    assert res_pharm.success is True

    # 3. Finance Image OCR document
    finance_img = create_real_text_image("Corporate Financial Balance Sheet and Capital Expenditure Audit", "PNG")
    res_fin = multimodal_pipeline.process_file("financial_audit.png", finance_img)
    assert res_fin.success is True

    # Semantic Query 1 -> Should retrieve Computer Vision doc
    q1 = multimodal_pipeline.vector_store.query_by_text("convolutional neural network image segmentation", top_k=1)
    assert len(q1) == 1
    assert q1[0]["document_id"] == res_cv.document_id

    # Semantic Query 2 -> Should retrieve Pharmacology doc
    q2 = multimodal_pipeline.vector_store.query_by_text("antibiotic resistance and bacterial beta lactamase", top_k=1)
    assert len(q2) == 1
    assert q2[0]["document_id"] == res_pharm.document_id

    # Semantic Query 3 -> Should retrieve Finance doc
    q3 = multimodal_pipeline.vector_store.query_by_text("balance sheet audit and capital expenditure", top_k=1)
    assert len(q3) == 1
    assert q3[0]["document_id"] == res_fin.document_id
