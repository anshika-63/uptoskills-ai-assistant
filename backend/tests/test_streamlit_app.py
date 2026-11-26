"""
Automated tests for Streamlit UI rendering and widget key uniqueness in backend/streamlit_app.py.
Verifies that multiple PDF, DOCX, TXT, and audio files can be rendered simultaneously without
encountering StreamlitDuplicateElementId errors.
"""

import sys
from pathlib import Path
import pytest

# Ensure backend package is discoverable
backend_dir = Path(__file__).resolve().parent.parent
project_root = backend_dir.parent
for p in (str(project_root), str(backend_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from streamlit.testing.v1 import AppTest
from backend.streamlit_app import render_extraction


def test_render_multiple_audio_transcripts_unique_keys():
    """
    Verifies that multiple audio transcripts rendered simultaneously assign stable,
    unique keys to st.text_area and st.download_button widgets without raising
    StreamlitDuplicateElementId.
    """
    def app():
        import sys
        from pathlib import Path
        b_dir = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path(".")
        sys.path.insert(0, str(b_dir.parent))
        from backend.streamlit_app import render_extraction

        audio1 = {
            "file_type": "mp3",
            "category": "audio",
            "raw_text": "First speech sample transcript.",
            "char_count": 30,
            "word_count": 4,
            "duration": 2.5,
            "language": "en",
            "language_probability": 0.99,
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.5, "text": "First speech sample transcript.", "timestamp": "[00:00.00 -> 00:02.50]"}
            ],
            "segment_count": 1,
        }
        audio2 = {
            "file_type": "wav",
            "category": "audio",
            "raw_text": "Second lecture transcript content.",
            "char_count": 34,
            "word_count": 4,
            "duration": 4.0,
            "language": "en",
            "language_probability": 0.95,
            "segments": [
                {"id": 0, "start": 0.0, "end": 4.0, "text": "Second lecture transcript content.", "timestamp": "[00:00.00 -> 00:04.00]"}
            ],
            "segment_count": 1,
        }

        render_extraction(audio1, document_id="doc_audio_001", key_prefix="upload_0")
        render_extraction(audio2, document_id="doc_audio_002", key_prefix="upload_1")

    at = AppTest.from_function(app).run()

    # Must run with zero exceptions (no StreamlitDuplicateElementId)
    assert len(at.exception) == 0

    # Must render both text_area widgets with unique keys
    assert len(at.text_area) == 2
    keys = [ta.key for ta in at.text_area]
    assert len(set(keys)) == 2
    assert "upload_0_doc_audio_001_audio_transcript" in keys
    assert "upload_1_doc_audio_002_audio_transcript" in keys

    # Verify download buttons also render without error
    dl_buttons = at.get("download_button")
    assert len(dl_buttons) == 2


def test_render_multiple_txt_files_unique_keys():
    """
    Verifies that multiple TXT document extractions assign unique keys to
    st.text_area widgets.
    """
    def app():
        from backend.streamlit_app import render_extraction

        txt1 = {
            "file_type": "txt",
            "raw_text": "Plain text document content A.",
            "char_count": 30,
            "word_count": 5,
        }
        txt2 = {
            "file_type": "txt",
            "raw_text": "Plain text document content B.",
            "char_count": 30,
            "word_count": 5,
        }

        render_extraction(txt1, document_id="doc_txt_001", key_prefix="upload_0")
        render_extraction(txt2, document_id="doc_txt_002", key_prefix="upload_1")

    at = AppTest.from_function(app).run()
    assert len(at.exception) == 0
    assert len(at.text_area) == 2
    keys = [ta.key for ta in at.text_area]
    assert len(set(keys)) == 2
    assert "upload_0_doc_txt_001_txt_content" in keys
    assert "upload_1_doc_txt_002_txt_content" in keys


def test_render_mixed_pdf_docx_audio_simultaneously():
    """
    Verifies simultaneous rendering of PDF, DOCX, and Audio documents
    in a single session without conflicts.
    """
    def app():
        from backend.streamlit_app import render_extraction

        pdf_payload = {
            "file_type": "pdf",
            "total_pages": 2,
            "pages": [
                {"page_number": 1, "text": "Page 1 intro text", "char_count": 16},
                {"page_number": 2, "text": "Page 2 conclusion text", "char_count": 22},
            ],
            "char_count": 38,
            "word_count": 7,
        }

        docx_payload = {
            "file_type": "docx",
            "paragraphs": ["First paragraph in docx.", "Second paragraph."],
            "tables": [[["Header1", "Header2"], ["Val1", "Val2"]]],
            "char_count": 41,
            "word_count": 6,
        }

        audio_payload = {
            "file_type": "mp3",
            "category": "audio",
            "raw_text": "Interview transcript recording.",
            "duration": 12.4,
            "language": "en",
            "language_probability": 0.97,
            "segments": [
                {"id": 0, "start": 0.0, "end": 12.4, "text": "Interview transcript recording.", "timestamp": "[00:00.00 -> 00:12.40]"}
            ],
            "char_count": 31,
            "word_count": 3,
        }

        render_extraction(pdf_payload, document_id="doc_pdf_1", key_prefix="upload_0")
        render_extraction(docx_payload, document_id="doc_docx_2", key_prefix="upload_1")
        render_extraction(audio_payload, document_id="doc_audio_3", key_prefix="upload_2")

    at = AppTest.from_function(app).run()
    assert len(at.exception) == 0
    # Audio has 1 text_area, PDF and DOCX don't use text_area
    assert len(at.text_area) == 1
    assert at.text_area[0].key == "upload_2_doc_audio_3_audio_transcript"
    assert len(at.get("download_button")) == 1


def test_render_duplicate_audio_uploads_unique_keys():
    """
    Verifies that when duplicate audio files are uploaded in the same batch,
    the distinct card prefixes ensure zero widget key collisions.
    """
    def app():
        from backend.streamlit_app import render_extraction

        dup_payload = {
            "file_type": "mp3",
            "category": "audio",
            "raw_text": "Identical audio payload content.",
            "char_count": 31,
            "word_count": 4,
            "duration": 5.0,
            "language": "en",
            "language_probability": 0.99,
            "segments": [],
        }

        # Simulating first upload card and second duplicate card
        render_extraction(dup_payload, document_id="doc_dup_original", key_prefix="upload_0")
        render_extraction(dup_payload, document_id="doc_dup_original", key_prefix="upload_1")

    at = AppTest.from_function(app).run()
    assert len(at.exception) == 0
    assert len(at.text_area) == 2
    keys = [ta.key for ta in at.text_area]
    assert len(set(keys)) == 2
    assert "upload_0_doc_dup_original_audio_transcript" in keys
    assert "upload_1_doc_dup_original_audio_transcript" in keys


def test_render_fallback_deterministic_key():
    """
    Verifies that calling render_extraction without explicit document_id or
    key_prefix generates a stable deterministic key from content hash.
    """
    def app():
        from backend.streamlit_app import render_extraction

        audio = {
            "file_type": "ogg",
            "category": "audio",
            "raw_text": "Audio without explicit doc_id.",
            "duration": 3.0,
            "segments": [],
            "char_count": 30,
            "word_count": 4,
        }
        render_extraction(audio)

    at = AppTest.from_function(app).run()
    assert len(at.exception) == 0
    assert len(at.text_area) == 1
    assert at.text_area[0].key.startswith("ext_")
    assert at.text_area[0].key.endswith("_audio_transcript")


def test_render_image_ocr_in_document_inspector():
    """
    Verifies that Image OCR extracted payload renders properly in the Document
    Inspector with stable 'inspect' prefix, displaying the recognized text and stats.
    """
    def app():
        from backend.streamlit_app import render_extraction

        image_payload = {
            "file_type": "image",
            "category": "images",
            "raw_text": "Invoice #10492 Total: $450.00 Due: 2026-12-01",
            "text": "Invoice #10492 Total: $450.00 Due: 2026-12-01",
            "char_count": 45,
            "word_count": 6,
            "extraction_method": "tesseract_ocr",
            "error": None,
        }

        render_extraction(image_payload, document_id="doc_img_999", key_prefix="inspect")

    at = AppTest.from_function(app).run()
    assert len(at.exception) == 0
    assert len(at.text_area) == 1
    assert at.text_area[0].key == "inspect_doc_img_999_image_ocr"
    assert "Invoice #10492" in at.text_area[0].value
    # Verify caption rendered with stats
    captions = [c.value for c in at.caption]
    assert any("Image OCR (tesseract_ocr)" in c and "45 chars" in c for c in captions)


def test_render_mixed_pdf_docx_audio_image_simultaneously():
    """
    Verifies simultaneous rendering of PDF, DOCX, Audio, and Image OCR
    documents without widget key collisions or errors.
    """
    def app():
        from backend.streamlit_app import render_extraction

        pdf_payload = {
            "file_type": "pdf",
            "total_pages": 1,
            "pages": [{"page_number": 1, "text": "PDF text content"}],
            "char_count": 16,
            "word_count": 3,
        }
        audio_payload = {
            "file_type": "mp3",
            "category": "audio",
            "raw_text": "Audio transcription sample.",
            "char_count": 27,
            "word_count": 3,
            "duration": 3.0,
            "segments": [],
        }
        image_payload = {
            "file_type": "image",
            "category": "images",
            "raw_text": "Optical character recognition text in image.",
            "char_count": 43,
            "word_count": 6,
            "extraction_method": "tesseract_ocr",
        }

        render_extraction(pdf_payload, document_id="doc_1", key_prefix="card_0")
        render_extraction(audio_payload, document_id="doc_2", key_prefix="card_1")
        render_extraction(image_payload, document_id="doc_3", key_prefix="card_2")

    at = AppTest.from_function(app).run()
    assert len(at.exception) == 0
    # Audio has 1 text_area, Image has 1 text_area
    assert len(at.text_area) == 2
    keys = {ta.key for ta in at.text_area}
    assert "card_1_doc_2_audio_transcript" in keys
    assert "card_2_doc_3_image_ocr" in keys


def test_document_inspector_is_extractable_logic():
    """
    Verifies that the is_extractable predicate in Document Inspector evaluates to True
    for all supported categories (documents, audio, images, videos) and False for unknown types.
    """
    from backend.ingestion.models import DocumentMetadata
    from backend.ingestion.text_extractor import (
        EXTRACTABLE_EXTENSIONS,
        IMAGE_EXTENSIONS,
        VIDEO_EXTENSIONS,
    )
    from backend.ingestion.audio_transcriber import AUDIO_EXTENSIONS

    def check_extractable(doc: DocumentMetadata) -> bool:
        return (
            doc.file_type in EXTRACTABLE_EXTENSIONS
            or doc.file_type in AUDIO_EXTENSIONS
            or doc.file_type in IMAGE_EXTENSIONS
            or doc.file_type in VIDEO_EXTENSIONS
            or doc.category in ("documents", "audio", "images", "videos")
        )

    # Documents
    doc_pdf = DocumentMetadata(document_id="d1", original_filename="a.pdf", safe_filename="a.pdf", category="documents", file_type=".pdf", mime_type="application/pdf", file_size_bytes=10, sha256_hash="h1", storage_path="p1", upload_timestamp="t1")
    assert check_extractable(doc_pdf) is True

    # Audio
    doc_audio = DocumentMetadata(document_id="d2", original_filename="a.mp3", safe_filename="a.mp3", category="audio", file_type=".mp3", mime_type="audio/mpeg", file_size_bytes=10, sha256_hash="h2", storage_path="p2", upload_timestamp="t2")
    assert check_extractable(doc_audio) is True

    # Images (The bug was that this previously evaluated to False!)
    doc_img = DocumentMetadata(document_id="d3", original_filename="scan.png", safe_filename="scan.png", category="images", file_type=".png", mime_type="image/png", file_size_bytes=10, sha256_hash="h3", storage_path="p3", upload_timestamp="t3")
    assert check_extractable(doc_img) is True

    # Videos
    doc_vid = DocumentMetadata(document_id="d4", original_filename="clip.mp4", safe_filename="clip.mp4", category="videos", file_type=".mp4", mime_type="video/mp4", file_size_bytes=10, sha256_hash="h4", storage_path="p4", upload_timestamp="t4")
    assert check_extractable(doc_vid) is True

    # Unknown category
    doc_other = DocumentMetadata(document_id="d5", original_filename="archive.zip", safe_filename="archive.zip", category="archives", file_type=".zip", mime_type="application/zip", file_size_bytes=10, sha256_hash="h5", storage_path="p5", upload_timestamp="t5")
    assert check_extractable(doc_other) is False


def test_inspector_loads_saved_ocr_without_reextraction(tmp_path):
    """
    Verifies that selecting an image document loads existing OCR extraction from SQLite
    metadata without performing OCR again or altering existing chunks and vector records.
    """
    from unittest.mock import patch
    from backend.ingestion.metadata import MetadataManager
    from backend.ingestion.storage import StorageManager
    from backend.ingestion.validator import FileValidator
    from backend.ingestion.pipeline import IngestionPipeline
    from backend.ingestion.models import DocumentMetadata

    db_path = tmp_path / "metadata.db"
    storage_dir = tmp_path / "storage"
    metadata_mgr = MetadataManager(db_path=db_path)
    storage_mgr = StorageManager(base_dir=storage_dir)
    validator = FileValidator()

    pipeline = IngestionPipeline(
        validator=validator,
        storage_manager=storage_mgr,
        metadata_manager=metadata_mgr,
        enable_multimodal=True,
    )

    doc_id = "doc_test_image_inspector"
    meta = DocumentMetadata(
        document_id=doc_id,
        original_filename="receipt.png",
        safe_filename="receipt.png",
        category="images",
        file_type=".png",
        mime_type="image/png",
        file_size_bytes=1024,
        sha256_hash="dummyhash12345",
        storage_path="images/receipt.png",
        upload_timestamp="2026-11-26T12:00:00",
        status="ocr_extracted",
    )
    metadata_mgr.save(meta)

    # Save initial OCR extraction result
    ocr_payload = {
        "file_type": "image",
        "category": "images",
        "raw_text": "Receipt #5521 Total: $99.50 Paid",
        "text": "Receipt #5521 Total: $99.50 Paid",
        "char_count": 32,
        "word_count": 5,
        "extraction_method": "tesseract_ocr",
        "error": None,
    }
    metadata_mgr.save_extraction(doc_id, ocr_payload, extraction_status="success")

    # Save initial chunk records
    initial_chunks = [
        {
            "chunk_id": f"{doc_id}_chunk_0",
            "document_id": doc_id,
            "text": "Receipt #5521 Total: $99.50 Paid",
            "chunk_index": 0,
            "page_number": 1,
            "media_type": "image",
            "estimated_tokens": 10,
        }
    ]
    metadata_mgr.save_chunks(doc_id, initial_chunks)

    # Patch extract_image so if it gets called, an exception is raised
    with patch("backend.ingestion.pipeline.extract_image", side_effect=AssertionError("extract_image was called!")):
        extracted, error = pipeline.get_or_reextract(doc_id)

    assert error is None
    assert extracted is not None
    assert extracted["raw_text"] == "Receipt #5521 Total: $99.50 Paid"
    assert extracted["extraction_method"] == "tesseract_ocr"
    assert extracted["char_count"] == 32
    assert extracted["word_count"] == 5

    # Verify chunks remain unchanged
    chunks = metadata_mgr.get_chunks(doc_id)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == f"{doc_id}_chunk_0"
    assert chunks[0]["text"] == "Receipt #5521 Total: $99.50 Paid"
