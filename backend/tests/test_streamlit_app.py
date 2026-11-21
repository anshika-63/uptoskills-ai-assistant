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
