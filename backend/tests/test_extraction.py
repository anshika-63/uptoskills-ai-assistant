"""
Unit and integration tests for text extraction, audio transcription fallback,
and persistent extraction metadata storage.
"""

import io
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from backend.ingestion.audio_transcriber import (
    AUDIO_EXTENSIONS,
    get_audio_duration,
    is_transcription_available,
    transcribe_audio,
)
from backend.ingestion.metadata import MetadataManager
from backend.ingestion.pipeline import IngestionPipeline
from backend.ingestion.text_extractor import (
    EXTRACTABLE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    extract_docx,
    extract_pdf,
    extract_text,
    extract_txt,
)
from backend.tests.conftest import (
    create_sample_docx,
    create_sample_pdf,
    create_sample_txt,
)


def create_sample_wav(duration_seconds: float = 1.0, sample_rate: int = 8000) -> bytes:
    """Creates a minimal valid PCM WAV in-memory bytes."""
    buf = io.BytesIO()
    num_frames = int(duration_seconds * sample_rate)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Generate silence / simple tone frames
        data = struct.pack(f"<{num_frames}h", *([0] * num_frames))
        wf.writeframes(data)
    return buf.getvalue()


def test_supported_extensions_constants():
    assert ".pdf" in EXTRACTABLE_EXTENSIONS
    assert ".docx" in EXTRACTABLE_EXTENSIONS
    assert ".txt" in EXTRACTABLE_EXTENSIONS
    assert ".mp3" in AUDIO_EXTENSIONS
    assert ".wav" in AUDIO_EXTENSIONS
    assert EXTRACTABLE_EXTENSIONS == SUPPORTED_EXTENSIONS


def test_extract_txt(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello UpToSkills AI Assistant!\nSecond line of text.", encoding="utf-8")

    result = extract_txt(txt_file)
    assert result["file_type"] == "txt"
    assert "Hello UpToSkills" in result["raw_text"]
    assert result["char_count"] > 0
    assert result["word_count"] == 8


def test_extract_pdf(tmp_path):
    pdf_bytes = create_sample_pdf()
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(pdf_bytes)

    result = extract_pdf(pdf_file)
    assert result["file_type"] == "pdf"
    assert result["total_pages"] >= 1
    assert "pages" in result
    assert "raw_text" in result
    assert "char_count" in result
    assert "word_count" in result


def test_extract_docx(tmp_path):
    from docx import Document

    doc = Document()
    doc.add_paragraph("First paragraph for testing UpToSkills.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Header 1"
    table.rows[0].cells[1].text = "Header 2"

    docx_file = tmp_path / "sample.docx"
    doc.save(str(docx_file))

    result = extract_docx(docx_file)
    assert result["file_type"] == "docx"
    assert "paragraphs" in result
    assert "tables" in result
    assert "raw_text" in result
    assert result["char_count"] > 0
    assert len(result["paragraphs"]) == 1
    assert len(result["tables"]) == 1


def test_extract_text_dispatcher(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Testing dispatcher", encoding="utf-8")

    res = extract_text(txt_file)
    assert res["file_type"] == "txt"
    assert res["raw_text"] == "Testing dispatcher"

    # Unsupported format
    unsupported = tmp_path / "test.xyz"
    unsupported.write_text("random")
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text(unsupported)


def test_audio_duration_calculation(tmp_path):
    wav_bytes = create_sample_wav(duration_seconds=1.5, sample_rate=8000)
    wav_file = tmp_path / "sample.wav"
    wav_file.write_bytes(wav_bytes)

    duration = get_audio_duration(wav_file)
    assert abs(duration - 1.5) < 0.1

    # Non-existent file
    assert get_audio_duration(tmp_path / "missing.wav") == 0.0


def test_audio_transcription_engine_check():
    avail = is_transcription_available()
    assert isinstance(avail, bool)


def test_audio_transcription_fallback(tmp_path):
    wav_bytes = create_sample_wav(duration_seconds=0.5)
    wav_file = tmp_path / "speech.wav"
    wav_file.write_bytes(wav_bytes)

    result = transcribe_audio(wav_file)
    assert "category" in result
    assert result["category"] == "audio"
    assert result["file_type"] == "wav"
    assert "duration" in result
    assert "segments" in result

    # When whisper/faster-whisper is not installed, it returns success: False with helpful error
    if not is_transcription_available():
        assert result["success"] is False
        assert "not installed" in result["error"].lower()


def test_audio_transcription_mocked(tmp_path):
    wav_bytes = create_sample_wav(duration_seconds=2.0)
    wav_file = tmp_path / "speech_mock.wav"
    wav_file.write_bytes(wav_bytes)

    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.0
    mock_segment.text = "Welcome to the UpToSkills course."

    mock_info = MagicMock()
    mock_info.duration = 2.0
    mock_info.language = "en"
    mock_info.language_probability = 0.98

    mock_whisper_cls = MagicMock()
    mock_instance = mock_whisper_cls.return_value
    mock_instance.transcribe.return_value = ([mock_segment], mock_info)

    with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_whisper_cls)}):
        result = transcribe_audio(wav_file)
        assert result["success"] is True
        assert result["raw_text"] == "Welcome to the UpToSkills course."
        assert result["duration"] == 2.0
        assert result["language"] == "en"
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "Welcome to the UpToSkills course."


def test_metadata_extraction_persistence(metadata_manager):
    doc_id = "doc_ext_test_1"
    sample_content = {
        "file_type": "txt",
        "raw_text": "Sample persistent text.",
        "char_count": 23,
        "word_count": 3,
    }

    # Save extraction
    assert metadata_manager.save_extraction(
        document_id=doc_id,
        extraction_status="success",
        extracted_content=sample_content,
    )

    # Retrieve stats
    stats = metadata_manager.get_extraction_stats(doc_id)
    assert stats is not None
    assert stats["extraction_status"] == "success"
    assert stats["char_count"] == 23
    assert stats["word_count"] == 3

    # Retrieve full extraction payload
    payload = metadata_manager.get_extraction(doc_id)
    assert payload is not None
    assert payload["raw_text"] == "Sample persistent text."

    # Delete cascades
    assert metadata_manager.delete(doc_id)
    assert metadata_manager.get_extraction(doc_id) is None
    assert metadata_manager.get_extraction_stats(doc_id) is None


def test_pipeline_get_or_reextract(pipeline):
    txt_bytes = create_sample_txt()
    res = pipeline.process_file("course_notes.txt", txt_bytes)
    assert res.success
    assert res.document_id is not None

    # Retrieve from DB via get_or_reextract
    extracted, err = pipeline.get_or_reextract(res.document_id)
    assert err is None
    assert extracted is not None
    assert extracted["file_type"] == "txt"
    assert extracted["char_count"] > 0
