"""
Unit tests for Video Audio Extraction and Transcription (Phase 5).
Tests FFmpeg audio extraction from video, Whisper video transcription,
timestamp generation, temporary file cleanup, and error handling for missing/unsupported/silent videos.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.ingestion.audio_transcriber import (
    VIDEO_EXTENSIONS,
    extract_audio_from_video,
    format_timestamp,
    transcribe_video,
)
from backend.tests.conftest import create_real_video_mp4


def test_extract_audio_from_real_video(tmp_path):
    """FFmpeg extracts 16kHz mono audio from a genuine MP4 video file."""
    video_bytes = create_real_video_mp4(duration_s=0.5)
    video_path = tmp_path / "sample_video.mp4"
    video_path.write_bytes(video_bytes)

    wav_out = extract_audio_from_video(video_path)
    try:
        assert wav_out.exists()
        assert wav_out.suffix.lower() == ".wav"
        assert wav_out.stat().st_size > 0
    finally:
        if wav_out.exists():
            wav_out.unlink()


def test_extract_audio_missing_video_raises():
    """extract_audio_from_video raises FileNotFoundError when video is missing."""
    with pytest.raises(FileNotFoundError):
        extract_audio_from_video("nonexistent_video.mp4")


def test_transcribe_video_mocked(tmp_path):
    """transcribe_video extracts audio, calls Whisper, formats timestamps, and cleans up temp WAV."""
    video_bytes = create_real_video_mp4(duration_s=0.5)
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(video_bytes)

    # Mock Whisper segment generator
    mock_segment = MagicMock()
    mock_segment.id = 0
    mock_segment.start = 0.0
    mock_segment.end = 0.5
    mock_segment.text = "Welcome to the UpToSkills lecture."

    mock_info = MagicMock()
    mock_info.duration = 0.5
    mock_info.language = "en"
    mock_info.language_probability = 0.98

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    with patch("backend.ingestion.audio_transcriber.WhisperModelManager.get_instance") as mock_mgr_get:
        mock_mgr_instance = MagicMock()
        mock_mgr_instance.get_model.return_value = mock_model
        mock_mgr_get.return_value = mock_mgr_instance

        result = transcribe_video(video_path)

    assert result["success"] is True
    assert result["category"] == "videos"
    assert result["media_type"] == "video"
    assert result["file_type"] == "mp4"
    assert "UpToSkills lecture" in result["raw_text"]
    assert result["duration"] == 0.5
    assert result["segment_count"] == 1
    assert len(result["segments"]) == 1

    seg0 = result["segments"][0]
    assert seg0["start"] == 0.0
    assert seg0["end"] == 0.5
    assert seg0["timestamp"] == "[00:00.00 -> 00:00.50]"


def test_transcribe_video_missing_file_raises():
    """transcribe_video raises FileNotFoundError when video path does not exist."""
    with pytest.raises(FileNotFoundError):
        transcribe_video("missing.mp4")


def test_transcribe_video_unsupported_format_raises(tmp_path):
    """transcribe_video raises ValueError when file format is not a supported video."""
    bad_file = tmp_path / "song.mp3"
    bad_file.write_bytes(b"dummy_content")

    with pytest.raises(ValueError):
        transcribe_video(bad_file)


def test_transcribe_video_extraction_failure_handled(tmp_path):
    """transcribe_video returns failure dictionary when audio extraction fails without crashing."""
    corrupt_video = tmp_path / "corrupt.mp4"
    corrupt_video.write_bytes(b"INVALID_VIDEO_HEADER")

    result = transcribe_video(corrupt_video)

    assert result["success"] is False
    assert result["category"] == "videos"
    assert result["error"] is not None
    assert result["raw_text"] == ""
    assert result["segment_count"] == 0
