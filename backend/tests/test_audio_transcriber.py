"""
Unit tests for the audio transcription engine (backend.ingestion.audio_transcriber).
Verifies:
  - FFmpeg detection and locator
  - Audio preprocessing (conversion to 16kHz mono WAV)
  - Video audio extraction helper utility
  - Timestamp formatting
  - Whisper Speech-to-Text transcription, segmentation, and metadata
  - Error handling for missing files and unsupported formats
  - Singleton model manager caching
"""

import os
import tempfile
import wave
from pathlib import Path

import pytest

from backend.ingestion.audio_transcriber import (
    AUDIO_EXTENSIONS,
    WhisperModelManager,
    extract_audio_from_video,
    format_timestamp,
    get_ffmpeg_path,
    preprocess_audio_ffmpeg,
    transcribe_audio,
)
from backend.tests.conftest import create_sample_wav


# ─────────────────────────────────────────────
# FFmpeg & Utility Tests
# ─────────────────────────────────────────────

def test_get_ffmpeg_path():
    """FFmpeg must be discoverable on PATH or via imageio-ffmpeg."""
    ffmpeg_bin = get_ffmpeg_path()
    assert ffmpeg_bin is not None
    assert os.path.exists(ffmpeg_bin)


def test_format_timestamp():
    """Verify conversion of second floats to timestamp strings."""
    assert format_timestamp(0.0) == "00:00.00"
    assert format_timestamp(3.456) == "00:03.46"
    assert format_timestamp(65.2) == "01:05.20"
    assert format_timestamp(3665.5) == "01:01:05.50"
    assert format_timestamp(-5.0) == "00:00.00"


def test_preprocess_audio_ffmpeg():
    """FFmpeg must convert input audio into 16kHz mono PCM WAV."""
    # Generate 8kHz stereo WAV to test resampling & channel downmixing
    raw_wav = create_sample_wav(duration_s=0.5, freq=440.0, sample_rate=8000)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_f:
        in_f.write(raw_wav)
        in_path = Path(in_f.name)

    try:
        out_wav = preprocess_audio_ffmpeg(in_path)
        assert out_wav.exists()

        # Inspect properties using standard wave module
        with wave.open(str(out_wav), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # 16-bit PCM

        out_wav.unlink(missing_ok=True)
    finally:
        in_path.unlink(missing_ok=True)


def test_preprocess_audio_nonexistent_file_raises():
    """Attempting to preprocess missing file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preprocess_audio_ffmpeg("non_existent_audio.wav")


def test_extract_audio_from_video_nonexistent_raises():
    """Nonexistent video file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        extract_audio_from_video("missing_video.mp4")


def test_whisper_model_singleton():
    """WhisperModelManager must return the same cached model instance."""
    manager = WhisperModelManager.get_instance()
    model1 = manager.get_model(model_size="tiny", device="cpu", compute_type="int8")
    model2 = manager.get_model(model_size="tiny", device="cpu", compute_type="int8")
    assert model1 is model2


# ─────────────────────────────────────────────
# Transcription & Segments Tests
# ─────────────────────────────────────────────

def test_transcribe_audio_structure():
    """
    Transcribing a valid WAV audio file must yield the complete
    RAG-ready dictionary with raw_text, timestamps, and segments.
    """
    raw_wav = create_sample_wav(duration_s=1.0, freq=440.0, sample_rate=16000)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(raw_wav)
        wav_path = Path(f.name)

    try:
        result = transcribe_audio(wav_path, model_size="tiny")

        assert result["file_type"] == "wav"
        assert result["category"] == "audio"
        assert "raw_text" in result
        assert isinstance(result["raw_text"], str)
        assert "char_count" in result
        assert "word_count" in result
        assert "duration" in result
        assert result["duration"] >= 0.8
        assert "language" in result
        assert "segments" in result
        assert isinstance(result["segments"], list)
        assert "segment_count" in result

        # Verify segment schema if segments were generated
        for seg in result["segments"]:
            assert "id" in seg
            assert "start" in seg
            assert "end" in seg
            assert "text" in seg
            assert "timestamp" in seg
            assert "[" in seg["timestamp"] and "->" in seg["timestamp"]
    finally:
        wav_path.unlink(missing_ok=True)


def test_transcribe_audio_unsupported_extension_raises():
    """Attempting to transcribe non-audio file must raise ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not an audio file")
        txt_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Unsupported audio format"):
            transcribe_audio(txt_path)
    finally:
        txt_path.unlink(missing_ok=True)


def test_transcribe_audio_missing_file_raises():
    """Attempting to transcribe a non-existent path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        transcribe_audio("ghost_file.mp3")
