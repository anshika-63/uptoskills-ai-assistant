"""
Audio transcription engine for the RAG ingestion pipeline.
Handles Speech-to-Text transcription using faster-whisper,
FFmpeg audio preprocessing, segmenting with timestamps, and is reusable
for future video-audio extraction workflows.

Also provides utility helpers (is_transcription_available, get_audio_duration)
for lightweight availability checks and duration estimation.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)

logger = logging.getLogger("rag.audio_transcriber")

# PyAV compatibility shim: PyAV >= 14 removed metadata_errors parameter from av.open,
# which causes faster-whisper's decode_audio to raise TypeError.
try:
    import av
    _orig_av_open = av.open

    def _safe_av_open(*args: Any, **kwargs: Any) -> Any:
        try:
            return _orig_av_open(*args, **kwargs)
        except TypeError as err:
            if "metadata_errors" in str(err) and "metadata_errors" in kwargs:
                kwargs.pop("metadata_errors", None)
                return _orig_av_open(*args, **kwargs)
            raise

    av.open = _safe_av_open
except ImportError:
    pass

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def get_ffmpeg_path() -> str:
    """
    Locates the FFmpeg executable.
    Prioritizes system PATH, then falls back to imageio-ffmpeg bundled binary.
    """
    path_on_sys = shutil.which("ffmpeg")
    if path_on_sys:
        return path_on_sys

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and os.path.exists(ffmpeg_exe):
            return ffmpeg_exe
    except ImportError:
        pass

    raise RuntimeError(
        "FFmpeg is required for audio preprocessing but was not found on PATH "
        "or through imageio-ffmpeg."
    )


def format_timestamp(seconds: float) -> str:
    """
    Formats a duration in seconds into MM:SS.ss (or HH:MM:SS.ss if >= 1 hour).
    Examples:
        3.45  -> '00:03.45'
        75.2  -> '01:15.20'
        3661  -> '01:01:01.00'
    """
    if seconds < 0:
        seconds = 0.0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def preprocess_audio_ffmpeg(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = 16000,
) -> Path:
    """
    Converts and standardizes an input audio file to 16kHz mono 16-bit PCM WAV.
    This is the ideal format for Whisper acoustic feature extraction.

    Args:
        input_path: Path to the source audio file.
        output_path: Destination path for the converted WAV. If None, a temporary
                     file is created.
        sample_rate: Target audio sampling rate (default 16000 Hz).

    Returns:
        Path to the preprocessed WAV file.
    """
    input_p = Path(input_path).resolve()
    if not input_p.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_p}")

    if output_path is None:
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tf.close()
        out_p = Path(tf.name)
    else:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = get_ffmpeg_path()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(input_p),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out_p),
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.debug("FFmpeg audio preprocessing succeeded: %s -> %s", input_p, out_p)
        return out_p
    except subprocess.CalledProcessError as exc:
        err_msg = exc.stderr.decode("utf-8", errors="replace")
        logger.error("FFmpeg preprocessing failed: %s", err_msg)
        raise RuntimeError(f"FFmpeg audio conversion failed: {err_msg}")


def extract_audio_from_video(
    video_path: Union[str, Path],
    output_audio_path: Optional[Union[str, Path]] = None,
    sample_rate: int = 16000,
) -> Path:
    """
    Reusable utility: Extracts audio track from a video file into 16kHz mono WAV.
    Reserved for future video processing pipelines.

    Args:
        video_path: Path to the video file.
        output_audio_path: Destination path for extracted audio.
        sample_rate: Target audio sampling rate (default 16000 Hz).

    Returns:
        Path to the extracted audio WAV file.
    """
    vid_p = Path(video_path).resolve()
    if not vid_p.exists():
        raise FileNotFoundError(f"Video file not found: {vid_p}")

    if output_audio_path is None:
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tf.close()
        out_p = Path(tf.name)
    else:
        out_p = Path(output_audio_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = get_ffmpeg_path()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(vid_p),
        "-vn",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out_p),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return out_p
    except subprocess.CalledProcessError as exc:
        err_msg = exc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Video audio extraction failed: {err_msg}")


class WhisperModelManager:
    """
    Lazy-loading singleton for faster-whisper WhisperModel instances.
    Avoids expensive reloads on every transcription call.
    """
    _instance: Optional["WhisperModelManager"] = None
    _models: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "WhisperModelManager":
        if cls._instance is None:
            cls._instance = WhisperModelManager()
        return cls._instance

    def get_model(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        from faster_whisper import WhisperModel

        target_size = model_size or WHISPER_MODEL_SIZE
        target_device = device or WHISPER_DEVICE
        target_compute = compute_type or WHISPER_COMPUTE_TYPE
        cache_key = f"{target_size}_{target_device}_{target_compute}"

        # If WhisperModel is not a concrete type (e.g. MagicMock in unit tests),
        # do not use or populate the cache — instantiate directly and return.
        if not isinstance(WhisperModel, type):
            return WhisperModel(
                target_size,
                device=target_device,
                compute_type=target_compute,
            )

        cached = self._models.get(cache_key)
        if cached is not None and not isinstance(cached, WhisperModel):
            del self._models[cache_key]
            cached = None

        if cached is None:
            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute=%s)...",
                target_size, target_device, target_compute,
            )
            self._models[cache_key] = WhisperModel(
                target_size,
                device=target_device,
                compute_type=target_compute,
            )
            logger.info("faster-whisper model '%s' loaded successfully.", target_size)

        return self._models[cache_key]


# ---------------------------------------------------------------------------
# Utility helpers (from origin/main — additive, no conflicts with HEAD)
# ---------------------------------------------------------------------------

def is_transcription_available() -> bool:
    """Checks if a supported Whisper transcription engine is installed."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        pass

    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        pass

    return False


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """
    Attempts to calculate the duration of an audio file in seconds.
    Supports standard WAV files directly via the standard library,
    and falls back to mutagen or estimates for other formats.
    """
    path = Path(file_path)
    if not path.exists():
        return 0.0

    # Try standard library wave module for WAV files
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return round(frames / float(rate), 2)
        except Exception:
            pass

    # Try mutagen if installed
    try:
        import mutagen
        audio = mutagen.File(str(path))
        if audio and audio.info and hasattr(audio.info, "length"):
            return round(float(audio.info.length), 2)
    except ImportError:
        pass
    except Exception:
        pass

    return 0.0


# ---------------------------------------------------------------------------
# Primary transcription entry point
# ---------------------------------------------------------------------------

def transcribe_audio(
    file_path: Union[str, Path],
    model_size: Optional[str] = None,
    language: Optional[str] = None,
    preprocess: bool = True,
) -> Dict[str, Any]:
    """
    Transcribes an audio file into structured segments with timestamps and full text.

    Args:
        file_path: Path to the audio file (MP3, WAV, M4A, AAC, OGG, FLAC).
        model_size: Optional Whisper model size (tiny, base, small, medium, large).
                    Defaults to WHISPER_MODEL_SIZE from config.
        language: Optional 2-letter language code (e.g. 'en') to bypass detection.
        preprocess: Whether to convert audio to 16kHz mono WAV via FFmpeg first (default True).

    Returns:
        Structured dictionary containing:
            - success: bool
            - file_type: str (e.g. 'mp3', 'wav')
            - category: 'audio'
            - raw_text: str (full text transcript)
            - char_count: int
            - word_count: int
            - duration: float (seconds)
            - language: str (detected or specified)
            - language_probability: float
            - segments: List[Dict[str, Any]] (id, start, end, text, timestamp)
            - segment_count: int
            - model_used: str
            - error: Optional[str]
    """
    audio_path = Path(file_path).resolve()

    # Raise for missing file (FileNotFoundError) — existing tests expect this
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ext = audio_path.suffix.lower()
    # Raise for unsupported extension (ValueError) — existing tests expect this
    if ext not in AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: '{ext}'. Supported: {sorted(AUDIO_EXTENSIONS)}"
        )

    temp_wav_path: Optional[Path] = None
    input_for_whisper = audio_path

    # Preprocessing with FFmpeg ensures 16kHz mono format for high accuracy
    if preprocess:
        try:
            temp_wav_path = preprocess_audio_ffmpeg(audio_path)
            input_for_whisper = temp_wav_path
        except Exception as prep_err:
            logger.warning(
                "FFmpeg preprocessing failed, attempting direct transcription: %s",
                prep_err,
            )
            input_for_whisper = audio_path

    try:
        model = WhisperModelManager.get_instance().get_model(model_size=model_size)
        segments_generator, info = model.transcribe(
            str(input_for_whisper),
            beam_size=5,
            language=language,
        )

        segments_list: List[Dict[str, Any]] = []
        raw_text_parts: List[str] = []

        for seg in segments_generator:
            text_clean = seg.text.strip()
            if text_clean:
                raw_text_parts.append(text_clean)

            start_sec = round(float(seg.start), 2)
            end_sec = round(float(seg.end), 2)
            segments_list.append(
                {
                    "id": seg.id,
                    "start": start_sec,
                    "end": end_sec,
                    "text": text_clean,
                    "timestamp": f"[{format_timestamp(start_sec)} -> {format_timestamp(end_sec)}]",
                }
            )

        raw_text = " ".join(raw_text_parts).strip()
        duration_sec = round(float(getattr(info, "duration", 0.0) or 0.0), 2)
        lang_detected = getattr(info, "language", "unknown")
        lang_prob = round(float(getattr(info, "language_probability", 0.0) or 0.0), 3)

        return {
            "success": True,
            "file_type": ext.lstrip("."),
            "category": "audio",
            "raw_text": raw_text,
            "char_count": len(raw_text),
            "word_count": len(raw_text.split()) if raw_text else 0,
            "duration": duration_sec,
            "language": lang_detected,
            "language_probability": lang_prob,
            "segments": segments_list,
            "segment_count": len(segments_list),
            "model_used": model_size or WHISPER_MODEL_SIZE,
            "error": None,
        }

    except Exception as exc:
        error_msg = str(exc)
        logger.error("Audio transcription failed: %s", error_msg)
        return {
            "success": False,
            "file_type": ext.lstrip("."),
            "category": "audio",
            "raw_text": "",
            "char_count": 0,
            "word_count": 0,
            "duration": 0.0,
            "language": "unknown",
            "language_probability": 0.0,
            "segments": [],
            "segment_count": 0,
            "model_used": model_size or WHISPER_MODEL_SIZE,
            "error": error_msg,
        }

    finally:
        # Clean up temporary preprocessed file if created
        if temp_wav_path and temp_wav_path.exists():
            try:
                temp_wav_path.unlink()
            except Exception:
                pass
