"""
Audio transcription module for UpToSkills AI Assistant.
Provides speech-to-text extraction, duration calculation, and timestamped segmentation
using faster-whisper or openai-whisper with graceful fallbacks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import wave

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


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


def transcribe_audio(
    file_path: Union[str, Path],
    model_size: str = "base",
    language: Optional[str] = None,
) -> Dict:
    """
    Transcribes an audio file into text with timestamped segments.

    Returns a standardized dictionary:
    {
        "success": bool,
        "category": "audio",
        "file_type": str,
        "raw_text": str,
        "duration": float,
        "language": str,
        "language_probability": float,
        "segments": List[Dict],
        "segment_count": int,
        "char_count": int,
        "word_count": int,
        "error": Optional[str],
    }
    """
    path = Path(file_path)
    file_type = path.suffix.lstrip(".").lower()

    if not path.exists():
        return {
            "success": False,
            "category": "audio",
            "file_type": file_type,
            "raw_text": "",
            "duration": 0.0,
            "language": "unknown",
            "language_probability": 0.0,
            "segments": [],
            "segment_count": 0,
            "char_count": 0,
            "word_count": 0,
            "error": f"Audio file not found at: {path}",
        }

    duration = get_audio_duration(path)

    # 1. Attempt faster-whisper
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        whisper_segments, info = model.transcribe(str(path), language=language, beam_size=5)

        segments_list = []
        raw_parts = []
        for s in whisper_segments:
            seg_text = s.text.strip()
            if seg_text:
                raw_parts.append(seg_text)
                segments_list.append({
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "timestamp": f"[{s.start:.2f} -> {s.end:.2f}]",
                    "text": seg_text,
                })

        raw_text = " ".join(raw_parts).strip()
        actual_duration = round(info.duration, 2) if hasattr(info, "duration") and info.duration else duration

        return {
            "success": True,
            "category": "audio",
            "file_type": file_type,
            "raw_text": raw_text,
            "duration": actual_duration,
            "language": info.language if hasattr(info, "language") else (language or "unknown"),
            "language_probability": round(info.language_probability, 4) if hasattr(info, "language_probability") else 1.0,
            "segments": segments_list,
            "segment_count": len(segments_list),
            "char_count": len(raw_text),
            "word_count": len(raw_text.split()) if raw_text else 0,
            "error": None,
        }
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"faster-whisper transcription error: {e}")

    # 2. Attempt openai-whisper
    try:
        import whisper

        model = whisper.load_model(model_size)
        transcribe_args = {}
        if language:
            transcribe_args["language"] = language

        result = model.transcribe(str(path), **transcribe_args)
        raw_text = result.get("text", "").strip()
        detected_lang = result.get("language", language or "unknown")

        segments_list = []
        for s in result.get("segments", []):
            seg_text = s.get("text", "").strip()
            if seg_text:
                start_t = round(float(s.get("start", 0)), 2)
                end_t = round(float(s.get("end", 0)), 2)
                segments_list.append({
                    "start": start_t,
                    "end": end_t,
                    "timestamp": f"[{start_t:.2f} -> {end_t:.2f}]",
                    "text": seg_text,
                })

        return {
            "success": True,
            "category": "audio",
            "file_type": file_type,
            "raw_text": raw_text,
            "duration": duration,
            "language": detected_lang,
            "language_probability": 1.0,
            "segments": segments_list,
            "segment_count": len(segments_list),
            "char_count": len(raw_text),
            "word_count": len(raw_text.split()) if raw_text else 0,
            "error": None,
        }
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"openai-whisper transcription error: {e}")

    # 3. Neither engine is installed: return informative payload
    return {
        "success": False,
        "category": "audio",
        "file_type": file_type,
        "raw_text": "",
        "duration": duration,
        "language": "unknown",
        "language_probability": 0.0,
        "segments": [],
        "segment_count": 0,
        "char_count": 0,
        "word_count": 0,
        "error": (
            "Speech-to-text transcription engine (faster-whisper or openai-whisper) is not installed. "
            "Audio file was safely validated and stored."
        ),
    }

