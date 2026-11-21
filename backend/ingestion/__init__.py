"""
RAG Ingestion Pipeline Package.
"""

from backend.ingestion.models import (
    DocumentMetadata,
    FileCategory,
    IngestionResult,
    ValidationResult,
    ValidationStatus,
)
from backend.ingestion.validator import FileValidator
from backend.ingestion.storage import StorageManager
from backend.ingestion.metadata import MetadataManager
from backend.ingestion.pipeline import IngestionPipeline
from backend.ingestion.text_extractor import extract_text, EXTRACTABLE_EXTENSIONS
from backend.ingestion.audio_transcriber import (
    AUDIO_EXTENSIONS,
    WhisperModelManager,
    extract_audio_from_video,
    preprocess_audio_ffmpeg,
    transcribe_audio,
)

__all__ = [
    "DocumentMetadata",
    "FileCategory",
    "IngestionResult",
    "ValidationResult",
    "ValidationStatus",
    "FileValidator",
    "StorageManager",
    "MetadataManager",
    "IngestionPipeline",
    "extract_text",
    "EXTRACTABLE_EXTENSIONS",
    "AUDIO_EXTENSIONS",
    "WhisperModelManager",
    "extract_audio_from_video",
    "preprocess_audio_ffmpeg",
    "transcribe_audio",
]
