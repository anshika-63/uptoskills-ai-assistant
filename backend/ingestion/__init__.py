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
from backend.ingestion.text_extractor import (
    DOCUMENT_EXTENSIONS,
    EXTRACTABLE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS as EXTRACTOR_VIDEO_EXTENSIONS,
    extract_image,
    extract_pdf,
    extract_text,
    extract_video,
    configure_tesseract,
    get_tesseract_cmd,
    is_ocr_available,
    is_scanned_pdf,
    reset_ocr_engine,
    set_ocr_engine,
)
from backend.ingestion.audio_transcriber import (
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    WhisperModelManager,
    extract_audio_from_video,
    preprocess_audio_ffmpeg,
    transcribe_audio,
    transcribe_video,
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
    "extract_image",
    "extract_pdf",
    "extract_video",
    "configure_tesseract",
    "get_tesseract_cmd",
    "is_ocr_available",
    "is_scanned_pdf",
    "set_ocr_engine",
    "reset_ocr_engine",
    "EXTRACTABLE_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "DOCUMENT_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
    "AUDIO_EXTENSIONS",
    "WhisperModelManager",
    "extract_audio_from_video",
    "preprocess_audio_ffmpeg",
    "transcribe_audio",
    "transcribe_video",
    "EmbeddingService",
    "EmbeddingModelManager",
    "VectorStoreManager",
]
from backend.ingestion.embeddings import EmbeddingModelManager, EmbeddingService
from backend.ingestion.vector_store import VectorStoreManager
