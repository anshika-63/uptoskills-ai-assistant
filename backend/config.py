"""
Configuration settings for the RAG ingestion pipeline.
Provides configurable parameters for file storage, database paths,
allowed extensions, and file size limits across categories.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger("rag.config")

# Base repository and backend paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent


def load_environment() -> List[Path]:
    """
    Discovers and loads .env files from project directories regardless of cwd.
    Searches PROJECT_ROOT, BACKEND_DIR, PROJECT_ROOT.parent, and current working directory.
    Returns list of successfully loaded .env file paths.
    """
    loaded: List[Path] = []
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed; relying on existing environment variables.")
        return loaded

    candidate_dirs = [
        PROJECT_ROOT,
        BACKEND_DIR,
        PROJECT_ROOT.parent,
        Path.cwd(),
    ]
    seen: Set[Path] = set()
    for directory in candidate_dirs:
        try:
            env_file = (directory / ".env").resolve()
            if env_file.is_file() and env_file not in seen:
                seen.add(env_file)
                load_dotenv(dotenv_path=env_file, override=False)
                loaded.append(env_file)
        except Exception as exc:
            logger.debug("Error checking .env in %s: %s", directory, exc)
    return loaded


LOADED_ENV_FILES = load_environment()


def resolve_tesseract_cmd() -> str:
    """
    Resolves the Tesseract executable path from environment, config, or well-known locations.
    Handles relative paths, quoted paths, and ensures path existence.
    """
    raw_cmd = os.getenv("TESSERACT_CMD", "").strip("\"' ")
    if raw_cmd:
        # Check as absolute or existing path directly
        candidate = Path(raw_cmd)
        if candidate.is_file():
            return str(candidate.resolve())

        # Check relative to project directories
        for base in (PROJECT_ROOT, BACKEND_DIR, PROJECT_ROOT.parent, Path.cwd()):
            resolved = (base / raw_cmd).resolve()
            if resolved.is_file():
                return str(resolved)

        return raw_cmd  # Return raw path so caller can report specific missing target

    # Check parent directory (e.g. E:/RAG/tesseract.exe)
    parent_tesseract = (PROJECT_ROOT.parent / "tesseract.exe").resolve()
    if parent_tesseract.is_file():
        return str(parent_tesseract)

    # Check project root directory
    proj_tesseract = (PROJECT_ROOT / "tesseract.exe").resolve()
    if proj_tesseract.is_file():
        return str(proj_tesseract)

    # Check system PATH
    which_path = shutil.which("tesseract")
    if which_path:
        return str(Path(which_path).resolve())

    # Standard Windows install locations
    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for p in candidate_paths:
        if p and Path(p).is_file():
            return str(Path(p).resolve())

    return ""


# Storage configuration
DEFAULT_STORAGE_DIR = BACKEND_DIR / "storage" / "uploads"
DEFAULT_DB_PATH = BACKEND_DIR / "storage" / "metadata.db"
DEFAULT_EXTRACTED_DIR = BACKEND_DIR / "storage" / "extracted"
DEFAULT_TRANSCRIPTS_DIR = BACKEND_DIR / "storage" / "transcripts"

STORAGE_BASE_DIR = Path(os.getenv("RAG_STORAGE_DIR", str(DEFAULT_STORAGE_DIR)))
METADATA_DB_PATH = Path(os.getenv("RAG_METADATA_DB_PATH", str(DEFAULT_DB_PATH)))
EXTRACTED_DIR = Path(os.getenv("RAG_EXTRACTED_DIR", str(DEFAULT_EXTRACTED_DIR)))
TRANSCRIPTS_DIR = Path(os.getenv("RAG_TRANSCRIPTS_DIR", str(DEFAULT_TRANSCRIPTS_DIR)))

# Category-specific directory mappings
CATEGORY_DIRECTORIES = {
    "documents": "documents",
    "images": "images",
    "videos": "videos",
    "audio": "audio",
}

# Allowed extensions categorized
ALLOWED_EXTENSIONS: Dict[str, Set[str]] = {
    "documents": {".pdf", ".docx", ".txt"},
    "images": {".png", ".jpg", ".jpeg", ".webp"},
    "videos": {".mp4", ".mov", ".avi", ".mkv", ".webm"},
    "audio": {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"},
}

# Inverted mapping: extension -> category
EXTENSION_TO_CATEGORY: Dict[str, str] = {}
for category, extensions in ALLOWED_EXTENSIONS.items():
    for ext in extensions:
        EXTENSION_TO_CATEGORY[ext] = category

# MIME types mapping for standard verification
ALLOWED_MIME_TYPES: Dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}

# Configurable size limits (in Megabytes)
# Can be overridden using environment variables
DEFAULT_MAX_SIZES_MB = {
    "documents": int(os.getenv("MAX_DOC_SIZE_MB", "25")),
    "images": int(os.getenv("MAX_IMG_SIZE_MB", "15")),
    "videos": int(os.getenv("MAX_VID_SIZE_MB", "100")),
    "audio": int(os.getenv("MAX_AUDIO_SIZE_MB", "50")),
    "global": int(os.getenv("MAX_GLOBAL_SIZE_MB", "150")),
}

# Size limits converted to bytes
MAX_SIZES_BYTES = {
    k: v * 1024 * 1024 for k, v in DEFAULT_MAX_SIZES_MB.items()
}

# Enable or disable duplicate detection
ENABLE_DUPLICATE_DETECTION = os.getenv("ENABLE_DUPLICATE_DETECTION", "true").lower() in ("true", "1", "yes")

# Whisper Speech-to-Text configuration
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Phase 4: Vector Storage & Embedding Configuration
DEFAULT_CHROMA_DIR = BACKEND_DIR / "storage" / "chroma"
CHROMA_PERSIST_DIR = Path(os.getenv("RAG_CHROMA_DIR", str(DEFAULT_CHROMA_DIR)))
CHROMA_COLLECTION_NAME = os.getenv("RAG_CHROMA_COLLECTION", "uptoskills_rag")

EMBEDDING_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("RAG_EMBEDDING_DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_NORMALIZE = os.getenv("RAG_EMBEDDING_NORMALIZE", "true").lower() in ("true", "1", "yes")

# Phase 5: Multimodal Ingestion (OCR & Video Transcription)
TESSERACT_CMD = resolve_tesseract_cmd()
ENABLE_MULTIMODAL = os.getenv("RAG_ENABLE_MULTIMODAL", "false").lower() in ("true", "1", "yes")
