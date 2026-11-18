"""
Configuration settings for the RAG ingestion pipeline.
Provides configurable parameters for file storage, database paths,
allowed extensions, and file size limits across categories.
"""

import os
from pathlib import Path
from typing import Dict, Set

# Base repository and backend paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Storage configuration
DEFAULT_STORAGE_DIR = BACKEND_DIR / "storage" / "uploads"
DEFAULT_DB_PATH = BACKEND_DIR / "storage" / "metadata.db"

STORAGE_BASE_DIR = Path(os.getenv("RAG_STORAGE_DIR", str(DEFAULT_STORAGE_DIR)))
METADATA_DB_PATH = Path(os.getenv("RAG_METADATA_DB_PATH", str(DEFAULT_DB_PATH)))

# Category-specific directory mappings
CATEGORY_DIRECTORIES = {
    "documents": "documents",
    "images": "images",
    "videos": "videos",
}

# Allowed extensions categorized
ALLOWED_EXTENSIONS: Dict[str, Set[str]] = {
    "documents": {".pdf", ".docx", ".txt"},
    "images": {".png", ".jpg", ".jpeg", ".webp"},
    "videos": {".mp4", ".mov", ".avi", ".mkv", ".webm"},
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
}

# Configurable size limits (in Megabytes)
# Can be overridden using environment variables
DEFAULT_MAX_SIZES_MB = {
    "documents": int(os.getenv("MAX_DOC_SIZE_MB", "25")),
    "images": int(os.getenv("MAX_IMG_SIZE_MB", "15")),
    "videos": int(os.getenv("MAX_VID_SIZE_MB", "100")),
    "global": int(os.getenv("MAX_GLOBAL_SIZE_MB", "150")),
}

# Size limits converted to bytes
MAX_SIZES_BYTES = {
    k: v * 1024 * 1024 for k, v in DEFAULT_MAX_SIZES_MB.items()
}

# Enable or disable duplicate detection
ENABLE_DUPLICATE_DETECTION = os.getenv("ENABLE_DUPLICATE_DETECTION", "true").lower() in ("true", "1", "yes")
