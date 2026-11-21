"""
Validation engine for the RAG document ingestion pipeline.
Handles file extension verification, magic bytes/signature checks,
empty file and corruption detection, safe filename sanitization,
SHA-256 calculation, and category-based size limits.
"""

import hashlib
import io
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image

from backend.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    EXTENSION_TO_CATEGORY,
    MAX_SIZES_BYTES,
    DEFAULT_MAX_SIZES_MB,
)
from backend.ingestion.models import FileCategory, ValidationResult, ValidationStatus


class FileValidator:
    """Validates files for format, size, corruption, and safe naming."""

    def __init__(
        self,
        allowed_extensions: Optional[Dict[str, set]] = None,
        size_limits_bytes: Optional[Dict[str, int]] = None,
    ):
        self.allowed_extensions = allowed_extensions or ALLOWED_EXTENSIONS
        self.size_limits_bytes = size_limits_bytes or MAX_SIZES_BYTES
        self.extension_to_category = EXTENSION_TO_CATEGORY

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitizes an untrusted filename to prevent path traversal and filesystem issues.
        Removes directory delimiters, Windows reserved characters, control characters,
        and enforces safe naming standards.
        """
        if not filename or not filename.strip():
            return "unnamed_file.bin"

        # Extract only the base name (strips path traversals like ../ or C:\)
        clean_name = os.path.basename(filename.strip())

        # Also strip any lingering forward or backward slashes
        clean_name = clean_name.replace("/", "").replace("\\", "")

        # Replace null bytes
        clean_name = clean_name.replace("\x00", "")

        # Windows invalid characters: < > : " / \ | ? *
        clean_name = re.sub(r'[<>:"/\\|?*]', "_", clean_name)

        # Replace control characters (ASCII 0-31)
        clean_name = re.sub(r"[\x00-\x1f\x7f]", "", clean_name)

        # Remove trailing and leading dots or spaces (Windows issues)
        clean_name = clean_name.strip(". ")

        # Prevent empty filename after cleaning
        if not clean_name:
            clean_name = "unnamed_file"

        # Split stem and suffix
        path_obj = Path(clean_name)
        stem = path_obj.stem
        suffix = path_obj.suffix

        # Limit stem length to 120 characters to prevent path length overflow
        if len(stem) > 120:
            stem = stem[:120]

        sanitized = f"{stem}{suffix}" if suffix else stem
        return sanitized

    @staticmethod
    def compute_sha256(content: bytes) -> str:
        """Computes the SHA-256 hexadecimal digest of byte content."""
        hasher = hashlib.sha256()
        hasher.update(content)
        return hasher.hexdigest()

    def get_category_for_extension(self, extension: str) -> Optional[str]:
        """Returns the category for a given lowercase extension (e.g. '.pdf' -> 'documents')."""
        norm_ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        return self.extension_to_category.get(norm_ext)

    def validate_size(self, size_bytes: int, category: str) -> Tuple[bool, Optional[str]]:
        """
        Checks whether file size exceeds configured category limit or global limit.
        Returns (is_valid, error_message).
        """
        cat_limit = self.size_limits_bytes.get(category, self.size_limits_bytes.get("global", 150 * 1024 * 1024))
        global_limit = self.size_limits_bytes.get("global", 150 * 1024 * 1024)

        if size_bytes > cat_limit:
            limit_mb = cat_limit / (1024 * 1024)
            actual_mb = size_bytes / (1024 * 1024)
            return False, f"File size ({actual_mb:.2f} MB) exceeds category '{category}' limit of {limit_mb:.1f} MB."

        if size_bytes > global_limit:
            global_mb = global_limit / (1024 * 1024)
            actual_mb = size_bytes / (1024 * 1024)
            return False, f"File size ({actual_mb:.2f} MB) exceeds global size limit of {global_mb:.1f} MB."

        return True, None

    def validate_file_content(self, content: bytes, extension: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Verifies magic bytes and detects empty or corrupted files.
        Returns (is_valid, detected_mime, error_message).
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"

        # 1. Empty file detection
        if len(content) == 0:
            return False, None, "File is empty (0 bytes). Ingestion requires non-empty files."

        expected_mime = ALLOWED_MIME_TYPES.get(ext, "application/octet-stream")

        # 2. Category-specific content & corruption inspection
        try:
            if ext == ".pdf":
                if not content.startswith(b"%PDF-"):
                    return False, None, "Corrupted or invalid PDF: missing '%PDF-' header signature."
                return True, expected_mime, None

            elif ext == ".docx":
                if not zipfile.is_zipfile(io.BytesIO(content)):
                    return False, None, "Corrupted or invalid DOCX: file is not a valid zip archive."
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    names = zf.namelist()
                    # A valid docx contains [Content_Types].xml and word/document.xml (or word/ directory)
                    has_content_types = "[Content_Types].xml" in names
                    has_word = any(name.startswith("word/") for name in names)
                    if not (has_content_types and has_word):
                        return False, None, "Corrupted DOCX: archive missing Word document structures."
                return True, expected_mime, None

            elif ext == ".txt":
                # Check for null bytes (indicative of binary file masquerading as text)
                if b"\x00" in content[:4096]:
                    return False, None, "Invalid text file: binary null bytes detected."
                # Verify text decodability
                try:
                    content.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        content.decode("latin-1")
                    except Exception:
                        return False, None, "Corrupted or unsupported text encoding."
                return True, expected_mime, None

            elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
                # Magic bytes check first
                if ext == ".png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
                    return False, None, "Corrupted PNG: invalid PNG magic signature."
                elif ext in {".jpg", ".jpeg"} and not content.startswith(b"\xff\xd8\xff"):
                    return False, None, "Corrupted JPEG: invalid SOI marker."
                elif ext == ".webp":
                    if not (content.startswith(b"RIFF") and len(content) >= 12 and content[8:12] == b"WEBP"):
                        return False, None, "Corrupted WEBP: missing RIFF/WEBP signature."

                # Deep integrity check with Pillow
                try:
                    with Image.open(io.BytesIO(content)) as img:
                        img.verify()
                except Exception as img_err:
                    return False, None, f"Corrupted image file: {str(img_err)}"

                return True, expected_mime, None

            elif ext in {".mp4", ".mov"}:
                # Check ISO Base Media File Format box structure (ftyp)
                if len(content) < 12:
                    return False, None, "Corrupted video: file too small for video header."
                box_type = content[4:8]
                if ext == ".mp4" and box_type != b"ftyp":
                    return False, None, "Corrupted MP4: missing 'ftyp' atom header."
                elif ext == ".mov" and box_type not in (b"ftyp", b"moov", b"mdat", b"wide", b"skip"):
                    return False, None, "Corrupted MOV: missing valid QuickTime atom header."
                return True, expected_mime, None

            elif ext == ".avi":
                if not (content.startswith(b"RIFF") and len(content) >= 12 and content[8:12] == b"AVI "):
                    return False, None, "Corrupted AVI: missing RIFF/AVI chunk header."
                return True, expected_mime, None

            elif ext in {".mkv", ".webm"}:
                # EBML header: 0x1A 0x45 0xDF 0xA3
                if not content.startswith(b"\x1a\x45\xdf\xa3"):
                    return False, None, f"Corrupted {ext[1:].upper()}: missing EBML signature."
                return True, expected_mime, None

            elif ext == ".mp3":
                # Check for ID3v2 tag or MPEG sync word (0xFFE0 / 0xFFFB etc.)
                has_id3 = content.startswith(b"ID3")
                has_sync = (
                    len(content) >= 2
                    and content[0] == 0xFF
                    and (content[1] & 0xE0) == 0xE0
                )
                has_id3_or_sync_in_prefix = (
                    b"ID3" in content[:1024]
                    or any(content[i:i+2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa") for i in range(min(512, len(content) - 1)))
                )
                if not (has_id3 or has_sync or has_id3_or_sync_in_prefix):
                    return False, None, "Corrupted MP3: missing ID3 tag or MPEG audio sync header."
                return True, expected_mime, None

            elif ext == ".wav":
                if not (content.startswith(b"RIFF") and len(content) >= 12 and content[8:12] == b"WAVE"):
                    return False, None, "Corrupted WAV: missing RIFF/WAVE header chunk."
                return True, expected_mime, None

            elif ext == ".ogg":
                if not content.startswith(b"OggS"):
                    return False, None, "Corrupted OGG: missing 'OggS' container signature."
                return True, expected_mime, None

            elif ext == ".flac":
                if not content.startswith(b"fLaC"):
                    return False, None, "Corrupted FLAC: missing 'fLaC' stream marker."
                return True, expected_mime, None

            elif ext == ".m4a":
                if len(content) < 12 or content[4:8] != b"ftyp":
                    return False, None, "Corrupted M4A: missing 'ftyp' audio container atom."
                return True, expected_mime, None

            elif ext == ".aac":
                has_id3 = content.startswith(b"ID3")
                has_adts = (
                    len(content) >= 2
                    and content[0] == 0xFF
                    and (content[1] & 0xF0) == 0xF0
                )
                if not (has_id3 or has_adts):
                    return False, None, "Corrupted AAC: missing ADTS sync word or ID3 tag."
                return True, expected_mime, None

            else:
                return False, None, f"Unsupported file extension: {ext}"

        except Exception as e:
            return False, None, f"File inspection failed with error: {str(e)}"

    def validate(self, filename: str, content: bytes) -> ValidationResult:
        """
        Executes full validation suite on an uploaded file.
        Returns a ValidationResult object with detailed metrics and errors if any.
        """
        safe_name = self.sanitize_filename(filename)
        ext = Path(safe_name).suffix.lower()
        file_size = len(content)
        sha256 = self.compute_sha256(content) if file_size > 0 else ""

        # 1. Extension and Category Check
        category = self.get_category_for_extension(ext)
        if not category:
            allowed_list = []
            for cat_name, ext_set in self.allowed_extensions.items():
                allowed_list.append(f"{cat_name.title()} ({', '.join(sorted(ext_set))})")
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.INVALID,
                file_type=ext,
                file_size_bytes=file_size,
                safe_filename=safe_name,
                sha256_hash=sha256,
                error_message=f"File extension '{ext}' is not supported. Supported: {'; '.join(allowed_list)}",
                details={"reason": "unsupported_extension"},
            )

        # 2. Empty File Check
        if file_size == 0:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.INVALID,
                category=category,
                file_type=ext,
                file_size_bytes=0,
                safe_filename=safe_name,
                sha256_hash="",
                error_message="File is empty (0 bytes). Uploaded files must contain data.",
                details={"reason": "empty_file"},
            )

        # 3. Size Limits Check
        is_size_valid, size_error = self.validate_size(file_size, category)
        if not is_size_valid:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.INVALID,
                category=category,
                file_type=ext,
                file_size_bytes=file_size,
                safe_filename=safe_name,
                sha256_hash=sha256,
                error_message=size_error,
                details={"reason": "size_exceeded"},
            )

        # 4. Content / Magic bytes / Corruption Check
        is_content_valid, mime_type, content_error = self.validate_file_content(content, ext)
        if not is_content_valid:
            return ValidationResult(
                is_valid=False,
                status=ValidationStatus.INVALID,
                category=category,
                file_type=ext,
                mime_type=mime_type,
                file_size_bytes=file_size,
                safe_filename=safe_name,
                sha256_hash=sha256,
                error_message=content_error,
                details={"reason": "corrupted_or_invalid_signature"},
            )

        # File is valid!
        return ValidationResult(
            is_valid=True,
            status=ValidationStatus.VALID,
            category=category,
            file_type=ext,
            mime_type=mime_type or ALLOWED_MIME_TYPES.get(ext, "application/octet-stream"),
            file_size_bytes=file_size,
            safe_filename=safe_name,
            sha256_hash=sha256,
            details={"checked": "signature_and_structure_verified"},
        )
