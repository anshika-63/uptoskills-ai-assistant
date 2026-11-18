"""
Ingestion Pipeline Orchestrator for UpToSkills AI Assistant.
Coordinates file validation, SHA-256 duplicate detection,
persistent categorized storage, and metadata recording.
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple, Union
from pathlib import Path

from backend.config import ENABLE_DUPLICATE_DETECTION
from backend.ingestion.metadata import MetadataManager
from backend.ingestion.models import (
    DocumentMetadata,
    IngestionResult,
    ValidationResult,
    ValidationStatus,
)
from backend.ingestion.storage import StorageManager
from backend.ingestion.validator import FileValidator


class IngestionPipeline:
    """End-to-end coordinator for document upload, validation, storage, and metadata."""

    def __init__(
        self,
        validator: Optional[FileValidator] = None,
        storage_manager: Optional[StorageManager] = None,
        metadata_manager: Optional[MetadataManager] = None,
        enable_duplicate_detection: bool = ENABLE_DUPLICATE_DETECTION,
    ):
        self.validator = validator or FileValidator()
        self.storage = storage_manager or StorageManager()
        self.metadata = metadata_manager or MetadataManager()
        self.enable_duplicate_detection = enable_duplicate_detection

    def process_file(self, filename: str, content: bytes) -> IngestionResult:
        """
        Executes Phase 1 ingestion for a single file:
        1. Validates format, size, header/corruption, and sanitizes filename.
        2. Detects duplicates via SHA-256 hash.
        3. Persists file to category storage folder.
        4. Saves persistent metadata to SQLite.

        Returns an IngestionResult.
        """
        # Step 1: Validation
        validation_res = self.validator.validate(filename, content)
        if not validation_res.is_valid:
            return IngestionResult(
                success=False,
                validation_result=validation_res,
                message=validation_res.error_message or "Validation failed.",
            )

        # Step 2: Duplicate Detection
        if self.enable_duplicate_detection and validation_res.sha256_hash:
            existing = self.metadata.get_by_hash(validation_res.sha256_hash)
            if existing:
                dup_validation = ValidationResult(
                    is_valid=False,
                    is_duplicate=True,
                    status=ValidationStatus.DUPLICATE,
                    category=validation_res.category,
                    file_type=validation_res.file_type,
                    mime_type=validation_res.mime_type,
                    file_size_bytes=validation_res.file_size_bytes,
                    sha256_hash=validation_res.sha256_hash,
                    safe_filename=validation_res.safe_filename,
                    existing_document_id=existing.document_id,
                    error_message=f"Duplicate file: matches existing document '{existing.original_filename}' (ID: {existing.document_id}).",
                    details={
                        "duplicate_of": existing.document_id,
                        "original_name": existing.original_filename,
                        "stored_at": existing.upload_timestamp,
                    },
                )
                return IngestionResult(
                    success=False,
                    document_id=existing.document_id,
                    validation_result=dup_validation,
                    metadata=existing,
                    message=f"Duplicate detected: File is identical to document '{existing.document_id}'.",
                )

        # Step 3: Unique ID Generation & Safe Storage
        doc_id = self.storage.generate_document_id()
        safe_name = validation_res.safe_filename or "document.bin"
        category = validation_res.category or "documents"

        abs_path, rel_path = self.storage.save_file(
            document_id=doc_id,
            safe_filename=safe_name,
            category=category,
            content=content,
        )

        # Step 4: Record Metadata
        timestamp_iso = datetime.now(timezone.utc).isoformat()
        meta = DocumentMetadata(
            document_id=doc_id,
            original_filename=filename,
            safe_filename=safe_name,
            category=category,
            file_type=validation_res.file_type or "",
            mime_type=validation_res.mime_type or "application/octet-stream",
            file_size_bytes=validation_res.file_size_bytes,
            sha256_hash=validation_res.sha256_hash or "",
            storage_path=rel_path,
            upload_timestamp=timestamp_iso,
            status="stored",
        )
        self.metadata.save(meta)

        return IngestionResult(
            success=True,
            document_id=doc_id,
            validation_result=validation_res,
            metadata=meta,
            message="Document successfully validated, stored, and cataloged.",
        )

    def process_batch(self, files: List[Tuple[str, bytes]]) -> List[IngestionResult]:
        """Processes multiple uploaded files in sequence."""
        results = []
        for filename, content in files:
            res = self.process_file(filename, content)
            results.append(res)
        return results
