"""
Ingestion Pipeline Orchestrator for UpToSkills AI Assistant.
Coordinates file validation, SHA-256 duplicate detection,
persistent categorized storage, and metadata recording.
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple, Union
from pathlib import Path

from backend.ingestion.text_extractor import extract_text
from backend.config import ENABLE_DUPLICATE_DETECTION
from backend.ingestion.metadata import MetadataManager
from backend.ingestion.models import (
    DocumentMetadata,
    IngestionResult,
    ValidationResult,
    ValidationStatus,
)
from backend.ingestion.preprocessing import (
    preprocess_text,
    remove_repeated_headers_footers,
    recursive_chunking,
    attach_chunk_metadata,
    verify_chunk_tokens,
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
        existing = None
        dup_validation = None

        if self.enable_duplicate_detection and validation_res.sha256_hash:
            existing = self.metadata.get_by_hash(validation_res.sha256_hash)

        if existing:
            doc_id = existing.document_id
            meta = existing
            abs_path = self.storage.base_dir / existing.storage_path

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
                error_message="Duplicate detected. Processing existing file.",
                details={
                    "duplicate_of": existing.document_id,
                    "original_name": existing.original_filename,
                    "stored_at": existing.upload_timestamp,
                },
            )

        else:
            # Step 3: Store new file
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


        # Step 5: Extract text from supported documents
        extracted_content = None
        extraction_error = None

        supported_extensions = [
            ".pdf", ".docx", ".txt",
            ".jpg", ".jpeg", ".png", ".webp",
            ".mp4", ".mov", ".avi", ".mkv", ".webm"
        ]

        if Path(abs_path).suffix.lower() in supported_extensions:
            try:
                extracted_content = extract_text(abs_path)
                print("Text extraction successful!")
            except Exception as e:
                extraction_error = str(e)
                print(f"Text extraction failed: {e}")

        # Step 6: Preprocessing and Chunking
        preprocessed_content = []

        if extracted_content:
            file_type = extracted_content.get("file_type", "")

            if file_type == "pdf":
                pages = remove_repeated_headers_footers(
                    extracted_content.get("pages", [])
                )

                all_chunks = []

                for page in pages:
                    text = preprocess_text(page["text"])
                    chunks = recursive_chunking(text)

                    for chunk in chunks:
                        all_chunks.append({
                            "text": chunk,
                            "page_number": page.get("page_number")
                        })

                chunks_only = [item["text"] for item in all_chunks]

                preprocessed_content = attach_chunk_metadata(
                    chunks_only, doc_id, filename, file_type
                )

                for index, chunk in enumerate(preprocessed_content):
                    chunk["chunk_index"] = index
                    chunk["page_number"] = all_chunks[index]["page_number"]

            elif file_type == "image":
                text = preprocess_text(
                    extracted_content.get("text", "")
                )

                chunks = recursive_chunking(text)

                preprocessed_content = attach_chunk_metadata(
                    chunks, doc_id, filename, file_type
                )

            elif file_type == "video":
                text = preprocess_text(
                    extracted_content.get("text", "")
                )

                chunks = recursive_chunking(text)

                preprocessed_content = attach_chunk_metadata(
                    chunks, doc_id, filename, file_type
                )

            elif file_type == "txt":
                text = preprocess_text(
                    extracted_content.get("text", "")
                )

                chunks = recursive_chunking(text)

                preprocessed_content = attach_chunk_metadata(
                    chunks, doc_id, filename, file_type
                )

            elif file_type == "docx":
                text_parts = extracted_content.get("paragraphs", [])
                table_parts = []

                # Each item in tables is already a row
                for row_index, row in enumerate(
                    extracted_content.get("tables", []), start=1
                ):
                    # Join cells normally instead of iterating over characters
                    row_text = " | ".join(
                        str(cell).strip()
                        for cell in row
                        if str(cell).strip()
                    )

                    if row_text:
                        table_parts.append(f"Table Row {row_index}: {row_text}")

                combined_text = "\n".join(text_parts)

                if table_parts:
                    combined_text += "\n\n" + "\n".join(table_parts)

                text = preprocess_text(combined_text)
                chunks = recursive_chunking(text)

                preprocessed_content = attach_chunk_metadata(
                    chunks, doc_id, filename, file_type
                )





        # Verify chunk token limits
        preprocessed_content = verify_chunk_tokens(
            preprocessed_content
        )

        return IngestionResult(
            success=not bool(existing),
            document_id=doc_id,
            validation_result=dup_validation or validation_res,
            metadata=meta,
            extracted_content=extracted_content,
            extraction_error=extraction_error,
            preprocessed_content=preprocessed_content,
            message=(
                f"Duplicate detected. Text processed using existing document {doc_id}."
                if existing
                else "Document successfully validated, stored, and processed."
            ),
        )






    def process_batch(self, files: List[Tuple[str, bytes]]) -> List[IngestionResult]:
        """Processes multiple uploaded files in sequence."""
        results = []
        for filename, content in files:
            res = self.process_file(filename, content)
            results.append(res)
        return results
