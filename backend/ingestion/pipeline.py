"""
Ingestion Pipeline Orchestrator for UpToSkills AI Assistant.
Coordinates file validation, SHA-256 duplicate detection,
persistent categorized storage, metadata recording, and text/audio extraction.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

from backend.config import ENABLE_DUPLICATE_DETECTION
from backend.ingestion.audio_transcriber import AUDIO_EXTENSIONS, transcribe_audio
from backend.ingestion.metadata import MetadataManager
from backend.ingestion.models import (
    DocumentMetadata,
    IngestionResult,
    ValidationResult,
    ValidationStatus,
)
from backend.ingestion.preprocessing import (
    attach_chunk_metadata,
    preprocess_text,
    recursive_chunking,
    remove_repeated_headers_footers,
    verify_chunk_tokens,
)
from backend.ingestion.storage import StorageManager
from backend.ingestion.text_extractor import EXTRACTABLE_EXTENSIONS, extract_text
from backend.ingestion.validator import FileValidator


class IngestionPipeline:
    """End-to-end coordinator for document upload, validation, storage, metadata, and extraction."""

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
        Executes ingestion for a single file:
        1. Validates format, size, header/corruption, and sanitizes filename.
        2. Detects duplicates via SHA-256 hash.
        3. Persists file to category storage folder.
        4. Saves persistent metadata to SQLite.
        5. Extracts text (PDF, DOCX, TXT, images, video) or transcribes audio.
        6. Preprocesses and chunks content with token estimation.
        7. Persists extraction records to SQLite.

        Returns an IngestionResult.
        """
        # Step 1: Validation
        validation_res = self.validator.validate(filename, content)
        if not validation_res.is_valid:
            return IngestionResult(
                success=False,
                upload_success=False,
                extraction_status="not_applicable",
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
            category = existing.category

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

        # Step 5: Extract text or transcribe audio
        extracted_content = None
        extraction_error = None
        extraction_status = "not_applicable"

        suffix = Path(abs_path).suffix.lower()

        # Check existing extraction if duplicate
        if existing:
            existing_extraction = self.metadata.get_extraction(doc_id)
            stats = self.metadata.get_extraction_stats(doc_id)
            if existing_extraction:
                extracted_content = existing_extraction
                extraction_status = stats.get("extraction_status", "success") if stats else "success"
            elif stats:
                extraction_status = stats.get("extraction_status", "failed")
                extraction_error = stats.get("error_message")

        # If not already extracted, proceed with extraction
        if extracted_content is None and extraction_error is None:
            if suffix in AUDIO_EXTENSIONS or category == "audio":
                try:
                    res = transcribe_audio(abs_path)
                    if res.get("success"):
                        extracted_content = res
                        extraction_status = "success"
                    else:
                        extraction_error = res.get("error", "Audio transcription failed")
                        extraction_status = "failed"
                except Exception as e:
                    extraction_error = str(e)
                    extraction_status = "failed"
            elif suffix in EXTRACTABLE_EXTENSIONS:
                try:
                    extracted_content = extract_text(abs_path)
                    extraction_status = "success"
                except Exception as e:
                    extraction_error = str(e)
                    extraction_status = "failed"

            # Persist extraction record
            if extraction_status != "not_applicable":
                self.metadata.save_extraction(
                    document_id=doc_id,
                    extraction_status=extraction_status,
                    extracted_content=extracted_content,
                    error_message=extraction_error,
                )

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
                    text = preprocess_text(page.get("text", ""))
                    if text:
                        chunks = recursive_chunking(text)
                        for chunk in chunks:
                            all_chunks.append({
                                "text": chunk,
                                "page_number": page.get("page_number")
                            })

                if all_chunks:
                    chunks_only = [item["text"] for item in all_chunks]
                    preprocessed_content = attach_chunk_metadata(
                        chunks_only, doc_id, filename, file_type
                    )
                    for index, chunk in enumerate(preprocessed_content):
                        chunk["chunk_index"] = index
                        chunk["page_number"] = all_chunks[index]["page_number"]

            elif file_type == "docx":
                text_parts = extracted_content.get("paragraphs", [])
                table_parts = []

                tables = extracted_content.get("tables", [])
                for t_idx, table in enumerate(tables, start=1):
                    if isinstance(table, list):
                        for row_index, row in enumerate(table, start=1):
                            if isinstance(row, list):
                                row_text = " | ".join(str(cell).strip() for cell in row if str(cell).strip())
                            else:
                                row_text = str(row).strip()
                            if row_text:
                                table_parts.append(f"Table {t_idx} Row {row_index}: {row_text}")

                combined_text = "\n".join(text_parts)
                if table_parts:
                    combined_text += "\n\n" + "\n".join(table_parts)

                text = preprocess_text(combined_text)
                if text:
                    chunks = recursive_chunking(text)
                    preprocessed_content = attach_chunk_metadata(
                        chunks, doc_id, filename, file_type
                    )

            elif file_type in ("txt", "image", "video"):
                raw_text = extracted_content.get("raw_text") or extracted_content.get("text", "")
                text = preprocess_text(raw_text)
                if text:
                    chunks = recursive_chunking(text)
                    preprocessed_content = attach_chunk_metadata(
                        chunks, doc_id, filename, file_type
                    )

            elif file_type == "audio" or extracted_content.get("category") == "audio" or suffix in AUDIO_EXTENSIONS:
                raw_text = extracted_content.get("raw_text", "")
                text = preprocess_text(raw_text)
                if text:
                    chunks = recursive_chunking(text)
                    preprocessed_content = attach_chunk_metadata(
                        chunks, doc_id, filename, "audio"
                    )

        # Step 7: Verify chunk token limits
        if preprocessed_content:
            preprocessed_content = verify_chunk_tokens(preprocessed_content)

        return IngestionResult(
            success=not bool(existing),
            upload_success=not bool(existing),
            extraction_status=extraction_status,
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

    def get_or_reextract(self, document_id: str) -> Tuple[Optional[dict], Optional[str]]:
        """
        Retrieves previously extracted content for a document from SQLite.
        If missing, re-extracts the file directly from storage disk and persists the result.
        Returns (extracted_content_dict, error_message).
        """
        # 1. Try SQLite metadata store first
        extracted = self.metadata.get_extraction(document_id)
        if extracted:
            return extracted, None

        # 2. Not in SQLite; locate document in storage
        doc_meta = self.metadata.get_by_id(document_id)
        if not doc_meta:
            return None, f"Document ID '{document_id}' not found in metadata database."

        abs_path = self.storage.base_dir / doc_meta.storage_path
        if not abs_path.exists():
            return None, f"File does not exist on disk at '{abs_path}'."

        # 3. Re-extract
        ext = abs_path.suffix.lower()
        try:
            if ext in AUDIO_EXTENSIONS or doc_meta.category == "audio":
                res = transcribe_audio(abs_path)
                if res.get("success"):
                    self.metadata.save_extraction(document_id, "success", res)
                    return res, None
                else:
                    err = res.get("error", "Audio transcription failed")
                    self.metadata.save_extraction(document_id, "failed", error_message=err)
                    return None, err
            elif ext in EXTRACTABLE_EXTENSIONS:
                res = extract_text(abs_path)
                self.metadata.save_extraction(document_id, "success", res)
                return res, None
            else:
                return None, f"File type '{ext}' is not extractable."
        except Exception as e:
            err = str(e)
            self.metadata.save_extraction(document_id, "failed", error_message=err)
            return None, err

    def process_batch(self, files: List[Tuple[str, bytes]]) -> List[IngestionResult]:
        """Processes multiple uploaded files in sequence."""
        results = []
        for filename, content in files:
            res = self.process_file(filename, content)
            results.append(res)
        return results
