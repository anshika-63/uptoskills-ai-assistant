"""
Ingestion Pipeline Orchestrator for UpToSkills AI Assistant.
Coordinates file validation, SHA-256 duplicate detection,
persistent categorized storage, metadata recording, and text/audio extraction.

Step order (new files):
  1. Validate format, size, header / corruption, sanitize filename.
  2. Detect duplicates via SHA-256. On duplicate, retrieve and return
     any existing persisted extraction so the caller never sees "no data".
  3. Store validated file bytes to categorized directory.
  4. Save document metadata row to SQLite (status='stored').
  5. Run text extraction for PDF/DOCX/TXT or audio transcription;
     persist result via MetadataManager.save_extraction(); update metadata status.
  6. Preprocess and chunk extracted content with token estimation.
  7. Return IngestionResult with separate upload_success and extraction_status.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

logger = logging.getLogger("rag.pipeline")


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_file(self, filename: str, content: bytes) -> IngestionResult:
        """
        Executes ingestion for a single file:
        1. Validates format, size, header/corruption, and sanitizes filename.
        2. Detects duplicates via SHA-256 hash. On duplicate, retrieves existing
           extraction so the caller never sees "no data".
        3. Persists file to category storage folder.
        4. Saves persistent metadata to SQLite.
        5. Extracts text (PDF, DOCX, TXT) or transcribes audio; persists result.
        6. Preprocesses and chunks content with token estimation.
        7. Returns IngestionResult with separate upload_success and extraction_status.

        Returns an IngestionResult with:
          - success         : True when file was stored (new) or is a known dup.
          - upload_success  : True only for genuinely new files.
          - extraction_status: 'success' | 'failed' | 'not_applicable' | 'retrieved'
          - extracted_content: the extraction dict (new or retrieved from DB).
          - extraction_error : populated when extraction raised an exception.
        """

        # ── Step 1: Validation ────────────────────────────────────────────
        validation_res = self.validator.validate(filename, content)
        if not validation_res.is_valid:
            return IngestionResult(
                success=False,
                upload_success=False,
                extraction_status="not_applicable",
                validation_result=validation_res,
                message=validation_res.error_message or "Validation failed.",
            )

        # ── Step 2: Duplicate Detection ───────────────────────────────────
        if self.enable_duplicate_detection and validation_res.sha256_hash:
            existing = self.metadata.get_by_hash(validation_res.sha256_hash)
            if existing:
                return self._handle_duplicate(existing, validation_res)

        # ── Step 3: Store File ────────────────────────────────────────────
        doc_id    = self.storage.generate_document_id()
        safe_name = validation_res.safe_filename or "document.bin"
        category  = validation_res.category or "documents"

        abs_path, rel_path = self.storage.save_file(
            document_id=doc_id,
            safe_filename=safe_name,
            category=category,
            content=content,
        )
        logger.info("File stored: doc_id=%s path=%s", doc_id, rel_path)

        # ── Step 4: Save Document Metadata ────────────────────────────────
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

        # ── Step 5: Text Extraction / Audio Transcription + Persistence ───
        extracted_content, extraction_error, extraction_status = \
            self._run_extraction_and_persist(doc_id, abs_path)

        # Update document status and metadata_json
        if extraction_status == "success":
            meta.status = "transcribed" if category == "audio" else "extracted"
        else:
            meta.status = "stored"

        if category == "audio":
            extra_meta = {
                "transcription_status": extraction_status,
                "transcript_path": extracted_content.get("transcript_path", f"transcripts/{doc_id}.json") if extracted_content else "",
                "duration": extracted_content.get("duration", 0.0) if extracted_content else 0.0,
                "language": extracted_content.get("language", "unknown") if extracted_content else "unknown",
                "segment_count": extracted_content.get("segment_count", 0) if extracted_content else 0,
            }
            if extraction_error:
                extra_meta["error"] = extraction_error
            meta.metadata_json = json.dumps(extra_meta)

        self.metadata.save(meta)

        # ── Step 6: Preprocessing and Chunking ────────────────────────────
        preprocessed_content: List[Any] = []
        suffix = abs_path.suffix.lower()

        if extracted_content:
            file_type = extracted_content.get("file_type", "")

            if file_type == "pdf":
                pages = remove_repeated_headers_footers(
                    extracted_content.get("pages", [])
                )
                all_chunks: List[Dict[str, Any]] = []
                for page in pages:
                    text = preprocess_text(page.get("text", ""))
                    if text:
                        chunks = recursive_chunking(text)
                        for chunk in chunks:
                            all_chunks.append({
                                "text": chunk,
                                "page_number": page.get("page_number"),
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
                table_parts: List[str] = []
                for t_idx, table in enumerate(extracted_content.get("tables", []), start=1):
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

        # ── Step 7: Verify chunk token limits ─────────────────────────────
        if preprocessed_content:
            preprocessed_content = verify_chunk_tokens(preprocessed_content)

        return IngestionResult(
            success=True,
            upload_success=True,
            document_id=doc_id,
            validation_result=validation_res,
            metadata=meta,
            extracted_content=extracted_content,
            extraction_error=extraction_error,
            extraction_status=extraction_status,
            preprocessed_content=preprocessed_content,
            message="Document successfully validated, stored, and processed.",
        )

    def process_batch(self, files: List[Tuple[str, bytes]]) -> List[IngestionResult]:
        """Processes multiple uploaded files in sequence."""
        return [self.process_file(filename, content) for filename, content in files]

    def get_or_reextract(self, document_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Retrieve a document's persisted extraction/transcript, or re-run
        extraction/transcription from the stored file when the record is missing.

        Used by the Streamlit Document Inspector and query modules.

        Returns:
            (extraction_dict, error_string) — error_string is None on success.
        """
        # 1. Try DB first
        cached = self.metadata.get_extraction(document_id)
        if cached is not None and cached.get("extraction_status") != "failed":
            if "transcript_path" not in cached and cached.get("category") == "audio":
                disk_t = self.storage.get_transcript_json(document_id)
                if disk_t:
                    cached.update(disk_t)
            return cached, None

        # 2. Check disk transcript JSON for audio
        disk_transcript = self.storage.get_transcript_json(document_id)
        if disk_transcript:
            self.metadata.save_extraction(document_id, disk_transcript, extraction_status="success")
            return disk_transcript, None

        # 3. No record — look up the stored file and re-extract / re-transcribe
        meta = self.metadata.get_by_id(document_id)
        if meta is None:
            return None, f"Document '{document_id}' not found in metadata store."

        abs_path = self.storage.base_dir / meta.storage_path
        if not abs_path.exists():
            return None, f"Physical file missing on disk: {abs_path}"

        extracted, error, _ = self._run_extraction_and_persist(document_id, abs_path)
        if extracted and meta.category == "audio":
            meta.status = "transcribed"
            meta.metadata_json = json.dumps({
                "transcription_status": "success",
                "transcript_path": extracted.get("transcript_path", f"transcripts/{document_id}.json"),
                "duration": extracted.get("duration", 0.0),
                "language": extracted.get("language", "unknown"),
                "segment_count": extracted.get("segment_count", 0),
            })
            self.metadata.save(meta)

        return extracted, error

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_duplicate(
        self,
        existing: DocumentMetadata,
        validation_res: ValidationResult,
    ) -> IngestionResult:
        """
        Build and return the IngestionResult for a duplicate upload.
        Retrieves existing extraction (or re-extracts from disk) so the caller
        receives extraction data instead of None.
        """
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
            error_message=(
                f"Duplicate file: matches existing document "
                f"'{existing.original_filename}' (ID: {existing.document_id})."
            ),
            details={
                "duplicate_of": existing.document_id,
                "original_name": existing.original_filename,
                "stored_at": existing.upload_timestamp,
            },
        )

        # Retrieve extraction — may trigger re-extraction if record missing
        extracted_content, extraction_error = self.get_or_reextract(existing.document_id)
        extraction_status = "retrieved" if extracted_content else (
            "failed" if extraction_error else "not_applicable"
        )

        return IngestionResult(
            success=False,
            upload_success=False,
            document_id=existing.document_id,
            validation_result=dup_validation,
            metadata=existing,
            extracted_content=extracted_content,
            extraction_error=extraction_error,
            extraction_status=extraction_status,
            message=(
                f"Duplicate detected: File is identical to document '{existing.document_id}'. "
                "Existing extraction data returned."
            ),
        )

    def _run_extraction_and_persist(
        self,
        document_id: str,
        abs_path: Path,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
        """
        Run text extraction or audio transcription for a supported file type and persist the result.

        Returns:
            (extracted_content, extraction_error, extraction_status)
            extraction_status is one of: 'success' | 'failed' | 'not_applicable'
        """
        ext = Path(abs_path).suffix.lower()

        # Handle Audio Transcription
        if ext in AUDIO_EXTENSIONS:
            try:
                transcript = transcribe_audio(abs_path)
                # Save JSON transcript on disk
                _, rel_transcript_path = self.storage.save_transcript_json(document_id, transcript)
                transcript["transcript_path"] = rel_transcript_path

                # Save into SQLite document_extraction
                self.metadata.save_extraction(document_id, transcript, extraction_status="success")
                logger.info(
                    "Audio transcription succeeded: doc_id=%s duration=%.2fs segments=%d",
                    document_id,
                    transcript.get("duration", 0.0),
                    transcript.get("segment_count", 0),
                )
                return transcript, None, "success"
            except Exception as exc:
                error_msg = str(exc)
                logger.error("Audio transcription failed: doc_id=%s error=%s", document_id, error_msg)
                self.metadata.save_extraction(
                    document_id,
                    {
                        "file_type": ext.lstrip("."),
                        "category": "audio",
                        "raw_text": "",
                        "char_count": 0,
                        "word_count": 0,
                        "segments": [],
                    },
                    extraction_status="failed",
                )
                return None, error_msg, "failed"

        # Handle Document Text Extraction
        elif ext in EXTRACTABLE_EXTENSIONS:
            try:
                extracted = extract_text(abs_path)
                self.metadata.save_extraction(document_id, extracted, extraction_status="success")
                logger.info(
                    "Extraction succeeded: doc_id=%s chars=%d words=%d",
                    document_id,
                    extracted.get("char_count", 0),
                    extracted.get("word_count", 0),
                )
                return extracted, None, "success"
            except Exception as exc:
                error_msg = str(exc)
                # Persist a failed record so we don't silently retry on every dashboard load
                self.metadata.save_extraction(
                    document_id,
                    {"file_type": ext.lstrip("."), "raw_text": "", "char_count": 0, "word_count": 0},
                    extraction_status="failed",
                )
                logger.error("Extraction failed: doc_id=%s error=%s", document_id, error_msg)
                return None, error_msg, "failed"

        return None, None, "not_applicable"
