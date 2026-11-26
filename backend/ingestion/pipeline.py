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

from backend.config import ENABLE_DUPLICATE_DETECTION, ENABLE_MULTIMODAL
from backend.ingestion.audio_transcriber import (
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    format_timestamp,
    transcribe_audio,
    transcribe_video,
)
from backend.ingestion.embeddings import EmbeddingService
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
from backend.ingestion.text_extractor import (
    EXTRACTABLE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    extract_image,
    extract_text,
    extract_video,
    is_ocr_available,
    is_scanned_pdf,
)
from backend.ingestion.validator import FileValidator
from backend.ingestion.vector_store import VectorStoreManager

logger = logging.getLogger("rag.pipeline")


class IngestionPipeline:
    """End-to-end coordinator for document upload, validation, storage, metadata, extraction, and vector indexing."""

    def __init__(
        self,
        validator: Optional[FileValidator] = None,
        storage_manager: Optional[StorageManager] = None,
        metadata_manager: Optional[MetadataManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStoreManager] = None,
        enable_duplicate_detection: bool = ENABLE_DUPLICATE_DETECTION,
        enable_multimodal: bool = ENABLE_MULTIMODAL,
    ):
        self.validator = validator or FileValidator()
        self.storage = storage_manager or StorageManager()
        self.metadata = metadata_manager or MetadataManager()
        self.embeddings = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStoreManager(embedding_service=self.embeddings)
        self.enable_duplicate_detection = enable_duplicate_detection
        self.enable_multimodal = enable_multimodal

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
            if category in ("audio", "videos"):
                meta.status = "transcribed"
            elif category == "images":
                meta.status = "ocr_extracted"
            else:
                meta.status = "extracted"
        else:
            meta.status = "stored"

        if category in ("audio", "videos") and extracted_content:
            extra_meta = {
                "transcription_status": extraction_status,
                "transcript_path": extracted_content.get("transcript_path", f"transcripts/{doc_id}.json"),
                "duration": extracted_content.get("duration", 0.0),
                "language": extracted_content.get("language", "unknown"),
                "segment_count": extracted_content.get("segment_count", 0),
            }
            if extraction_error:
                extra_meta["error"] = extraction_error
            meta.metadata_json = json.dumps(extra_meta)
        elif category == "images" and extracted_content:
            extra_meta = {
                "ocr_status": extraction_status,
                "extraction_method": extracted_content.get("extraction_method", "tesseract_ocr"),
                "char_count": extracted_content.get("char_count", 0),
                "word_count": extracted_content.get("word_count", 0),
            }
            if extraction_error:
                extra_meta["error"] = extraction_error
            meta.metadata_json = json.dumps(extra_meta)

        self.metadata.save(meta)

        # ── Step 6: Preprocessing and Chunking ────────────────────────────
        preprocessed_content = []
        suffix = abs_path.suffix.lower()
        if extracted_content:
            preprocessed_content = self._build_chunks(doc_id, filename, extracted_content, suffix)

        # ── Step 7: Verify chunk token limits ─────────────────────────────
        if preprocessed_content:
            preprocessed_content = verify_chunk_tokens(preprocessed_content)

        # ── Step 8: Embedding Generation & ChromaDB Vector Storage ────────
        embedding_status = "not_applicable"
        embedding_error = None
        vector_count = 0

        if preprocessed_content:
            try:
                # 1. Persist chunks in SQLite
                self.metadata.save_chunks(doc_id, preprocessed_content)

                # 2. Generate embeddings and store in ChromaDB
                vector_count = self.vector_store.update_document(doc_id, preprocessed_content)
                embedding_status = "indexed"
                logger.info(
                    "Generated embeddings and indexed %d chunks for doc_id=%s in ChromaDB",
                    vector_count,
                    doc_id,
                )

                # Update metadata status to reflect vector indexing
                meta.status = "indexed"
                self.metadata.save(meta)
            except Exception as exc:
                embedding_error = str(exc)
                embedding_status = "failed"
                logger.error(
                    "Embedding generation/vector storage failed for doc_id=%s: %s",
                    doc_id,
                    exc,
                )

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
            embedding_status=embedding_status,
            embedding_error=embedding_error,
            vector_count=vector_count,
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
            if "transcript_path" not in cached and cached.get("category") in ("audio", "videos"):
                disk_t = self.storage.get_transcript_json(document_id)
                if disk_t:
                    cached.update(disk_t)
            return cached, None

        # 2. Check disk transcript JSON for audio or video
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
        if extracted and meta.category in ("audio", "videos"):
            meta.status = "transcribed"
            meta.metadata_json = json.dumps({
                "transcription_status": "success",
                "transcript_path": extracted.get("transcript_path", f"transcripts/{document_id}.json"),
                "duration": extracted.get("duration", 0.0),
                "language": extracted.get("language", "unknown"),
                "segment_count": extracted.get("segment_count", 0),
            })
            self.metadata.save(meta)
        elif extracted and meta.category == "images":
            meta.status = "ocr_extracted"
            meta.metadata_json = json.dumps({
                "ocr_status": "success",
                "extraction_method": extracted.get("extraction_method", "tesseract_ocr"),
                "char_count": extracted.get("char_count", 0),
                "word_count": extracted.get("word_count", 0),
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

        # Retrieve chunks and vector count from existing document
        chunks = self.metadata.get_chunks(existing.document_id)
        vector_count = self.vector_store.count_by_document(existing.document_id)

        # If chunks/vectors not yet generated for this existing doc, generate them now
        if not chunks and extracted_content:
            suffix = Path(existing.storage_path).suffix.lower()
            chunks = self._build_chunks(existing.document_id, existing.original_filename, extracted_content, suffix)
            if chunks:
                chunks = verify_chunk_tokens(chunks)
                self.metadata.save_chunks(existing.document_id, chunks)
                vector_count = self.vector_store.update_document(existing.document_id, chunks)

        embedding_status = "retrieved" if vector_count > 0 else "not_applicable"

        return IngestionResult(
            success=False,
            upload_success=False,
            document_id=existing.document_id,
            validation_result=dup_validation,
            metadata=existing,
            extracted_content=extracted_content,
            extraction_error=extraction_error,
            extraction_status=extraction_status,
            preprocessed_content=chunks if chunks else None,
            embedding_status=embedding_status,
            vector_count=vector_count,
            message=(
                f"Duplicate detected: File is identical to document '{existing.document_id}'. "
                "Existing extraction and vector data returned."
            ),
        )

    def delete_document(self, document_id: str) -> bool:
        """
        Completely deletes a document:
        1. Deletes associated vectors from ChromaDB.
        2. Deletes chunks, extraction records, and metadata from SQLite.
        3. Deletes physical file and any transcript JSON from disk.

        Returns True if records or files were removed.
        """
        doc_meta = self.metadata.get_by_id(document_id)
        deleted_file = False
        if doc_meta:
            deleted_file = self.storage.delete_file(doc_meta.storage_path)
            self.storage.delete_transcript_json(document_id)

        # 1. Delete from ChromaDB
        self.vector_store.delete_document(document_id)

        # 2. Delete from SQLite (metadata, extraction, and chunks)
        deleted_meta = self.metadata.delete(document_id)
        return deleted_meta or deleted_file

    def reprocess_document(self, document_id: str) -> Optional[IngestionResult]:
        """
        Reprocesses an existing document without creating duplicate records:
        re-extracts, re-chunks, regenerates embeddings, and updates ChromaDB and SQLite.
        """
        doc_meta = self.metadata.get_by_id(document_id)
        if not doc_meta:
            return None

        abs_path = self.storage.base_dir / doc_meta.storage_path
        if not abs_path.exists():
            return None

        extracted_content, extraction_error = self.get_or_reextract(document_id)
        suffix = abs_path.suffix.lower()
        chunks = []
        if extracted_content:
            chunks = self._build_chunks(document_id, doc_meta.original_filename, extracted_content, suffix)
            if chunks:
                chunks = verify_chunk_tokens(chunks)

        vector_count = 0
        embedding_status = "not_applicable"
        embedding_error = None
        if chunks:
            try:
                self.metadata.save_chunks(document_id, chunks)
                vector_count = self.vector_store.update_document(document_id, chunks)
                embedding_status = "indexed"
                doc_meta.status = "indexed"
                self.metadata.save(doc_meta)
            except Exception as exc:
                embedding_error = str(exc)
                embedding_status = "failed"

        val_res = ValidationResult(
            is_valid=True,
            category=doc_meta.category,
            file_type=doc_meta.file_type,
            mime_type=doc_meta.mime_type,
            file_size_bytes=doc_meta.file_size_bytes,
            sha256_hash=doc_meta.sha256_hash,
            safe_filename=doc_meta.safe_filename,
        )

        return IngestionResult(
            success=True,
            upload_success=False,
            document_id=document_id,
            validation_result=val_res,
            metadata=doc_meta,
            extracted_content=extracted_content,
            extraction_error=extraction_error,
            extraction_status="success" if extracted_content else "failed",
            preprocessed_content=chunks if chunks else None,
            embedding_status=embedding_status,
            embedding_error=embedding_error,
            vector_count=vector_count,
            message=f"Document '{document_id}' successfully reprocessed and indexed.",
        )

    def _build_chunks(
        self,
        doc_id: str,
        filename: str,
        extracted_content: Dict[str, Any],
        suffix: str,
    ) -> List[Dict[str, Any]]:
        """Splits extracted document content into structured chunks with metadata."""
        file_type = extracted_content.get("file_type", "")
        chunks_result: List[Dict[str, Any]] = []

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
                            "is_scanned": page.get("is_scanned", False),
                            "extraction_method": page.get("extraction_method", "pymupdf"),
                        })

            if all_chunks:
                chunks_only = [item["text"] for item in all_chunks]
                chunks_result = attach_chunk_metadata(
                    chunks_only, doc_id, filename, file_type
                )
                for index, chunk in enumerate(chunks_result):
                    chunk["chunk_index"] = index
                    chunk["page_number"] = all_chunks[index]["page_number"]
                    chunk["is_scanned"] = all_chunks[index]["is_scanned"]
                    chunk["extraction_method"] = all_chunks[index]["extraction_method"]

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
                chunks_result = attach_chunk_metadata(
                    chunks, doc_id, filename, file_type
                )

        elif file_type == "txt":
            raw_text = extracted_content.get("raw_text") or extracted_content.get("text", "")
            text = preprocess_text(raw_text)
            if text:
                chunks = recursive_chunking(text)
                chunks_result = attach_chunk_metadata(
                    chunks, doc_id, filename, file_type
                )

        elif (
            file_type in ("image", "images")
            or extracted_content.get("category") == "images"
            or suffix in IMAGE_EXTENSIONS
        ):
            raw_text = extracted_content.get("raw_text") or extracted_content.get("text", "")
            text = preprocess_text(raw_text)
            if text:
                chunks = recursive_chunking(text)
                chunks_result = attach_chunk_metadata(
                    chunks, doc_id, filename, "image"
                )
                for index, chunk in enumerate(chunks_result):
                    chunk["chunk_index"] = index
                    chunk["category"] = "images"
                    chunk["media_type"] = "image"
                    chunk["page_number"] = 1
                    chunk["extraction_method"] = extracted_content.get(
                        "extraction_method", "tesseract_ocr"
                    )

        elif (
            file_type in ("audio", "video")
            or extracted_content.get("category") in ("audio", "videos")
            or suffix in AUDIO_EXTENSIONS
            or suffix in VIDEO_EXTENSIONS
        ):
            segments = extracted_content.get("segments", [])
            cat = extracted_content.get(
                "category", "videos" if suffix in VIDEO_EXTENSIONS else "audio"
            )

            if segments:
                all_chunks = []
                current_texts = []
                chunk_start = segments[0].get("start", 0.0)
                chunk_end = segments[0].get("end", 0.0)
                current_char_count = 0

                for seg in segments:
                    seg_text = seg.get("text", "").strip()
                    if not seg_text:
                        continue
                    if current_char_count + len(seg_text) > 400 and current_texts:
                        combined_text = " ".join(current_texts)
                        preprocessed = preprocess_text(combined_text)
                        if preprocessed:
                            all_chunks.append({
                                "text": preprocessed,
                                "start_time": chunk_start,
                                "end_time": chunk_end,
                                "timestamp": f"[{format_timestamp(chunk_start)} -> {format_timestamp(chunk_end)}]",
                            })
                        current_texts = [seg_text]
                        chunk_start = seg.get("start", 0.0)
                        chunk_end = seg.get("end", 0.0)
                        current_char_count = len(seg_text)
                    else:
                        current_texts.append(seg_text)
                        chunk_end = seg.get("end", 0.0)
                        current_char_count += len(seg_text) + 1

                if current_texts:
                    combined_text = " ".join(current_texts)
                    preprocessed = preprocess_text(combined_text)
                    if preprocessed:
                        all_chunks.append({
                            "text": preprocessed,
                            "start_time": chunk_start,
                            "end_time": chunk_end,
                            "timestamp": f"[{format_timestamp(chunk_start)} -> {format_timestamp(chunk_end)}]",
                        })

                if all_chunks:
                    chunk_texts = [c["text"] for c in all_chunks]
                    chunks_result = attach_chunk_metadata(
                        chunk_texts, doc_id, filename, file_type or cat
                    )
                    for idx, c in enumerate(chunks_result):
                        c["chunk_index"] = idx
                        c["category"] = cat
                        c["start_time"] = all_chunks[idx]["start_time"]
                        c["end_time"] = all_chunks[idx]["end_time"]
                        c["timestamp"] = all_chunks[idx]["timestamp"]
                        c["media_type"] = "video" if cat == "videos" else "audio"
            else:
                raw_text = extracted_content.get("raw_text") or extracted_content.get("text", "")
                text = preprocess_text(raw_text)
                if text:
                    chunks = recursive_chunking(text)
                    chunks_result = attach_chunk_metadata(
                        chunks, doc_id, filename, file_type or cat
                    )
                    for idx, c in enumerate(chunks_result):
                        c["chunk_index"] = idx
                        c["category"] = cat
                        c["media_type"] = "video" if cat == "videos" else "audio"

        return chunks_result

    def _run_extraction_and_persist(
        self,
        document_id: str,
        abs_path: Path,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
        """
        Run text extraction, audio transcription, or video processing for a supported file type and persist the result.

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
                    error_message=error_msg,
                )
                return None, error_msg, "failed"

        # Handle Video Audio Extraction + Transcription
        elif ext in VIDEO_EXTENSIONS:
            if not self.enable_multimodal:
                return None, None, "not_applicable"

            try:
                transcript = transcribe_video(abs_path)
                if transcript.get("success"):
                    _, rel_transcript_path = self.storage.save_transcript_json(document_id, transcript)
                    transcript["transcript_path"] = rel_transcript_path
                    self.metadata.save_extraction(document_id, transcript, extraction_status="success")
                    logger.info(
                        "Video transcription succeeded: doc_id=%s duration=%.2fs segments=%d",
                        document_id,
                        transcript.get("duration", 0.0),
                        transcript.get("segment_count", 0),
                    )
                    return transcript, None, "success"
                else:
                    # Video had no audio stream or failed audio extraction — try frame OCR fallback
                    try:
                        frame_ocr = extract_video(abs_path)
                        if frame_ocr.get("char_count", 0) > 0:
                            self.metadata.save_extraction(document_id, frame_ocr, extraction_status="success")
                            return frame_ocr, None, "success"
                    except Exception:
                        pass

                    err_msg = transcript.get("error") or "Video transcription produced no text."
                    self.metadata.save_extraction(
                        document_id,
                        {
                            "file_type": ext.lstrip("."),
                            "category": "videos",
                            "raw_text": "",
                            "char_count": 0,
                            "word_count": 0,
                            "segments": [],
                        },
                        extraction_status="failed",
                        error_message=err_msg,
                    )
                    return None, err_msg, "failed"
            except Exception as exc:
                error_msg = str(exc)
                logger.error("Video processing failed: doc_id=%s error=%s", document_id, error_msg)
                self.metadata.save_extraction(
                    document_id,
                    {
                        "file_type": ext.lstrip("."),
                        "category": "videos",
                        "raw_text": "",
                        "char_count": 0,
                        "word_count": 0,
                        "segments": [],
                    },
                    extraction_status="failed",
                    error_message=error_msg,
                )
                return None, error_msg, "failed"

        # Handle Image OCR Extraction
        elif ext in IMAGE_EXTENSIONS:
            if not self.enable_multimodal:
                return None, None, "not_applicable"

            try:
                extracted = extract_image(abs_path)
                if extracted.get("extraction_method") == "failed":
                    err_msg = extracted.get("error") or "Image OCR extraction failed."
                    self.metadata.save_extraction(
                        document_id,
                        extracted,
                        extraction_status="failed",
                        error_message=err_msg,
                    )
                    return None, err_msg, "failed"
                elif extracted.get("extraction_method") == "ocr_unavailable":
                    err_msg = extracted.get("error") or "Tesseract OCR engine is not available."
                    self.metadata.save_extraction(
                        document_id,
                        extracted,
                        extraction_status="failed",
                        error_message=err_msg,
                    )
                    return None, err_msg, "failed"
                else:
                    self.metadata.save_extraction(document_id, extracted, extraction_status="success")
                    logger.info(
                        "Image OCR succeeded: doc_id=%s chars=%d words=%d",
                        document_id,
                        extracted.get("char_count", 0),
                        extracted.get("word_count", 0),
                    )
                    return extracted, None, "success"
            except Exception as exc:
                error_msg = str(exc)
                logger.error("Image OCR failed: doc_id=%s error=%s", document_id, error_msg)
                self.metadata.save_extraction(
                    document_id,
                    {
                        "file_type": ext.lstrip("."),
                        "category": "images",
                        "raw_text": "",
                        "char_count": 0,
                        "word_count": 0,
                    },
                    extraction_status="failed",
                    error_message=error_msg,
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
                    error_message=error_msg,
                )
                logger.error("Extraction failed: doc_id=%s error=%s", document_id, error_msg)
                return None, error_msg, "failed"

        return None, None, "not_applicable"
