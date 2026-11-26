"""
Metadata management engine for the RAG document ingestion pipeline.
Provides persistent storage, retrieval, indexing, duplicate detection,
and aggregation using SQLite.

Two tables are managed here:
  - document_metadata : file-level record (upload status, hashes, paths).
  - document_extraction: extracted text payload persisted as JSON alongside
    summary statistics. Query modules access extraction via get_extraction().
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.config import METADATA_DB_PATH
from backend.ingestion.models import DocumentMetadata

logger = logging.getLogger("rag.metadata")


class MetadataManager:
    """Manages persistent SQLite storage of document metadata records."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path or METADATA_DB_PATH).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency and integrity
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        """Initializes tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_metadata (
                    document_id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    safe_filename TEXT NOT NULL,
                    category TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    sha256_hash TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    upload_timestamp TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'stored',
                    metadata_json TEXT
                );
                """
            )
            # Create indexes for efficient querying and instant deduplication
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_hash ON document_metadata(sha256_hash);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_cat ON document_metadata(category);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_timestamp ON document_metadata(upload_timestamp DESC);"
            )

            # Extraction persistence table (created alongside document_metadata
            # so the same DB file covers both upload and extraction state).
            # Merged schema: HEAD columns + origin/main error_message column.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_extraction (
                    document_id       TEXT PRIMARY KEY,
                    file_type         TEXT NOT NULL DEFAULT '',
                    raw_text          TEXT NOT NULL DEFAULT '',
                    char_count        INTEGER NOT NULL DEFAULT 0,
                    word_count        INTEGER NOT NULL DEFAULT 0,
                    page_count        INTEGER NOT NULL DEFAULT 0,
                    extraction_status TEXT NOT NULL DEFAULT 'success',
                    extracted_at      TEXT NOT NULL,
                    payload_json      TEXT,
                    error_message     TEXT
                );
                """
            )
            # Two indexes: one for fast ID look-up, one for status-based queries
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ext_id ON document_extraction(document_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ext_status ON document_extraction(extraction_status);"
            )

            # Processed chunks persistence table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id          TEXT PRIMARY KEY,
                    document_id       TEXT NOT NULL,
                    chunk_index       INTEGER NOT NULL,
                    text              TEXT NOT NULL,
                    page_number       INTEGER DEFAULT 0,
                    estimated_tokens  INTEGER DEFAULT 0,
                    created_at        TEXT NOT NULL,
                    metadata_json     TEXT
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_doc_id ON document_chunks(document_id);"
            )
            conn.commit()

        # Schema migration: add error_message column if upgrading from older DB
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "ALTER TABLE document_extraction ADD COLUMN error_message TEXT;"
                )
                conn.commit()
                logger.info("DB migration: added error_message column to document_extraction.")
        except Exception:
            # Column already exists (OperationalError) — safe to ignore
            pass

    # ------------------------------------------------------------------
    # Extraction persistence helpers
    # ------------------------------------------------------------------

    def save_extraction(
        self,
        document_id: str,
        extraction: Union[Dict[str, Any], str, None] = None,
        extraction_status: str = "success",
        extracted_content: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        extracted_at: Optional[str] = None,
    ) -> bool:
        """
        Persist the full extraction payload for a document.

        Accepts two calling conventions to stay compatible with both pipeline
        versions that emerged from the merge:

          HEAD style   : save_extraction(doc_id, extraction_dict, extraction_status="success")
          origin/main  : save_extraction(doc_id, "success", extracted_content=res, error_message=err)

        The payload is stored as JSON in `payload_json`.  Key statistics
        (raw_text, char_count, word_count, page_count) are promoted to
        dedicated columns so they can be queried without deserialising JSON.

        Returns True on success.
        """
        # Detect calling convention:
        #   origin/main style : save_extraction(doc_id, "success", extracted_content=res)
        #                       second arg is the status string
        #   HEAD style        : save_extraction(doc_id, extraction_dict, extraction_status="success")
        #                       second arg is the dict
        #   keyword-only      : save_extraction(document_id=doc_id, extraction_status="success",
        #                                       extracted_content=res)
        #                       extraction is None; status comes from extraction_status kwarg
        if extraction is None:
            # Keyword-only call: use extraction_status kwarg + extracted_content kwarg
            effective_status = extraction_status
            effective_dict = extracted_content or {}
        elif isinstance(extraction, str):
            # origin/main style: second arg is the status string
            effective_status = extraction
            effective_dict = extracted_content or {}
        else:
            # HEAD style: second arg is the extraction dict
            effective_status = extraction_status
            effective_dict = extraction

        effective_dict = effective_dict or {}
        file_type   = effective_dict.get("file_type", "")
        raw_text    = effective_dict.get("raw_text", "")
        char_count  = effective_dict.get("char_count", len(raw_text) if raw_text else 0)
        word_count  = effective_dict.get("word_count", len(raw_text.split()) if raw_text else 0)
        page_count  = effective_dict.get("total_pages", 1 if file_type not in ("pdf", "") else 0)
        ts = extracted_at or datetime.now(timezone.utc).isoformat()

        try:
            payload = json.dumps(effective_dict, ensure_ascii=False) if effective_dict else None
        except Exception:
            payload = None

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO document_extraction (
                    document_id, file_type, raw_text, char_count, word_count,
                    page_count, extraction_status, extracted_at, payload_json, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    document_id,
                    file_type,
                    raw_text,
                    char_count,
                    word_count,
                    page_count,
                    effective_status,
                    ts,
                    payload,
                    error_message,
                ),
            )
            conn.commit()
        logger.info(
            "Extraction saved: doc_id=%s file_type=%s status=%s chars=%d words=%d",
            document_id, file_type, effective_status, char_count, word_count,
        )
        return True

    def get_extraction(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the persisted extraction payload for a document.

        Returns the full deserialized dict (as returned by extract_text())
        or None if no extraction record exists yet.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT payload_json, extraction_status FROM document_extraction "
                "WHERE document_id = ? LIMIT 1;",
                (document_id,),
            )
            row = cursor.fetchone()
            if row and row["payload_json"]:
                try:
                    return json.loads(row["payload_json"])
                except Exception:
                    return None
        return None

    def extraction_exists(self, document_id: str) -> bool:
        """Returns True when a successful extraction record is stored."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM document_extraction "
                "WHERE document_id = ? AND extraction_status = 'success' LIMIT 1;",
                (document_id,),
            )
            return cursor.fetchone() is not None

    def get_extraction_stats(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns lightweight extraction statistics (no raw_text) for a document.
        Useful for dashboard display without loading full text into memory.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT file_type, char_count, word_count, page_count, "
                "extraction_status, extracted_at, error_message "
                "FROM document_extraction WHERE document_id = ? LIMIT 1;",
                (document_id,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def delete_extraction(self, document_id: str) -> bool:
        """Deletes the extraction record for a document."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM document_extraction WHERE document_id = ?;",
                (document_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def save(self, metadata: DocumentMetadata) -> bool:
        """
        Inserts or updates a document metadata record.
        Returns True on success.
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO document_metadata (
                    document_id,
                    original_filename,
                    safe_filename,
                    category,
                    file_type,
                    mime_type,
                    file_size_bytes,
                    sha256_hash,
                    storage_path,
                    upload_timestamp,
                    status,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    metadata.document_id,
                    metadata.original_filename,
                    metadata.safe_filename,
                    metadata.category,
                    metadata.file_type,
                    metadata.mime_type,
                    metadata.file_size_bytes,
                    metadata.sha256_hash,
                    metadata.storage_path,
                    metadata.upload_timestamp,
                    metadata.status,
                    metadata.metadata_json or json.dumps({}),
                ),
            )
            conn.commit()
            return True

    def get_by_id(self, document_id: str) -> Optional[DocumentMetadata]:
        """Fetches metadata by document ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM document_metadata WHERE document_id = ? LIMIT 1;",
                (document_id,),
            )
            row = cursor.fetchone()
            if row:
                return DocumentMetadata(**dict(row))
        return None

    def get_by_hash(self, sha256_hash: str) -> Optional[DocumentMetadata]:
        """
        Fetches an existing document record matching the provided SHA-256 hash.
        Used for instant duplicate detection.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM document_metadata WHERE sha256_hash = ? LIMIT 1;",
                (sha256_hash,),
            )
            row = cursor.fetchone()
            if row:
                return DocumentMetadata(**dict(row))
        return None

    def list_documents(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DocumentMetadata]:
        """
        Returns a paginated list of documents with optional filtering by
        category and text search on original filename or document_id.
        """
        query = "SELECT * FROM document_metadata WHERE 1=1"
        params: list = []

        if category and category.lower() != "all":
            query += " AND category = ?"
            params.append(category.lower())

        if search and search.strip():
            query += " AND (original_filename LIKE ? OR document_id LIKE ?)"
            term = f"%{search.strip()}%"
            params.extend([term, term])

        query += " ORDER BY upload_timestamp DESC LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [DocumentMetadata(**dict(row)) for row in rows]

    def count_total(self) -> int:
        """Returns the total number of ingested documents."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM document_metadata;")
            return cursor.fetchone()[0]

    def count_by_category(self) -> Dict[str, int]:
        """Returns document counts broken down by category."""
        counts = {"documents": 0, "images": 0, "videos": 0, "audio": 0}
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT category, COUNT(*) as count FROM document_metadata GROUP BY category;"
            )
            for row in cursor.fetchall():
                cat = row["category"]
                counts[cat] = row["count"]
        return counts

    def get_total_storage_bytes(self) -> int:
        """Returns total size in bytes of all stored documents."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COALESCE(SUM(file_size_bytes), 0) FROM document_metadata;")
            return cursor.fetchone()[0]

    # ------------------------------------------------------------------
    # Chunks persistence helpers (Phase 4)
    # ------------------------------------------------------------------

    def save_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> bool:
        """
        Persists preprocessed chunks for a document in SQLite.
        Replaces any prior chunks for this document to ensure idempotency.
        """
        if not chunks:
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            # Delete any existing chunks for this document first
            conn.execute("DELETE FROM document_chunks WHERE document_id = ?;", (document_id,))

            insert_rows = []
            for idx, c in enumerate(chunks):
                chunk_idx = c.get("chunk_index", idx)
                chunk_id = c.get("chunk_id", f"{document_id}_chunk_{chunk_idx}")
                text = c.get("text", "")
                page_no = c.get("page_number") or 0
                tokens = c.get("estimated_tokens", 0)

                # Store metadata payload without the heavy raw text to avoid duplication
                meta_copy = {k: v for k, v in c.items() if k not in ("text", "embedding")}
                meta_json = json.dumps(meta_copy, ensure_ascii=False)

                insert_rows.append((
                    chunk_id,
                    document_id,
                    chunk_idx,
                    text,
                    page_no,
                    tokens,
                    now_iso,
                    meta_json,
                ))

            conn.executemany(
                """
                INSERT OR REPLACE INTO document_chunks (
                    chunk_id, document_id, chunk_index, text, page_number,
                    estimated_tokens, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                insert_rows,
            )
            conn.commit()

        logger.info("Persisted %d chunks for doc_id=%s in SQLite.", len(chunks), document_id)
        return True

    def get_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieves persisted chunks for a document sorted by chunk_index."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT chunk_id, document_id, chunk_index, text, page_number,
                       estimated_tokens, created_at, metadata_json
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC;
                """,
                (document_id,),
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = {
                    "chunk_id": r["chunk_id"],
                    "document_id": r["document_id"],
                    "chunk_index": r["chunk_index"],
                    "text": r["text"],
                    "page_number": r["page_number"],
                    "estimated_tokens": r["estimated_tokens"],
                    "created_at": r["created_at"],
                }
                if r["metadata_json"]:
                    try:
                        extra = json.loads(r["metadata_json"])
                        item.update(extra)
                    except Exception:
                        pass
                results.append(item)
            return results

    def delete_chunks(self, document_id: str) -> int:
        """Deletes all persisted chunks for a document."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM document_chunks WHERE document_id = ?;", (document_id,))
            conn.commit()
            return cursor.rowcount

    def count_chunks(self, document_id: Optional[str] = None) -> int:
        """Returns total chunk count across all documents, or for a specific document."""
        with self._get_connection() as conn:
            if document_id:
                cursor = conn.execute("SELECT COUNT(*) FROM document_chunks WHERE document_id = ?;", (document_id,))
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM document_chunks;")
            return cursor.fetchone()[0]

    def delete(self, document_id: str) -> bool:
        """Deletes document metadata, extraction records, and chunks by ID."""
        with self._get_connection() as conn:
            cur_chk = conn.execute(
                "DELETE FROM document_chunks WHERE document_id = ?;",
                (document_id,),
            )
            cur_ext = conn.execute(
                "DELETE FROM document_extraction WHERE document_id = ?;",
                (document_id,),
            )
            cur_doc = conn.execute(
                "DELETE FROM document_metadata WHERE document_id = ?;",
                (document_id,),
            )
            conn.commit()
            return (cur_doc.rowcount > 0) or (cur_ext.rowcount > 0) or (cur_chk.rowcount > 0)
