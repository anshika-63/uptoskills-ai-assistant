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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_extraction (
                    document_id       TEXT PRIMARY KEY,
                    file_type         TEXT NOT NULL,
                    raw_text          TEXT NOT NULL DEFAULT '',
                    char_count        INTEGER NOT NULL DEFAULT 0,
                    word_count        INTEGER NOT NULL DEFAULT 0,
                    page_count        INTEGER NOT NULL DEFAULT 0,
                    extraction_status TEXT NOT NULL DEFAULT 'success',
                    extracted_at      TEXT NOT NULL,
                    payload_json      TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ext_id ON document_extraction(document_id);"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Extraction persistence helpers
    # ------------------------------------------------------------------

    def save_extraction(
        self,
        document_id: str,
        extraction: Dict[str, Any],
        extraction_status: str = "success",
    ) -> bool:
        """
        Persist the full extraction payload for a document.

        Args:
            document_id: ID of the parent document record.
            extraction:  Dict returned by text_extractor.extract_text().
            extraction_status: 'success' or 'failed'.

        The payload is stored as JSON in `payload_json`.  Key statistics
        (raw_text, char_count, word_count, page_count) are promoted to
        dedicated columns so they can be queried without deserialising JSON.

        Returns True on success.
        """
        file_type   = extraction.get("file_type", "")
        raw_text    = extraction.get("raw_text", "")
        char_count  = extraction.get("char_count", len(raw_text))
        word_count  = extraction.get("word_count", len(raw_text.split()) if raw_text else 0)
        page_count  = extraction.get("total_pages", 1 if file_type != "pdf" else 0)
        extracted_at = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO document_extraction (
                    document_id, file_type, raw_text, char_count, word_count,
                    page_count, extraction_status, extracted_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    document_id,
                    file_type,
                    raw_text,
                    char_count,
                    word_count,
                    page_count,
                    extraction_status,
                    extracted_at,
                    json.dumps(extraction, ensure_ascii=False),
                ),
            )
            conn.commit()
        logger.info(
            "Extraction saved: doc_id=%s file_type=%s chars=%d words=%d",
            document_id, file_type, char_count, word_count,
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
            if row:
                return json.loads(row["payload_json"])
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
                "extraction_status, extracted_at "
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

    def delete(self, document_id: str) -> bool:
        """Deletes document metadata record by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM document_metadata WHERE document_id = ?;",
                (document_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
