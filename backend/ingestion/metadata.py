"""
Metadata management engine for the RAG document ingestion pipeline.
Provides persistent storage, retrieval, indexing, duplicate detection,
and aggregation using SQLite.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Union

from backend.config import METADATA_DB_PATH
from backend.ingestion.models import DocumentMetadata


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
            # Create extraction table and index
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_extraction (
                    document_id TEXT PRIMARY KEY,
                    extraction_status TEXT NOT NULL,
                    char_count INTEGER,
                    word_count INTEGER,
                    page_count INTEGER,
                    extracted_at TEXT NOT NULL,
                    extracted_json TEXT,
                    error_message TEXT,
                    FOREIGN KEY (document_id) REFERENCES document_metadata (document_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ext_status ON document_extraction(extraction_status);"
            )
            conn.commit()

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

    def save_extraction(
        self,
        document_id: str,
        extraction_status: str,
        extracted_content: Optional[dict] = None,
        error_message: Optional[str] = None,
        extracted_at: Optional[str] = None,
    ) -> bool:
        """
        Saves or updates extraction results for a document in SQLite.
        """
        from datetime import datetime, timezone

        if extracted_at is None:
            extracted_at = datetime.now(timezone.utc).isoformat()

        char_count = None
        word_count = None
        page_count = None
        extracted_json = None

        if extracted_content:
            char_count = extracted_content.get("char_count")
            if char_count is None and "raw_text" in extracted_content:
                char_count = len(extracted_content["raw_text"])

            word_count = extracted_content.get("word_count")
            if word_count is None and extracted_content.get("raw_text"):
                word_count = len(extracted_content["raw_text"].split())

            page_count = extracted_content.get("total_pages")
            if page_count is None and "pages" in extracted_content:
                page_count = len(extracted_content["pages"])

            try:
                extracted_json = json.dumps(extracted_content)
            except Exception:
                extracted_json = None

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO document_extraction (
                    document_id,
                    extraction_status,
                    char_count,
                    word_count,
                    page_count,
                    extracted_at,
                    extracted_json,
                    error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    document_id,
                    extraction_status,
                    char_count,
                    word_count,
                    page_count,
                    extracted_at,
                    extracted_json,
                    error_message,
                ),
            )
            conn.commit()
            return True

    def get_extraction(self, document_id: str) -> Optional[dict]:
        """
        Retrieves the full extracted content dictionary for a document.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT extracted_json FROM document_extraction WHERE document_id = ? LIMIT 1;",
                (document_id,),
            )
            row = cursor.fetchone()
            if row and row["extracted_json"]:
                try:
                    return json.loads(row["extracted_json"])
                except Exception:
                    return None
        return None

    def get_extraction_stats(self, document_id: str) -> Optional[dict]:
        """
        Retrieves summary extraction metadata/stats for a document without loading the full JSON payload.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT extraction_status, char_count, word_count, page_count, extracted_at, error_message
                FROM document_extraction
                WHERE document_id = ?
                LIMIT 1;
                """,
                (document_id,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def delete(self, document_id: str) -> bool:
        """Deletes document metadata and extraction records by ID."""
        with self._get_connection() as conn:
            cur_ext = conn.execute(
                "DELETE FROM document_extraction WHERE document_id = ?;",
                (document_id,),
            )
            cur_doc = conn.execute(
                "DELETE FROM document_metadata WHERE document_id = ?;",
                (document_id,),
            )
            conn.commit()
            return (cur_doc.rowcount > 0) or (cur_ext.rowcount > 0)
