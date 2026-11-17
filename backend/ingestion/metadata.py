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
        counts = {"documents": 0, "images": 0, "videos": 0}
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
