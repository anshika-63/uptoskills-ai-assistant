"""
Storage manager for the RAG document ingestion pipeline.
Responsible for generating unique document IDs, directory creation,
categorized file persistence, and accidental overwrite prevention.
"""

import os
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple, Union

from backend.config import CATEGORY_DIRECTORIES, STORAGE_BASE_DIR


class StorageManager:
    """Manages persistent disk storage for validated documents."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        self.base_dir = Path(base_dir or STORAGE_BASE_DIR).resolve()
        self._ensure_storage_structure()

    def _ensure_storage_structure(self) -> None:
        """Creates the root and category directories if they do not exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for cat_dir in CATEGORY_DIRECTORIES.values():
            (self.base_dir / cat_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_document_id() -> str:
        """
        Generates a unique, standardized document identifier.
        Example: 'doc_5b1a3f6289d04ef1'
        """
        return f"doc_{uuid.uuid4().hex[:16]}"

    def get_category_path(self, category: str) -> Path:
        """Returns the folder path for a specific category."""
        target_dir = CATEGORY_DIRECTORIES.get(category, category)
        folder = self.base_dir / target_dir
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_file(
        self,
        document_id: str,
        safe_filename: str,
        category: str,
        content: bytes,
    ) -> Tuple[Path, str]:
        """
        Persistently saves validated file content to category folder.
        Uses document_id prefix to prevent namespace collisions.
        Includes overwrite detection to prevent accidental loss.

        Returns:
            Tuple[Path, str]: (absolute_file_path, relative_storage_path)
        """
        cat_dir = self.get_category_path(category)

        # Standard filename pattern: doc_xxx_filename.ext
        target_filename = f"{document_id}_{safe_filename}"
        target_path = cat_dir / target_filename

        # Prevent accidental overwrite if a file with identical name exists
        if target_path.exists():
            timestamp_ns = int(time.time() * 1000)
            path_obj = Path(safe_filename)
            target_filename = f"{document_id}_{path_obj.stem}_{timestamp_ns}{path_obj.suffix}"
            target_path = cat_dir / target_filename

        # Write binary content
        with open(target_path, "wb") as f:
            f.write(content)

        relative_path = str(target_path.relative_to(self.base_dir)).replace("\\", "/")
        return target_path, relative_path

    def _resolve_safe_path(self, path: Union[str, Path]) -> Path:
        """Resolves path and enforces that it remains within base_dir."""
        p = Path(path)
        if not p.is_absolute():
            p = self.base_dir / p
        resolved = p.resolve()
        if not resolved.is_relative_to(self.base_dir):
            raise PermissionError(f"Access denied: path '{path}' escapes storage directory.")
        return resolved

    def file_exists(self, path: Union[str, Path]) -> bool:
        """Checks if a file exists given an absolute or relative path."""
        try:
            p = self._resolve_safe_path(path)
            return p.exists() and p.is_file()
        except PermissionError:
            return False

    def read_file(self, path: Union[str, Path]) -> bytes:
        """Reads file bytes from disk."""
        p = self._resolve_safe_path(path)
        with open(p, "rb") as f:
            return f.read()

    def delete_file(self, path: Union[str, Path]) -> bool:
        """Deletes a file if it exists."""
        p = self._resolve_safe_path(path)
        if p.exists() and p.is_file():
            p.unlink()
            return True
        return False
