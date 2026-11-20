"""
Data models and schemas for the RAG document ingestion pipeline.
Defines types for validation results, document metadata, categories, and pipeline responses.
"""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class FileCategory(str, Enum):
    """Supported file categories for document routing."""
    DOCUMENTS = "documents"
    IMAGES = "images"
    VIDEOS = "videos"


class ValidationStatus(str, Enum):
    """Status outcomes for document validation."""
    VALID = "valid"
    INVALID = "invalid"
    DUPLICATE = "duplicate"


class ValidationResult(BaseModel):
    """Result of file validation."""
    is_valid: bool
    is_duplicate: bool = False
    status: ValidationStatus = ValidationStatus.VALID
    category: Optional[str] = None
    file_type: Optional[str] = None
    mime_type: Optional[str] = None
    file_size_bytes: int = 0
    sha256_hash: Optional[str] = None
    safe_filename: Optional[str] = None
    error_message: Optional[str] = None
    existing_document_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    """Persistent metadata schema for ingested documents."""
    document_id: str
    original_filename: str
    safe_filename: str
    category: str
    file_type: str
    mime_type: str
    file_size_bytes: int
    sha256_hash: str
    storage_path: str
    upload_timestamp: str
    status: str = "stored"
    metadata_json: Optional[str] = None



class IngestionResult(BaseModel):
    """Final outcome of the ingestion pipeline for a single file."""
    success: bool
    document_id: Optional[str] = None
    validation_result: ValidationResult
    metadata: Optional[DocumentMetadata] = None
    extracted_content: Optional[Dict[str, Any]] = None
    preprocessed_content: Optional[list] = None
    extraction_error: Optional[str] = None
    message: str