# Backend Module

Server-side functionality, API services, database integration, and backend business logic for UptoSkills.

## Guidelines
- All server-side endpoints, authentication, database models, and background tasks reside under `backend/`.
- Maintain strict separation from the `frontend/` codebase.

## RAG Ingestion Pipeline (Phase 1: Document Upload & Validation)

The `backend/ingestion` package implements Phase 1 of the enterprise RAG ingestion pipeline:
- **`FileValidator` (`backend/ingestion/validator.py`)**: Format & extension checks, configurable category size limits, empty & corrupted file detection, magic bytes/signature verification, and safe filename sanitization.
- **`StorageManager` (`backend/ingestion/storage.py`)**: Generates unique document IDs (`doc_<uuid>`), routes files to categorized directories (`documents/`, `images/`, `videos/`), prevents accidental overwrites, and maintains persistent storage.
- **`MetadataManager` (`backend/ingestion/metadata.py`)**: SQLite-backed indexed store for document tracking, instant SHA-256 duplicate detection, category filtering, search, and storage diagnostics.
- **`IngestionPipeline` (`backend/ingestion/pipeline.py`)**: Orchestrator coordinating validation, deduplication, storage, and metadata recording.

### Running the Streamlit Interface
To test and inspect the ingestion pipeline interactively:
```bash
python -m streamlit run backend/streamlit_app.py
```

### Running Tests
Execute unit and integration test suite:
```bash
python -m pytest backend/tests -v
```

