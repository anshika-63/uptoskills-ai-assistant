# Backend Module

Server-side functionality, API services, database integration, and backend business logic for UptoSkills.

## Guidelines
- All server-side endpoints, authentication, database models, and background tasks reside under `backend/`.
- Maintain strict separation from the `frontend/` codebase.

## RAG Ingestion Pipeline

The `backend/ingestion` package implements the enterprise RAG ingestion pipeline:
- **`FileValidator` (`backend/ingestion/validator.py`)**: Format & extension checks across Documents, Images, Videos, and Audio (`.mp3`, `.wav`, `.m4a`, `.aac`, `.ogg`, `.flac`), configurable category size limits, empty & corrupted file detection, magic bytes/signature verification, and safe filename sanitization.
- **`StorageManager` (`backend/ingestion/storage.py`)**: Generates unique document IDs (`doc_<uuid>`), routes files to categorized directories (`documents/`, `images/`, `videos/`, `audio/`), prevents accidental overwrites, and maintains persistent storage including transcript JSON exports.
- **`MetadataManager` (`backend/ingestion/metadata.py`)**: SQLite-backed indexed store for document tracking, instant SHA-256 duplicate detection, category filtering, search, and storage diagnostics. Houses `document_metadata` and `document_extraction` tables.
- **`TextExtractor` (`backend/ingestion/text_extractor.py`)**: Structured text extraction for PDF (PyMuPDF), DOCX (python-docx, 3D table structure), and TXT files.
- **`AudioTranscriber` (`backend/ingestion/audio_transcriber.py`)**: Speech-to-Text transcription via faster-whisper, FFmpeg preprocessing (resampling/conversion to 16kHz mono WAV), timestamped segmented transcript generation, and reusable video-audio extraction utilities.
- **`IngestionPipeline` (`backend/ingestion/pipeline.py`)**: Orchestrator coordinating validation, deduplication, storage, text extraction, audio transcription, and persistent metadata recording.

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

