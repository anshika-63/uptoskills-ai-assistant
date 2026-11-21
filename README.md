# UptoSkills AI Assistant

UptoSkills AI Assistant is a modular application architecture for the UptoSkills platform, with frontend feature modules and a backend document ingestion and processing pipeline designed to prepare uploaded content for downstream search, RAG, and vector-indexing workflows.

---

## Project Overview

The project is organized into independent frontend feature modules and a dedicated backend service.

The backend currently provides a document ingestion pipeline that supports file validation, duplicate detection, persistent storage, text/content extraction, preprocessing, chunking, metadata management, and processing-status tracking.

The ingestion pipeline is designed to transform uploaded documents and media into structured, searchable data that can be consumed by downstream AI/RAG components.

---

## Project Architecture

```text
uptoskills-ai-assistant/
│
├── frontend/
│   ├── dashboard/
│   │   └── README.md
│   │
│   ├── assessment/
│   │   └── README.md
│   │
│   ├── internship/
│   │   └── README.md
│   │
│   └── ai-assistant/
│       └── README.md
│
├── backend/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── audio_transcriber.py
│   │   ├── metadata.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── preprocessing.py
│   │   ├── storage.py
│   │   ├── text_extractor.py
│   │   └── validator.py
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_extraction.py
│   │   ├── test_metadata.py
│   │   ├── test_pipeline.py
│   │   ├── test_storage.py
│   │   └── test_validator.py
│   │
│   ├── storage/
│   │   └── metadata.db
│   │
│   ├── config.py
│   ├── requirements.txt
│   └── streamlit_app.py
│
├── .gitignore
└── README.md
```

---

# Frontend Modules

## Dashboard

`frontend/dashboard/`

Provides the dashboard module for user analytics, progress monitoring, metrics, and widgets.

## Assessment

`frontend/assessment/`

Provides assessment-related workflows including skill evaluations, tests, and scorecards.

## Internship

`frontend/internship/`

Provides internship discovery, application workflows, and project tracking.

## AI Assistant

`frontend/ai-assistant/`

Provides the frontend module for the conversational AI assistant and its supporting components.

Detailed documentation for each frontend module is maintained in its respective README.

---

# Backend

The backend contains the document ingestion and processing pipeline used to validate, store, extract, process, and prepare uploaded content for downstream AI/RAG workflows.

Backend implementation is located under:

```text
backend/
```

---

# Document Ingestion Pipeline

The ingestion pipeline follows this general workflow:

```text
Upload
   ↓
File Validation
   ↓
Duplicate Detection
   ↓
Persistent File Storage
   ↓
Content Extraction
   ↓
Text Cleaning / Preprocessing
   ↓
Chunk Generation
   ↓
Metadata Generation
   ↓
Persistent Metadata Storage
   ↓
Processing Status
   ↓
Ready for Search / Vector Indexing
```

---

# Supported File Types

The ingestion pipeline supports multiple content categories.

## Documents

- PDF
- DOCX
- TXT

## Images

- PNG
- JPG
- JPEG
- WEBP

## Audio

Supported audio formats are handled through the audio transcription module.

## Video

Supported video formats can be processed through the content extraction pipeline.

Exact supported extensions and maximum file sizes are controlled by the backend configuration.

---

# PDF Processing

PDF processing is a primary part of the document ingestion workflow.

The PDF pipeline supports:

- PDF file-type validation
- File-extension validation
- File-size validation
- Empty-file detection
- Corrupted-file detection
- File-signature / magic-byte verification
- Safe filename sanitization
- SHA-256 based duplicate detection
- Persistent file storage
- PDF text extraction
- Page-level information
- Page count
- Document metadata
- Text cleaning and normalization
- Content chunking
- Chunk-level processing
- Persistent metadata storage
- Processing-status tracking
- Preparation for downstream search and vector indexing

---

# PDF Processing Flow

```text
PDF Upload
    ↓
Validate PDF
    ↓
Check File Size
    ↓
Verify File Signature
    ↓
Calculate SHA-256
    ↓
Check for Duplicate
    ↓
Store File
    ↓
Extract PDF Text
    ↓
Capture Page Information
    ↓
Clean Extracted Text
    ↓
Generate Chunks
    ↓
Generate Document Metadata
    ↓
Persist Metadata
    ↓
Track Processing Status
    ↓
Prepare Data for Search / Vector Indexing
```

---

# Backend Components

## File Validation

### `backend/ingestion/validator.py`

The `FileValidator` validates uploaded files before they enter the ingestion pipeline.

Validation includes:

- Supported file extension checks
- File-type validation
- Configurable file-size limits
- Empty-file detection
- Corrupted-file detection
- Magic-byte / file-signature verification
- Safe filename sanitization

---

## Text and Content Extraction

### `backend/ingestion/text_extractor.py`

The text extraction module extracts content from supported file formats.

For PDF documents, extraction provides:

- Extracted text
- Page information
- Page count
- Document metadata
- Extraction statistics
- Processing information

Extracted content can be persisted and reused for inspection and downstream processing.

---

## Text Preprocessing and Chunking

### `backend/ingestion/preprocessing.py`

The preprocessing module prepares extracted content for downstream search and RAG workflows.

Responsibilities include:

- Removing unnecessary formatting
- Cleaning extracted text
- Normalizing whitespace
- Reducing structural noise
- Splitting content into smaller chunks
- Preparing chunk-level information

The generated chunks provide the intermediate document representation required for downstream semantic search and vector indexing.

---

## Metadata Management

### `backend/ingestion/metadata.py`

The `MetadataManager` provides persistent SQLite-backed document tracking.

It manages information such as:

- Document IDs
- Original filenames
- File metadata
- File categories
- File sizes
- SHA-256 hashes
- Processing information
- Extraction information
- Duplicate detection
- Document lookup
- Category filtering
- Search
- Storage diagnostics

---

## Storage Management

### `backend/ingestion/storage.py`

The `StorageManager` handles persistent file storage.

Responsibilities include:

- Generating unique document IDs
- Organizing uploaded files
- Routing files to appropriate storage locations
- Preventing accidental overwrites
- Maintaining persistent file paths

---

## Ingestion Pipeline

### `backend/ingestion/pipeline.py`

The `IngestionPipeline` coordinates the complete document-processing workflow.

It integrates:

1. File validation
2. Duplicate detection
3. File storage
4. Content extraction
5. Text preprocessing
6. Chunk generation
7. Metadata generation
8. Persistent metadata storage
9. Processing-status tracking

---

## Data Models

### `backend/ingestion/models.py`

Defines the data structures used throughout the ingestion workflow.

These models represent information such as:

- Validation results
- Document information
- Extraction results
- Processing state
- Chunk information
- Metadata

---

## Audio Transcription

### `backend/ingestion/audio_transcriber.py`

Provides audio transcription support for supported audio formats using the configured transcription engine.

The resulting transcript can be processed through the same downstream cleaning and chunking workflow.

---

# Persistent Storage

Document metadata is maintained using SQLite.

The current backend storage location is:

```text
backend/storage/metadata.db
```

Persistent metadata allows uploaded documents and their processing information to be tracked across application sessions.

---

# Processing Status

The ingestion pipeline tracks document processing state so that the application can distinguish between different stages of document handling.

Processing information can be used to identify whether a document has been:

- Validated
- Stored
- Extracted
- Processed
- Successfully completed
- Marked as failed

This status information supports document inspection, troubleshooting, and future re-processing workflows.

---

# Streamlit Verification Dashboard

The backend includes an interactive Streamlit dashboard for testing and inspecting the ingestion pipeline.

The dashboard allows users to interact with uploaded files and inspect information such as:

- Upload results
- Validation results
- Duplicate detection
- Stored documents
- Extraction results
- Document metadata
- Page information
- Generated chunks
- Processing status

---

# Environment Setup

## 1. Create the virtual environment

From the project root:

```bash
python3 -m venv backend/.venv
```

## 2. Activate the virtual environment

### macOS / Linux

```bash
source backend/.venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install -r backend/requirements.txt
```

---

# Running the Application

From the project root, activate the virtual environment:

```bash
source backend/.venv/bin/activate
```

Then start the Streamlit verification dashboard:

```bash
./backend/.venv/bin/python3 -m streamlit run backend/streamlit_app.py --server.port 8520
```

The dashboard will run on port `8520`.

---

# Running Tests

The project includes automated unit and integration tests covering the major ingestion components.

Run the complete test suite:

```bash
./backend/.venv/bin/python3 -m pytest -q
```

Current verified result:

```text
49 passed, 5 warnings
```

All automated tests pass successfully.

The reported warnings are dependency deprecation warnings and are not test failures.

---

# Test Coverage

The backend test suite covers major areas of the ingestion system, including:

- File validation
- File type validation
- File-size validation
- Metadata management
- Storage behavior
- Duplicate detection
- Pipeline behavior
- Text extraction
- Document processing
- Chunking-related processing

---

# Manual PDF Verification

A manual PDF ingestion test was successfully completed using:

```text
Automatic Rain Sensing Roof (1).pdf
```

Verified results:

```text
Pages:              10
Extracted characters: 3,102
Extracted words:      492
Generated chunks:     10
Extraction status:    Successful
```

This confirms that the implemented PDF pipeline can successfully validate, extract, process, and chunk a real PDF document.

---

# Development Notes

The project uses a Python virtual environment for backend dependencies.

Generated files and local development artifacts should remain outside version control.

Typical generated directories include:

```text
backend/.venv/
__pycache__/
.pytest_cache/
```

These should be excluded through `.gitignore`.

---

# Repository Documentation

The repository contains additional module-level documentation:

```text
backend/README.md
frontend/dashboard/README.md
frontend/assessment/README.md
frontend/internship/README.md
frontend/ai-assistant/README.md
```

The root `README.md` provides the overall project documentation, while module-level README files provide documentation specific to individual components.

---

# Current Implementation Status

The current backend ingestion implementation provides:

- File validation
- File-size validation
- File signature verification
- Safe filename handling
- SHA-256 duplicate detection
- Persistent file storage
- Document metadata management
- PDF text extraction
- Page-level PDF information
- Text preprocessing
- Text chunking
- Persistent extraction and metadata storage
- Processing-status tracking
- Audio transcription support
- Interactive Streamlit verification dashboard
- Automated regression tests

Current automated test status:

```text
49 passed
```

The document ingestion pipeline is ready to provide structured document data for downstream search and vector-indexing components.# UptoSkills AI Assistant

A modern, scalable architecture for the UptoSkills platform featuring modular frontend applications and a dedicated backend service.

## Project Architecture

```
uptoskills-ai-assistant/
│
├── frontend/
│   ├── dashboard/       # Dashboard pages, metrics, and widgets
│   ├── assessment/      # Skill assessments and testing workflows
│   ├── internship/      # Internship portal and project tracking
│   └── ai-assistant/    # Intelligent conversational assistant
│       ├── components/  # Chat window, message bubbles, input controls
│       ├── services/    # API clients, LLM integration, response handlers
│       ├── hooks/       # Conversation state, streaming, session hooks
│       └── styles/      # AI assistant-specific CSS and styling
│
└── backend/             # Server-side APIs, database, and business logic
```

## Module Overview

- **`frontend/dashboard/`**: User dashboard, analytics, progress monitoring, and widgets.
- **`frontend/assessment/`**: Candidate assessments, skill evaluations, tests, and scorecards.
- **`frontend/internship/`**: Internship discovery, application pipelines, and project milestones.
- **`frontend/ai-assistant/`**: Feature-isolated AI assistant with modular components, services, hooks, and styles.
- **`backend/`**: Server-side APIs, business logic, authentication, and database services.
