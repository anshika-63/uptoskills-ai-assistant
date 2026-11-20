# RECOVERY EVIDENCE & IMPLEMENTATION AUDIT REPORT

**Project Name:** `uptoskills-ai-assistant`  
**Repository Path:** `/Users/anshika/Desktop/uptoskills-ai-assistant`  
**Report Date:** November 21, 2026  
**Auditor:** Antigravity Forensic Engineering Agent (DeepMind)  
**Standard of Verification:** Non-Destructive Code & Git History Inspection  
**Report Goal:** Concrete technical evidence to enable safe restoration, bug remediation, and pipeline completion.

---

## TABLE OF CONTENTS

1. [Part 1 — Verification of Prior Audit Claims](#part-1--verification-of-prior-audit-claims)
2. [Part 2 — Current Repository Snapshot](#part-2--current-repository-snapshot)
3. [Part 3 — Current Streamlit App (`backend/streamlit_app.py`)](#part-3--current-streamlit-app-backendstreamlit_apppy)
4. [Part 4 — Last Known Good Streamlit Version (`efafcf0`)](#part-4--last-known-good-streamlit-version-efafcf0)
5. [Part 5 — Deep Investigation of Commit `747baf3`](#part-5--deep-investigation-of-commit-747baf3)
6. [Part 6 — PR & Merge History Analysis (`aac8b13`, `747baf3`, `3d7141e`, `efafcf0`)](#part-6--pr--merge-history-analysis-aac8b13-747baf3-3d7141e-efafcf0)
7. [Part 7 — Audio & Transcription Forensics](#part-7--audio--transcription-forensics)
8. [Part 8 — Database Schema & State Forensics](#part-8--database-schema--state-forensics)
9. [Part 9 — End-to-End Ingestion Pipeline Trace](#part-9--end-to-end-ingestion-pipeline-trace)
10. [Part 10 — Document Extraction Forensics](#part-10--document-extraction-forensics)
11. [Part 11 — Chunking Forensics & Edge Cases](#part-11--chunking-forensics--edge-cases)
12. [Part 12 — Test Coverage Matrix](#part-12--test-coverage-matrix)
13. [Part 13 — Environment vs. Code Bug Disambiguation](#part-13--environment-vs-code-bug-disambiguation)
14. [Part 14 — Requirements & Dependency Reconciliation](#part-14--requirements--dependency-reconciliation)
15. [Part 15 — Security & Data Risk Inventory](#part-15--security--data-risk-inventory)
16. [Part 16 — Safest Recovery Strategy Comparison](#part-16--safest-recovery-strategy-comparison)
17. [Part 17 — Exact Step-by-Step Recovery Checklist](#part-17--exact-step-by-step-recovery-checklist)
18. [Part 18 — "Do Not Touch" Stability List](#part-18--do-not-touch-stability-list)
19. [Part 19 — File-by-File Remediation Matrix](#part-19--file-by-file-remediation-matrix)
20. [Part 20 — Final Recovery Verdict](#part-20--final-recovery-verdict)

---

## PART 1 — VERIFICATION OF PRIOR AUDIT CLAIMS

This section audits every major claim made in `UPTOSKILLS_PROJECT_FORENSIC_AUDIT.md` against live repository evidence.

| # | Claim in Forensic Audit | Evidence Given by Audit | Independent Re-Verification & Evidence | Status |
|---|---|---|---|---|
| 1 | `backend/streamlit_app.py` fails on launch with `ModuleNotFoundError` for `audio_transcriber`. | Python import test failing at line 33. | Executed `./backend/.venv/bin/python3 -c "import backend.streamlit_app"` -> `ModuleNotFoundError: No module named 'backend.ingestion.audio_transcriber'`. | **VERIFIED** |
| 2 | `EXTRACTABLE_EXTENSIONS` is imported in `streamlit_app.py` line 38 but missing from `text_extractor.py`. | Python import test failing on attribute import. | Executed `./backend/.venv/bin/python3 -c "from backend.ingestion.text_extractor import EXTRACTABLE_EXTENSIONS"` -> `ImportError: cannot import name 'EXTRACTABLE_EXTENSIONS'`. | **VERIFIED** |
| 3 | Tesseract OCR is missing from the system PATH, breaking image, video, and scanned PDF extraction. | `which tesseract` returned command not found; Python test threw `TesseractNotFoundError`. | Executed `which tesseract` -> not found. Executed `pytesseract.image_to_string(Image.new('RGB', (10, 10)))` -> raised `TesseractNotFoundError`. | **VERIFIED** |
| 4 | Virtual environment `.venv` bin scripts fail with exit code 126 due to stale path with spaces. | Error: `/Users/anshika/Desktop/uptoskills ai assistant /...: No such file`. | Inspected `backend/.venv/bin/pytest` line 2: hardcoded path containing `uptoskills ai assistant `. `backend/.venv/pyvenv.cfg` confirms original command contained spaces. | **VERIFIED** |
| 5 | Table `document_extraction` referenced in `streamlit_app.py` does not exist in `metadata.db`. | SQLite schema inspection in `metadata.db`. | Executed `sqlite3 backend/storage/metadata.db "SELECT name FROM sqlite_master WHERE type='table'"` -> output `['document_metadata']`. Table `document_extraction` is non-existent. | **VERIFIED** |
| 6 | 38 tests pass when executed via `python3 -m pytest`. | Pytest run output showing 38 passed in 0.50s. | Executed `./backend/.venv/bin/python3 -m pytest backend/tests` -> 38 passed in 0.50s. | **VERIFIED** |
| 7 | No vector database, embedding models, or LLM integrations exist in the repository. | Codebase grep for Chroma, FAISS, LangChain, OpenAI. | Grepped all tracked files. Zero embedding, vector store, or LLM inference code exists. | **VERIFIED** |
| 8 | Commit `747baf3` was the breaking commit that introduced the broken Streamlit code. | Git diff showing 349 lines modified in `streamlit_app.py` with no backend changes. | `git show 747baf3` modifies only `streamlit_app.py` (+282/-67) and introduced all 6 broken references. | **VERIFIED** |
| 9 | Commit `efafcf0` contained a working version of `streamlit_app.py` displaying chunks. | `git show efafcf0:backend/streamlit_app.py` inspects previous code. | Verified line 324-370 of `efafcf0:backend/streamlit_app.py` contains preprocessed chunk expanders and no audio imports. | **VERIFIED** |
| 10 | Preprocessed chunks are not persisted to database or disk. | Code inspection of `pipeline.py#L240-L260`. | Inspected `IngestionPipeline.process_file()`: `preprocessed_content` is only placed in the in-memory `IngestionResult` object; never written to SQLite or storage. | **VERIFIED** |
| 11 | `verify_chunk_tokens` is duplicated three times in `backend/ingestion/preprocessing.py`. | Code inspection of `preprocessing.py`. | Lines 116, 273, and 293 in `preprocessing.py` define `verify_chunk_tokens` three times. | **VERIFIED** |
| 12 | Arbitrary file read / path traversal risk exists in `StorageManager.read_file()`. | `read_file()` does not check if path is within `base_dir`. | Inspected `backend/ingestion/storage.py#L88-L94`: if path is absolute, it opens it bare without containment validation. | **VERIFIED** |

---

## PART 2 — CURRENT REPOSITORY SNAPSHOT

### 2.1 Git Repository Status
- **Current Branch:** `main`
- **Current HEAD Commit:** `3d7141e3f1e84eae681d367ad763b457dd80e64a` (`Complete Phase 1 RAG ingestion pipeline`)
- **Remote Origin:** `https://github.com/anshika-63/uptoskills-ai-assistant.git` (fetch & push)
- **Branch Tracking Status:** `main` is ahead of `origin/main` by 1 commit (`3d7141e`).
- **Working Tree State:** Clean (no modified, staged, or deleted tracked files).
- **Stash List:** Clean (0 stash entries).
- **Recent Reflog Entries:**
  ```
  3d7141e HEAD@{0}: commit (amend): Complete Phase 1 RAG ingestion pipeline
  36d30dd HEAD@{1}: rebase (finish): returning to refs/heads/main
  36d30dd HEAD@{2}: rebase (continue): Complete Phase 1 RAG ingestion pipeline
  aac8b13 HEAD@{3}: pull --rebase origin main (start): checkout aac8b13ad34459ff892e202f18857098dcf234a7
  efafcf0 HEAD@{4}: commit (amend): Complete Phase 1 RAG ingestion pipeline
  b24f3ad HEAD@{5}: commit: Complete Phase 1 RAG ingestion pipeline
  bacb477 HEAD@{6}: commit: Display extracted document text in Streamlit
  6efb4ec HEAD@{7}: commit: Integrate PDF DOCX and TXT text extraction
  c01e0fa HEAD@{8}: commit: Add document text extraction module
  3663cd9 HEAD@{9}: pull origin main: Fast-forward
  ```

### 2.2 Complete File Tree, Sizes, and Modification Timestamps

```
Path                                                              Size (Bytes)   Last Modified
────────────────────────────────────────────────────────────────────────────────────────────────────
./.gitignore                                                               716   2026-11-18 18:39:20
./README.md                                                               1449   2026-11-20 20:33:31
./UPTOSKILLS_PROJECT_FORENSIC_AUDIT.md                                   65173   2026-11-21 01:04:22 [Untracked]
./test_frame.png                                                          4517   2026-11-20 22:43:28 [Untracked]
./test_image.png                                                        632232   2026-11-20 22:22:59 [Untracked]
./test_image.webp                                                        85398   2026-11-20 22:22:59 [Untracked]
./test_video.mp4                                                          5751   2026-11-20 22:26:51 [Untracked]
./test_video_with_text.mp4                                                7365   2026-11-20 22:43:54 [Untracked]

./frontend/assessment/README.md                                            260   2026-11-16 23:29:24
./frontend/dashboard/README.md                                             270   2026-11-16 23:29:24
./frontend/internship/README.md                                            285   2026-11-16 23:29:24
./frontend/ai-assistant/README.md                                          678   2026-11-16 23:25:54
./frontend/ai-assistant/styles/README.md                                    96   2026-11-16 23:29:24
./frontend/ai-assistant/components/README.md                               301   2026-11-16 23:29:24
./frontend/ai-assistant/hooks/README.md                                    278   2026-11-16 23:29:24
./frontend/ai-assistant/services/README.md                                 316   2026-11-16 23:29:24

./backend/README.md                                                       1547   2026-11-18 18:39:20
./backend/__init__.py                                                       49   2026-11-18 18:39:20
./backend/config.py                                                       2367   2026-11-18 18:39:20
./backend/requirements.txt                                                 155   2026-11-18 18:39:20
./backend/streamlit_app.py                                               28745   2026-11-21 00:32:21
./backend/ingestion/__init__.py                                            627   2026-11-18 18:39:20
./backend/ingestion/models.py                                             1924   2026-11-20 23:43:24
./backend/ingestion/validator.py                                         12420   2026-11-18 18:39:20
./backend/ingestion/storage.py                                            3579   2026-11-18 18:39:20
./backend/ingestion/metadata.py                                           7555   2026-11-18 18:39:20
./backend/ingestion/text_extractor.py                                     4891   2026-11-20 23:43:24
./backend/ingestion/preprocessing.py                                      8278   2026-11-20 23:43:24
./backend/ingestion/pipeline.py                                           9737   2026-11-21 00:36:27

./backend/tests/__init__.py                                                 50   2026-11-18 18:39:20
./backend/tests/conftest.py                                               4698   2026-11-18 18:39:20
./backend/tests/test_storage.py                                           2730   2026-11-18 18:39:20
./backend/tests/test_validator.py                                         7222   2026-11-18 18:39:20
./backend/tests/test_metadata.py                                          4353   2026-11-18 18:39:20
./backend/tests/test_pipeline.py                                          4513   2026-11-18 18:39:20

./backend/storage/metadata.db                                            32768   2026-11-20 23:00:47 [Runtime DB]
./backend/storage/uploads/ (18 stored files across documents, images, videos) 8.0 MB   2026-11-20 [Runtime Storage]
```

---

## PART 3 — CURRENT STREAMLIT APP (`backend/streamlit_app.py`)

A full, line-by-line inspection of all 664 lines of `backend/streamlit_app.py` was conducted.

### 3.1 Import Audit
- **Standard Library Imports (Succeed):** `hashlib` (L7), `json` (L8), `os` (L9), `sys` (L10), `pathlib.Path` (L11), `typing.Optional` (L12), `sqlite3` (L655).
- **Third-Party Imports (Succeed):** `pandas as pd` (L23), `streamlit as st` (L24).
- **Backend Ingestion Imports (Succeed):**
  - `backend.config` (`ALLOWED_EXTENSIONS`, `CATEGORY_DIRECTORIES`, `DEFAULT_MAX_SIZES_MB`, `METADATA_DB_PATH`, `STORAGE_BASE_DIR`) (L26-32)
  - `backend.ingestion.metadata.MetadataManager` (L34, L654)
  - `backend.ingestion.models.ValidationStatus` (L35)
  - `backend.ingestion.pipeline.IngestionPipeline` (L36)
  - `backend.ingestion.storage.StorageManager` (L37)
  - `backend.ingestion.validator.FileValidator` (L39)
- **Broken Imports (Fail):**
  - `from backend.ingestion.audio_transcriber import AUDIO_EXTENSIONS` (L33) -> **FATAL: Module does not exist.**
  - `from backend.ingestion.text_extractor import EXTRACTABLE_EXTENSIONS` (L38) -> **FATAL: Attribute does not exist in `text_extractor.py`.**

### 3.2 Functions Defined
1. `get_pipeline() -> IngestionPipeline` (L101-L111): Cached Streamlit resource initializing `FileValidator`, `StorageManager`, and `MetadataManager`.
2. `format_bytes(num_bytes: int) -> str` (L114-L123): Converts byte counts to B, KB, MB, GB.
3. `render_extraction(extracted: dict, document_id: Optional[str] = None, key_prefix: Optional[str] = None) -> None` (L126-L264): Renders extraction payload for PDF, DOCX, TXT, and audio. Generates stable element keys to avoid duplicate ID errors.

### 3.3 Complete Dependency & Invocation Map

```
streamlit_app.py
  │
  ├──► FileValidator()                     [backend/ingestion/validator.py]
  │      └── Validates format & headers
  │
  ├──► StorageManager()                     [backend/ingestion/storage.py]
  │      ├── read_file()                   [Reads stored binaries for preview]
  │      └── file_exists()                 [Checks persistence on disk]
  │
  ├──► MetadataManager()                    [backend/ingestion/metadata.py]
  │      ├── count_total()                 [KPI: total cataloged count]
  │      ├── count_by_category()           [KPI: counts per category]
  │      ├── get_total_storage_bytes()     [KPI: storage used]
  │      ├── list_documents()              [Table: paginated registry]
  │      └── get_extraction_stats()        [BROKEN: Method does not exist!]
  │
  ├──► IngestionPipeline()                  [backend/ingestion/pipeline.py]
  │      ├── process_file()                [Batch/single file processor]
  │      └── get_or_reextract()            [BROKEN: Method does not exist!]
  │
  ├──► audio_transcriber                   [BROKEN: Non-existent module!]
  │
  └──► sqlite3 direct query                 [backend/storage/metadata.db]
         └── SELECT COUNT(*) FROM document_extraction [BROKEN: Non-existent table!]
```

### 3.4 Broken References in Current `streamlit_app.py`

| Line | Code Reference | Problem | Consequence |
|---|---|---|---|
| **L33** | `from backend.ingestion.audio_transcriber import AUDIO_EXTENSIONS` | Module `audio_transcriber.py` does not exist. | Fatal startup crash (`ModuleNotFoundError`). |
| **L38** | `from backend.ingestion.text_extractor import EXTRACTABLE_EXTENSIONS` | `text_extractor.py` has no `EXTRACTABLE_EXTENSIONS`. | Fatal startup crash (`ImportError`). |
| **L290** | `ALLOWED_EXTENSIONS['audio']` | Key `'audio'` is not in `ALLOWED_EXTENSIONS`. | Fatal startup crash (`KeyError: 'audio'`). |
| **L290** | `DEFAULT_MAX_SIZES_MB['audio']` | Key `'audio'` is not in `DEFAULT_MAX_SIZES_MB`. | Fatal startup crash (`KeyError: 'audio'`). |
| **L365** | `r.upload_success` | Model `IngestionResult` has no `upload_success`. | Crash on file upload (`AttributeError`). |
| **L368** | `r.extraction_status == "success"` | Model `IngestionResult` has no `extraction_status`. | Crash on file upload (`AttributeError`). |
| **L388** | `if res.upload_success:` | Model `IngestionResult` has no `upload_success`. | Crash on file upload (`AttributeError`). |
| **L417** | `res.extraction_status` | Model `IngestionResult` has no `extraction_status`. | Crash on file upload (`AttributeError`). |
| **L494** | `metadata_mgr.get_extraction_stats(d.document_id)` | `MetadataManager` has no `get_extraction_stats`. | Crash on opening Dashboard (`AttributeError`). |
| **L544** | `metadata_mgr.get_extraction_stats(...)` | `MetadataManager` has no `get_extraction_stats`. | Crash on selecting document (`AttributeError`). |
| **L582** | `pipeline.get_or_reextract(...)` | `IngestionPipeline` has no `get_or_reextract`. | Crash on inspecting document (`AttributeError`). |
| **L637** | `ALLOWED_EXTENSIONS["audio"]` (Tab 3) | Key `'audio'` does not exist in config. | Crash on opening Audit tab (`KeyError`). |
| **L658** | `SELECT COUNT(*) FROM document_extraction` | SQLite table `document_extraction` does not exist. | SQL error caught by broad `except Exception`. |

---

## PART 4 — RECOVER THE LAST KNOWN GOOD STREAMLIT VERSION (`efafcf0`)

Historical commit `efafcf0` (`Complete Phase 1 RAG ingestion pipeline`, authored November 20, 2026) was inspected using `git show efafcf0:backend/streamlit_app.py`.

### 4.1 Structural Comparison: Current `streamlit_app.py` vs. `efafcf0:streamlit_app.py`

```
Feature / Section               efafcf0 (Last Known Good)               Current HEAD (3d7141e)
──────────────────────────────────────────────────────────────────────────────────────────────────
Imports: audio_transcriber      NOT PRESENT                             PRESENT (Line 33 - BROKEN)
Imports: EXTRACTABLE_EXTENSIONS NOT PRESENT                             PRESENT (Line 38 - BROKEN)
Config: audio references        NOT PRESENT                             PRESENT (Lines 290, 637 - BROKEN)
Models: upload_success          NOT USED (Uses r.success)               USED (Lines 365, 388 - BROKEN)
Models: extraction_status       NOT USED                                USED (Lines 368, 417 - BROKEN)
DB: document_extraction         NOT QUERIED                             QUERIED (Lines 658-659 - BROKEN)
Methods: get_extraction_stats   NOT CALLED                              CALLED (Lines 494, 544 - BROKEN)
Methods: get_or_reextract       NOT CALLED                              CALLED (Line 582 - BROKEN)
Rendering: Preprocessed Chunks  PRESENT & WORKING (Lines 324-370)       COMPLETELY MISSING (0 lines)
Rendering: Text Extraction      PRESENT in expanders (Lines 289-322)    PRESENT via render_extraction()
Rendering: Element Key Hashing  Basic expanders                         SHA-256 stable key hashing
Registry Columns                8 verified columns                      10 columns (includes 2 missing)
Category Filter Options         ["All", "Documents", "Images", "Videos"] Adds "Audio" (unsupported)
```

### 4.2 Code from `efafcf0` that Was Working and Lost
In `efafcf0`, lines 324-370 rendered the preprocessed chunks directly from `IngestionResult.preprocessed_content`:
```python
# Display preprocessed chunks (LOST IN CURRENT VERSION)
if res.preprocessed_content:
    st.markdown("#### 🧹 Preprocessed Text & Chunks")
    chunks = res.preprocessed_content
    total_chunks = len(chunks)
    within_limit = sum(1 for chunk in chunks if chunk.get("within_token_limit", False))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", total_chunks)
    col2.metric("Within Token Limit", within_limit)
    col3.metric("Over Token Limit", total_chunks - within_limit)

    for chunk in chunks:
        with st.expander(
            f"Chunk {chunk['chunk_index'] + 1} ({chunk.get('estimated_tokens', 0)} estimated tokens)"
        ):
            st.write(chunk["text"])
            st.caption(
                f"Document ID: {chunk['document_id']} | "
                f"Filename: {chunk['filename']} | "
                f"File Type: {chunk['file_type']} | "
                f"Page: {chunk.get('page_number', 'N/A')}"
            )
            st.caption(
                f"Token limit status: "
                f"{'Within limit' if chunk.get('within_token_limit') else 'Over limit'}"
            )
```

### 4.3 Code in Current Version that is Worth Retaining
In the current version, function `render_extraction()` (lines 126-264) introduced stable widget key generation using SHA-256 (`hashlib.sha256(content_repr).hexdigest()[:12]`). This prevents the `StreamlitDuplicateElementId` exception when multiple files with similar page counts or names are uploaded together.

---

## PART 5 — INVESTIGATE COMMIT `747baf3`

- **Commit ID:** `747baf3b6fd1d9b39f8d87d6408bfab3a6e89744`
- **Parent Commit:** `bacb4777b8e17c3242b874b45b587627adcef269`
- **Author:** Patel Dev (`pateldev102938gamil.com`)
- **Author Date:** Thu Nov 19 21:02:29 2026 +0530
- **Commit Message:** `fix: resolve Streamlit duplicate element ID error`

### 5.1 What Actually Changed
Running `git diff --stat 747baf3^ 747baf3` reveals:
```
backend/streamlit_app.py | 349 ++++++++++++++++++++++++++++++++++++++---------
1 file changed, 282 insertions(+), 67 deletions(-)
```
**Zero backend modules were changed.** No models, no database schemas, no validators, no audio packages, and no tests were touched.

### 5.2 Suspicious References Introduced by Commit `747baf3`

```
Suspicious Reference introduced     Target That Should Exist            Does Target Exist?
──────────────────────────────────────────────────────────────────────────────────────────
backend.ingestion.audio_transcriber backend/ingestion/audio_transcriber.py NO (Never committed)
EXTRACTABLE_EXTENSIONS              text_extractor.py attribute         NO
ALLOWED_EXTENSIONS["audio"]         backend/config.py dictionary key    NO
DEFAULT_MAX_SIZES_MB["audio"]       backend/config.py dictionary key    NO
r.upload_success                    IngestionResult field (models.py)   NO (Uses r.success)
r.extraction_status                 IngestionResult field (models.py)   NO
metadata_mgr.get_extraction_stats   MetadataManager method              NO
pipeline.get_or_reextract           IngestionPipeline method            NO
table document_extraction           SQLite table in metadata.db         NO
```

---

## PART 6 — PR & MERGE HISTORY ANALYSIS

### 6.1 Chronological Development Flow

```
2026-11-19 19:00:19 +0530 [bacb477]
  Author: Anshika Verma
  Commit: Display extracted document text in Streamlit
  Status: Base commit for both branches.
       │
       ├──► BRANCH A: Feature PR (Patel Dev)
       │      │
       │      └── 2026-11-19 21:02:29 +0530 [747baf3]
       │            Commit: fix: resolve Streamlit duplicate element ID error
       │            Modifies: ONLY backend/streamlit_app.py
       │            Injects: Uncommitted audio & DB dependencies
       │            │
       │            └── 2026-11-19 21:08:43 +0530 [aac8b13]
       │                  Merge PR #2 on GitHub web interface into origin/main
       │
       └──► BRANCH B: Local Development (Anshika Verma)
              │
              └── 2026-11-20 23:42:33 +0530 [b24f3ad / efafcf0]
                    Commit: Complete Phase 1 RAG ingestion pipeline
                    Introduces: preprocessing.py, recursive_chunking, OCR/video in text_extractor.py
                    Updates: streamlit_app.py (Displays preprocessed chunks cleanly)
                    Parent: bacb477 (Clean historical ancestor)
                    │
                    ▼ (Anshika runs `git pull --rebase origin main`)
              2026-11-20 23:42:33 +0530 [36d30dd -> amended to 3d7141e]
                    Rebase conflict occurred on backend/streamlit_app.py.
                    Resolution selected origin/main version (aac8b13 / 747baf3).
                    Anshika's working streamlit_app.py was completely overwritten.
```

### 6.2 Forensic Verdict on Git Merge
- **Did a merge overwrite working code?** **YES.** During `pull --rebase origin main`, the conflict resolution on `backend/streamlit_app.py` discarded Anshika's working UI (blob `a2a6d7a` in `efafcf0`) and kept Patel Dev's unintegrated, broken UI (blob `9ba2dc3` in `aac8b13`).
- **Was uncommitted work referenced?** **YES.** Patel Dev wrote UI code referencing an audio transcription subsystem and a `document_extraction` SQLite table that were never committed to any branch.

---

## PART 7 — AUDIO / TRANSCRIPTION FORENSICS

A comprehensive search was performed across all Git branches, commits, reflog objects, and unstaged files.

### 7.1 Search Results
1. `git log -S "audio_transcriber" --all --oneline`: Only matches commit `747baf3`.
2. `git log --all --full-history -- "**/audio_transcriber*"`: **0 files returned.**
3. `git log -S "AUDIO_EXTENSIONS" --all --oneline`: Only matches commit `747baf3`.
4. `git log -S "whisper" --all --oneline`: Only matches commit `747baf3` (in UI text only).
5. `git log -S "ffmpeg" --all --oneline`: **0 commits returned.**

### 7.2 Forensic Answers
1. **Does the implementation currently exist?** **NO.**
2. **Did it ever exist in Git history?** **NO.** There is no commit in the entire Git history of this repository that ever created `audio_transcriber.py`.
3. **Was it deleted?** **NO.** It was never added.
4. **Was it referenced before being implemented?** **YES.** It was introduced in commit `747baf3` as an external import without the file being created.
5. **Which commit first introduced the reference?** `747baf3b6fd1d9b39f8d87d6408bfab3a6e89744`.
6. **Can it be recovered from Git history?** **NO.** There are no git blobs containing this file.
7. **Is it safe to remove the reference temporarily?** **YES.** Removing line 33 and the audio references in `streamlit_app.py` restores stability without affecting any working backend component.
8. **Is audio actually required by the intended project scope?** Not for Phase 1. According to `backend/README.md` and `efafcf0:streamlit_app.py`, Phase 1 scope is strictly document, image, and video ingestion. Audio transcription was earmarked for Phase 2/3.

---

## PART 8 — DATABASE FORENSICS

### 8.1 SQLite Schema & Table Inspection
- **Database Path:** `/Users/anshika/Desktop/uptoskills-ai-assistant/backend/storage/metadata.db`
- **File Size:** 32,768 bytes (32 KB).
- **SQLite Pragmas:** WAL mode (`PRAGMA journal_mode = WAL`), synchronous normal.
- **Existing Tables:** Exactly ONE table exists: `document_metadata`.
- **Existing Indexes:**
  - `idx_doc_hash` ON `document_metadata(sha256_hash)`
  - `idx_doc_cat` ON `document_metadata(category)`
  - `idx_doc_timestamp` ON `document_metadata(upload_timestamp DESC)`
  - `sqlite_autoindex_document_metadata_1` ON `document_metadata(document_id)` (Primary Key)
- **Table Columns in `document_metadata`:**
  `document_id` (TEXT PK), `original_filename` (TEXT), `safe_filename` (TEXT), `category` (TEXT), `file_type` (TEXT), `mime_type` (TEXT), `file_size_bytes` (INTEGER), `sha256_hash` (TEXT), `storage_path` (TEXT), `upload_timestamp` (TEXT), `status` (TEXT), `metadata_json` (TEXT).

### 8.2 Database Expectations Comparison

```
Database Element        Current Streamlit       Current Backend         efafcf0 Streamlit       Actual DB State
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
document_metadata table Used (L299, 485)        Used (metadata.py L35)  Used (L129, 360)        EXISTS (18 rows)
document_extraction tbl Expected (L658-659)     NOT REFERENCED          NOT REFERENCED          DOES NOT EXIST
upload_success column   Expected in memory      NOT IN SCHEMA           NOT REFERENCED          DOES NOT EXIST
extraction_status col   Expected in memory      NOT IN SCHEMA           NOT REFERENCED          DOES NOT EXIST
metadata_json column    Ignored                 Used (empty JSON '{}')  Ignored                 EXISTS ('{}')
```

- **Verdict on `document_extraction`:** The table `document_extraction` never existed in the database and was never defined in `backend/ingestion/metadata.py`.

---

## PART 9 — END-TO-END INGESTION PIPELINE TRACE

The pipeline was traced step-by-step through execution in `backend/ingestion/pipeline.py`:

```
Step 1: UPLOAD
  ├─ Function: IngestionPipeline.process_file(filename: str, content: bytes)
  ├─ File: backend/ingestion/pipeline.py (L42)
  └─ Caller: streamlit_app.py (L358) or pytest

Step 2: VALIDATION
  ├─ Function: FileValidator.validate(filename, content)
  ├─ File: backend/ingestion/validator.py (L170)
  ├─ Checks: Allowed extension, empty file check (0 bytes), category size limits, magic bytes check.
  ├─ Output: ValidationResult (is_valid, category, file_type, sha256_hash, safe_filename)
  └─ Error Handling: If invalid, returns IngestionResult(success=False, error_message=...)

Step 3: DUPLICATE CHECK
  ├─ Function: MetadataManager.get_by_hash(validation_res.sha256_hash)
  ├─ File: backend/ingestion/metadata.py (L114)
  ├─ Action: Indexed query on SQLite.
  └─ Behavior: If match found, marks dup_validation, points abs_path to existing file, bypasses disk write.

Step 4: STORAGE (New Files Only)
  ├─ Function: StorageManager.save_file(doc_id, safe_name, category, content)
  ├─ File: backend/ingestion/storage.py (L46)
  ├─ Persistence: Writes binary to backend/storage/uploads/<category>/<doc_id>_<safe_filename>.
  └─ Collision Check: Appends millisecond timestamp if filename already exists.

Step 5: METADATA RECORDING (New Files Only)
  ├─ Function: MetadataManager.save(DocumentMetadata)
  ├─ File: backend/ingestion/metadata.py (L70)
  └─ Persistence: Inserts record into SQLite document_metadata table.

Step 6: TEXT EXTRACTION
  ├─ Function: extract_text(abs_path)
  ├─ File: backend/ingestion/text_extractor.py (L173)
  ├─ Call Site: pipeline.py (L144)
  ├─ Supported: .pdf (PyMuPDF), .docx (python-docx), .txt (UTF-8), .png/.jpg (Tesseract), .mp4 (OpenCV)
  └─ Error Handling: Wrapped in try/except. Sets extracted_content or extraction_error.

Step 7: PREPROCESSING & CLEANING
  ├─ Functions: remove_repeated_headers_footers(), preprocess_text()
  ├─ File: backend/ingestion/preprocessing.py (L33, L43)
  └─ Action: Strips repeated header/footer lines across pages; normalizes Unicode NFKC; collapses whitespace.

Step 8: CHUNKING
  ├─ Function: recursive_chunking(text, chunk_size=500, overlap=100)
  ├─ File: backend/ingestion/preprocessing.py (L140)
  └─ Action: Sentence-aware unit building and word-safe overlap accumulation.

Step 9: METADATA ATTACHMENT & TOKEN CHECK
  ├─ Functions: attach_chunk_metadata(), verify_chunk_tokens()
  ├─ File: backend/ingestion/preprocessing.py (L251, L293)
  └─ Action: Injects document_id, filename, file_type, page_number, chunk_index, and estimated_tokens.

Step 10: RETURN INGESTION RESULT
  ├─ File: backend/ingestion/pipeline.py (L245-L260)
  └─ Output: IngestionResult(success, document_id, metadata, extracted_content, preprocessed_content)
```

**Where the Pipeline Currently Terminates:**
The pipeline stops at Step 10. `preprocessed_content` is returned as a transient in-memory Python list. It is **never written to SQLite**, **never written to disk**, **never embedded**, and **never indexed into a vector store**.

---

## PART 10 — DOCUMENT EXTRACTION FORENSICS

Each format was tested directly using `./backend/.venv/bin/python3`:

| Format | Extension Checked | Validated by Magic Bytes? | Extraction Function | Persistence Status | Chunking Status | Test Result / Execution Behavior |
|---|---|---|---|---|---|---|
| **PDF** | `.pdf` | Yes (`%PDF-`) | `extract_pdf` (PyMuPDF) | In-Memory Only | Yes (Per page) | **Working on digital text** (5 chunks produced from `Assignment No. 5.pdf`). **Crashes on scanned pages** (`TesseractNotFoundError`). |
| **DOCX** | `.docx` | Yes (`PK\x03\x04` ZIP) | `extract_docx` (python-docx) | In-Memory Only | Yes | **Working for paragraphs** (9 chunks produced from `bme 8.docx`). Table extraction flattens rows into 1D list. |
| **TXT** | `.txt` | Yes (UTF-8 & null check) | `extract_txt` | In-Memory Only | Yes | **Working cleanly** (1 chunk produced from `sample.txt`). |
| **Images** | `.png`, `.jpg`, `.webp` | Yes (Pillow verification) | `extract_image` (Tesseract) | None | Blocked | **BROKEN:** Fails with `pytesseract.pytesseract.TesseractNotFoundError` because `tesseract` binary is missing from macOS PATH. |
| **Videos** | `.mp4`, `.mov`, `.avi` | Yes (ftyp, moov, RIFF) | `extract_video` (OpenCV + OCR) | None | Blocked | **BROKEN:** Fails with `TesseractNotFoundError`. OpenCV loop also lacks frame bounding/timeout. |
| **Audio** | `.mp3`, `.wav`, etc. | No (Not in config) | None (`audio_transcriber` missing) | None | None | **NOT IMPLEMENTED.** |
| **Spreadsheets** | `.csv`, `.xlsx` | No | None | None | None | **NOT IMPLEMENTED.** |
| **Presentations**| `.ppt`, `.pptx` | No | None | None | None | **NOT IMPLEMENTED.** |

---

## PART 11 — CHUNKING FORENSICS & EDGE CASES

The chunking algorithm in `backend/ingestion/preprocessing.py#L140-L248` was analyzed:

1. **Algorithm Type:** It is an iterative sentence accumulator with word-based overlap. It is not recursive.
2. **Historical Versions:** Git history confirms only ONE chunking implementation has ever existed in this repository (introduced in `efafcf0` / `3d7141e`). There is no older or alternative chunking script.
3. **Verified Overlap Discard Bug (`preprocessing.py#L242-L243`):**
   ```python
   if len(current) > chunk_size:
       current = unit
   ```
   When the overlap `prefix` plus the next sentence `unit` exceeds `chunk_size` (e.g., prefix is 90 chars, unit is 420 chars, chunk_size is 500 chars), the entire overlap prefix is dropped, producing a chunk with 0 overlap.
4. **Per-Page Severance (`pipeline.py#L159-L168`):**
   In PDF processing, `recursive_chunking` is called per page. Cross-page sentences are severed, and there is 0 overlap across page boundaries.

---

## PART 12 — TEST COVERAGE MATRIX

All 38 tests collected by pytest were executed and audited:

| Test File | Test Name | Tested Feature | Execution Result | What It Actually Proves | What It Fails to Test (Blind Spot) |
|---|---|---|---|---|---|
| `test_storage.py` | `test_storage_init_creates_directories` | Storage directories | PASSED | Creates `documents/`, `images/`, `videos/`. | Does not test audio directory. |
| `test_storage.py` | `test_generate_document_id` | Document ID UUID | PASSED | ID begins with `doc_` and is unique. | — |
| `test_storage.py` | `test_save_file_categorization` | File saving | PASSED | Saves file to correct subfolder. | Does not test path containment. |
| `test_storage.py` | `test_save_file_overwrite_prevention` | Collision safety | PASSED | Suffixes millisecond timestamp on collision. | — |
| `test_storage.py` | `test_file_exists_read_and_delete` | Storage I/O | PASSED | Reads and deletes files cleanly. | Does not test arbitrary file traversal. |
| `test_validator.py` | (18 separate tests) | File validation | ALL 18 PASSED | Validates 12 formats, headers, and sizes. | Does not test audio formats. |
| `test_metadata.py` | `test_metadata_save_and_get_by_id` | SQLite CRUD | PASSED | Saves and retrieves document records. | Does not test chunk persistence. |
| `test_metadata.py` | `test_metadata_get_by_hash` | Deduplication | PASSED | Instant SHA-256 indexed lookup works. | — |
| `test_metadata.py` | `test_metadata_list_and_category_filter`| Querying | PASSED | Filters by category accurately. | Does not test audio category. |
| `test_metadata.py` | `test_metadata_search` | Keyword search | PASSED | Filename and ID search works. | — |
| `test_metadata.py` | `test_metadata_aggregations` | Storage metrics | PASSED | Accurately aggregates total bytes. | — |
| `test_metadata.py` | `test_metadata_delete` | Record deletion | PASSED | Deletes record from SQLite. | Does not test cascading disk delete. |
| `test_pipeline.py` | `test_pipeline_ingest_document` | Pipeline Document | PASSED | Validates, stores, and saves metadata. | **Fails to assert `extracted_content` or `preprocessed_content`!** |
| `test_pipeline.py` | `test_pipeline_ingest_image` | Pipeline Image | PASSED | Validates and stores image file. | Fails to assert OCR extraction. |
| `test_pipeline.py` | `test_pipeline_ingest_video` | Pipeline Video | PASSED | Validates and stores video file. | Fails to assert OCR extraction. |
| `test_pipeline.py` | `test_pipeline_duplicate_detection` | Pipeline Deduplication | PASSED | Rejects duplicate content. | — |
| `test_pipeline.py` | `test_pipeline_reject_corrupted_file` | Corruption check | PASSED | Rejects fake PDF. | — |
| `test_pipeline.py` | `test_pipeline_reject_empty_file` | Empty file check | PASSED | Rejects 0-byte file. | — |
| `test_pipeline.py` | `test_pipeline_reject_unsupported_ext` | Extension check | PASSED | Rejects `.py` file. | — |
| `test_pipeline.py` | `test_pipeline_batch_processing` | Batch upload | PASSED | Sequences batch files cleanly. | — |
| `test_pipeline.py` | `test_pipeline_prevents_file_overwrite`| Overwrite safety | PASSED | Two files with identical names coexist. | — |

**Critical Finding:** Zero automated tests exist for `text_extractor.py`, `preprocessing.py`, or `streamlit_app.py`.

---

## PART 13 — ENVIRONMENT VS. CODE BUG DISAMBIGUATION

A critical distinction must be drawn between environmental defects and source code bugs:

### 13.1 The Environment Bug (Exit Code 126)
- **Symptom:** Executing `./backend/.venv/bin/pytest` or `./backend/.venv/bin/pip` returns:
  `./backend/.venv/bin/pytest: line 2: /Users/anshika/Desktop/uptoskills ai assistant /uptoskills-ai-assistant/backend/.venv/bin/python3.11: No such file or directory` (Exit code 126).
- **Cause:** The project was previously located in a path containing spaces: `/Users/anshika/Desktop/uptoskills ai assistant /uptoskills-ai-assistant`. When the directory was renamed or moved to `/Users/anshika/Desktop/uptoskills-ai-assistant`, the absolute shebang paths generated by `virtualenv` inside `backend/.venv/bin/*` became invalid.
- **Evidence:** `backend/.venv/pyvenv.cfg` contains:
  `command = /usr/local/bin/python3 -m venv /Users/anshika/Desktop/uptoskills ai assistant /uptoskills-ai-assistant/backend/.venv`.
- **Classification:** **ENVIRONMENT BUG ONLY.** The source code is not broken; the virtual environment's bin wrapper scripts are simply pointing to an old directory path.
- **Immediate Workaround:** Invoking via `./backend/.venv/bin/python3 -m pytest` or `./backend/.venv/bin/python3 -m streamlit` bypasses the broken wrapper scripts completely.

### 13.2 The Missing System Binary Bug
- **Symptom:** Image, video, and scanned PDF extraction fails with `pytesseract.pytesseract.TesseractNotFoundError`.
- **Cause:** Tesseract OCR is an external C++ binary (`/opt/homebrew/bin/tesseract` or `/usr/local/bin/tesseract`) that must be installed on the operating system. `pytesseract` is merely a Python wrapper that spawns `subprocess.Popen(["tesseract", ...])`.
- **Classification:** **HOST ENVIRONMENT REQUIREMENT.**

### 13.3 The Code Bugs
- `backend/streamlit_app.py` importing non-existent `audio_transcriber` and `EXTRACTABLE_EXTENSIONS`.
- `backend/streamlit_app.py` querying non-existent table `document_extraction` and attributes `r.upload_success`.
- `backend/ingestion/storage.py` lacking path containment in `read_file()`.
- **Classification:** **ACTUAL CODE BUGS.**

---

## PART 14 — REQUIREMENTS & DEPENDENCY RECONCILIATION

### 14.1 Complete Package Reconciliation Table

| Top-Level Python Package | Required By (Source File) | Declared in `requirements.txt`? | Installed in `.venv`? | Version in `.venv` | Reconciliation Action Required |
|---|---|---|---|---|---|
| `streamlit` | `backend/streamlit_app.py` | YES (`>=1.30.0`) | YES | 1.67.0 | Keep in `requirements.txt`. |
| `pydantic` | `backend/ingestion/models.py` | YES (`>=2.0.0`) | YES | 2.14.3 | Keep in `requirements.txt`. |
| `pillow` (`PIL`) | `validator.py`, `text_extractor.py` | YES (`>=10.0.0`) | YES | 13.0.0 | Keep in `requirements.txt`. |
| `pytest` | Test suite (`backend/tests/`) | YES (`>=7.0.0`) | YES | 9.2.0 | Keep in `requirements.txt`. |
| `python-dotenv` | Nowhere (Not imported) | YES (`>=1.0.0`) | YES | 1.2.4 | Unused; can be retained for config. |
| `pandas` | `backend/streamlit_app.py` (L23) | **NO (MISSING)** | YES | 3.1.1 | **ADD to `requirements.txt` (`pandas>=2.0.0`).** |
| `pymupdf` (`fitz`) | `backend/ingestion/text_extractor.py` (L5) | **NO (MISSING)** | YES | 2.1 | **ADD to `requirements.txt` (`pymupdf>=1.23.0`).** |
| `python-docx` (`docx`) | `backend/ingestion/text_extractor.py` (L9) | **NO (MISSING)** | YES | 1.2.0 | **ADD to `requirements.txt` (`python-docx>=1.0.0`).** |
| `pytesseract` | `backend/ingestion/text_extractor.py` (L7) | **NO (MISSING)** | YES | 0.3.13 | **ADD to `requirements.txt` (`pytesseract>=0.3.10`).** |
| `opencv-python-headless` (`cv2`) | `backend/ingestion/text_extractor.py` (L5) | **NO (MISSING)** | YES | 5.1.0.95 | **ADD to `requirements.txt` (`opencv-python-headless>=4.8.0`).** |
| `onnxruntime` | Nowhere | NO | YES | 1.32.0 | Dead package; ignore. |
| `uvicorn` | Nowhere | NO | YES | 0.54.0 | Dead package; ignore. |
| `starlette` | Nowhere | NO | YES | 1.10.0 | Dead package; ignore. |

---

## PART 15 — SECURITY & DATA RISK INVENTORY

| Vulnerability / Risk | Severity | Source File & Location | Technical Description & Evidence | Remediation |
|---|---|---|---|---|
| **Arbitrary File Read (Path Traversal)** | **HIGH** | `backend/ingestion/storage.py#L88-L94` | `read_file(path)` checks `if not p.is_absolute(): p = self.base_dir / p`. If `path` is `/etc/passwd`, it opens it directly. If `path` is `../../sensitive.txt`, it resolves outside `base_dir`. | Enforce `resolved_path.is_relative_to(self.base_dir)`. |
| **Denial of Service (DoS) via Video OCR** | **HIGH** | `backend/ingestion/text_extractor.py#L125-L155` | `extract_video()` processes frames in an unconstrained `while True` loop without a maximum frame threshold. A 10-minute video triggers 600 consecutive OCR calls, blocking the process. | Cap maximum processed frames (e.g. `max_frames=60`) and add processing timeout. |
| **Crash via Missing Tesseract Binary** | **MEDIUM** | `backend/ingestion/text_extractor.py#L22-L31` | Calling `pytesseract.image_to_string` on scanned PDFs or images raises unhandled `TesseractNotFoundError`. | Wrap in `try/except TesseractNotFoundError` and record graceful warning. |
| **Ghost File / Orphan Records on Deletion**| **LOW** | `backend/ingestion/metadata.py#L168` | Deleting via `metadata.delete(doc_id)` does not delete physical file on disk; deleting file on disk leaves database record. | Implement a unified `delete_document()` in `IngestionPipeline`. |

---

## PART 16 — SAFEST RECOVERY STRATEGY COMPARISON

| Dimension | Option A: Full Revert to `efafcf0` | Option B: Patch Current `streamlit_app.py` | Option C: Rewrite from Scratch | Option D: Hybrid Recovery (Recommended) |
|---|---|---|---|---|
| **Description** | Overwrite current `streamlit_app.py` with commit `efafcf0` version. | Manually edit current version to remove broken lines. | Discard `streamlit_app.py` and write a new UI script. | Retain stable element key hashing from current version; excise broken audio/DB references; restore chunk expanders from `efafcf0`. |
| **Advantages** | Guaranteed to start immediately; restores chunk displays. | Keeps element key hashing. | Clean slate. | **Best of both:** Prevents duplicate element ID errors AND restores chunk visibility without broken dependencies. |
| **Disadvantages** | Re-introduces `StreamlitDuplicateElementId` error on batch uploads. | Highly error-prone manual patching of 150+ lines. | Throws away 600 lines of working UI styling and layout. | Requires careful, targeted surgical edits. |
| **Risk Level** | Low | Medium | High | **Very Low** |
| **Files Affected** | `backend/streamlit_app.py` | `backend/streamlit_app.py` | `backend/streamlit_app.py` | `backend/streamlit_app.py`, `backend/requirements.txt` |
| **Verification** | Launch Streamlit | Launch Streamlit | Re-test everything | Launch Streamlit, run upload batch, inspect chunks |

### Recommendation
**OPTION D (Hybrid Recovery)** is objectively the safest and most robust path. It avoids reverting the duplicate element ID fix from commit `747baf3` while completely eliminating the uncommitted audio imports and restoring the missing chunk displays from commit `efafcf0`.

---

## PART 17 — EXACT STEP-BY-STEP RECOVERY CHECKLIST

```
[ ] STEP 1: Reconcile Dependencies
    ├─ File: backend/requirements.txt
    ├─ Action: Append pandas>=2.0.0, pymupdf>=1.23.0, python-docx>=1.0.0, pytesseract>=0.3.10, opencv-python-headless>=4.8.0.
    ├─ Reason: Ensures any developer or CI runner can install required libraries.
    └─ Verification: Run `./backend/.venv/bin/python3 -m pip check` to verify compatibility.

[ ] STEP 2: Excised Broken Imports in `backend/streamlit_app.py`
    ├─ File: backend/streamlit_app.py
    ├─ Action: Delete line 33 (`from backend.ingestion.audio_transcriber import AUDIO_EXTENSIONS`).
    ├─ Action: Delete line 38 (`from backend.ingestion.text_extractor import EXTRACTABLE_EXTENSIONS`).
    ├─ Action: Remove `'audio'` references from lines 290, 466, 475, and 637.
    ├─ Reason: Eliminates immediate startup ModuleNotFoundError and KeyError.
    └─ Verification: Run `./backend/.venv/bin/python3 -c "import backend.streamlit_app"` (must exit 0).

[ ] STEP 3: Align Result Model Attributes in `backend/streamlit_app.py`
    ├─ File: backend/streamlit_app.py (Tab 1, Lines 365-420)
    ├─ Action: Replace `r.upload_success` with `r.success`.
    ├─ Action: Replace `r.extraction_status == "success"` with `bool(r.extracted_content)`.
    ├─ Action: Remove `r.extraction_status` badge lookup.
    ├─ Reason: Aligns UI with actual `IngestionResult` fields in `backend/ingestion/models.py`.
    └─ Verification: Ingest sample file via UI; verify no AttributeError is thrown.

[ ] STEP 4: Restore Preprocessed Chunk Display in `backend/streamlit_app.py`
    ├─ File: backend/streamlit_app.py (Tab 1)
    ├─ Action: Port the chunk expander block from `efafcf0:backend/streamlit_app.py` (lines 324-370).
    ├─ Reason: Makes the working preprocessing and chunking pipeline visible to users and testers.
    └─ Verification: Upload a text document; verify chunk cards and token counters render in expanders.

[ ] STEP 5: Fix Dashboard Registry Queries in `backend/streamlit_app.py`
    ├─ File: backend/streamlit_app.py (Tab 2)
    ├─ Action: Remove calls to `metadata_mgr.get_extraction_stats()` (lines 494, 544).
    ├─ Action: Remove call to `pipeline.get_or_reextract()` (line 582).
    ├─ Action: Display persistent metadata directly from `DocumentMetadata`.
    ├─ Reason: Removes references to non-existent methods and non-existent `document_extraction` table.
    └─ Verification: Navigate to "Verification Dashboard" tab; verify registry table loads 18 rows.

[ ] STEP 6: Secure Storage Manager Against Path Traversal
    ├─ File: backend/ingestion/storage.py
    ├─ Action: In `read_file()` and `delete_file()`, resolve path and enforce `target.is_relative_to(self.base_dir)`.
    ├─ Reason: Mitigates arbitrary file reading vulnerability.
    └─ Verification: Add test asserting `read_file("/etc/passwd")` raises `ValueError`.

[ ] STEP 7: Add Graceful Exception Handling to Tesseract OCR
    ├─ File: backend/ingestion/text_extractor.py
    ├─ Action: Wrap `pytesseract.image_to_string` calls in `try/except Exception`.
    ├─ Action: Return empty string with warning if Tesseract binary is not installed on system.
    ├─ Reason: Prevents scanned PDFs or image uploads from crashing the entire ingestion pipeline.
    └─ Verification: Pass a blank PDF page to `extract_pdf()`; verify it returns text="" without crashing.

[ ] STEP 8: Clean Up Duplicate Function in `backend/ingestion/preprocessing.py`
    ├─ File: backend/ingestion/preprocessing.py
    ├─ Action: Delete the duplicate definitions of `verify_chunk_tokens` at lines 116 and 273, keeping line 293.
    ├─ Reason: Removes redundant dead code.
    └─ Verification: Run `./backend/.venv/bin/python3 -m pytest backend/tests` (must pass 38/38).
```

---

## PART 18 — "DO NOT TOUCH" STABILITY LIST

The following components are strictly verified as working and must **NOT** be modified during recovery:

1. **`backend/ingestion/validator.py`:** Fully functional and verified by 18 unit tests. Magic-bytes checking, size enforcement, and filename sanitization must remain untouched.
2. **`backend/ingestion/metadata.py`:** Core SQLite metadata CRUD, SHA-256 deduplication, and aggregation logic are fully working and verified by 6 unit tests. Do not alter table creation or indexing.
3. **`backend/ingestion/storage.py` (Core Logic):** Directory structures, UUID generation, and file collision timestamping are verified and stable. (Only apply the security containment check in `read_file`).
4. **Existing Test Suite (`backend/tests/`):** All 38 existing tests pass cleanly. Do not modify existing test assertions.
5. **Runtime Storage & Database (`backend/storage/`):** The 18 cataloged documents and `metadata.db` represent real verified ingested state. Do not delete or reset this directory.
6. **Git History:** Do not perform force-pushes or rebases that rewrite historical commits.

---

## PART 19 — FILE-BY-FILE REMEDIATION MATRIX

| File Path | Current Status | Problem | Proposed Action | Technical Reason | Verification Method |
|---|---|---|---|---|---|
| `backend/requirements.txt` | Incomplete | Missing 5 required packages. | **EDIT** | Declare pandas, pymupdf, python-docx, pytesseract, opencv. | `pip check` |
| `backend/streamlit_app.py` | BROKEN | Crashes on launch; phantom audio imports; missing chunks. | **EDIT** (Option D) | Excise audio imports; align result models; restore chunk expanders. | Launch Streamlit app |
| `backend/ingestion/models.py` | Working | Missing fields if Phase 2 is adopted. | **KEEP** | Already contains `preprocessed_content`. Compatible with Phase 1. | `pytest` |
| `backend/ingestion/pipeline.py` | Working | Chunks not persisted to disk. | **KEEP** (Phase 1) | Ingestion and chunking execute cleanly. Defer persistence to Phase 2. | Python pipeline trace |
| `backend/ingestion/preprocessing.py` | Working | `verify_chunk_tokens` defined 3 times. | **EDIT** | Remove 2 duplicate function definitions. | AST parse & pytest |
| `backend/ingestion/text_extractor.py` | Partially Working | OCR crashes if tesseract missing; DOCX table flattened. | **EDIT** | Add try/except to OCR; structure DOCX tables as 2D/3D grids. | Run sample PDF/DOCX |
| `backend/ingestion/storage.py` | Working | Path traversal in `read_file()`. | **EDIT** | Add `is_relative_to(self.base_dir)` check. | Unit test traversal |
| `backend/ingestion/validator.py` | Complete | None. | **KEEP** | 18 passing tests; stable. | `test_validator.py` |
| `backend/ingestion/metadata.py` | Complete | None. | **KEEP** | 6 passing tests; stable SQLite store. | `test_metadata.py` |
| `backend/config.py` | Working | No audio category. | **KEEP** | Standard for Phase 1. Add `.env.example`. | Python import |
| `.env.example` | Missing | No template for environment variables. | **CREATE** | Document `RAG_STORAGE_DIR`, `MAX_DOC_SIZE_MB`, etc. | File inspection |

---

## PART 20 — FINAL RECOVERY VERDICT

1. **What is definitely working?**
   File validation (magic bytes, size, sanitation), SHA-256 duplicate detection, categorized file persistence, SQLite metadata tracking, digital text extraction (PDF, DOCX, TXT), Unicode cleaning, and sentence-aware chunking.
2. **What is definitely broken?**
   `backend/streamlit_app.py` (crashes on launch), Tesseract OCR on images/scanned PDFs/videos, DOCX table formatting in UI, and `.venv/bin` wrapper scripts (due to stale path with spaces).
3. **What was broken by later changes?**
   `backend/streamlit_app.py` was broken by commit `747baf3` (merged via `aac8b13`), which introduced uncommitted audio imports, model attribute mismatches, and non-existent database queries.
4. **What can be recovered from Git?**
   The working Streamlit preprocessed chunk display and clean 5-column dashboard can be recovered directly from commit `efafcf0` (`git show efafcf0:backend/streamlit_app.py`).
5. **What must be newly implemented?**
   A graceful fallback for OCR when Tesseract is missing, persistent database storage for chunks (`document_chunks` table), and missing package declarations in `backend/requirements.txt`.
6. **What should be preserved?**
   The core ingestion engine (`validator.py`, `storage.py`, `metadata.py`, `pipeline.py`), the existing 38 unit tests, and the 18 cataloged files in `backend/storage/`.
7. **What should be deferred?**
   Audio transcription (faster-whisper), vector database integration (ChromaDB), embedding models, LLM answer generation, and frontend React development.
8. **What is the safest first code change?**
   Update `backend/requirements.txt` to include `pandas`, `pymupdf`, `python-docx`, `pytesseract`, and `opencv-python-headless`.
9. **What should be tested immediately after that change?**
   Execute `./backend/.venv/bin/python3 -c "import pandas, pymupdf, docx, pytesseract, cv2; print('Dependencies OK')"` to verify environment readiness.
10. **What is the complete path from current state to a working application?**
    Reconcile dependencies -> Patch `streamlit_app.py` using Hybrid Option D -> Verify Streamlit UI launches -> Add chunk persistence table to SQLite -> Create unit tests for chunking and extraction.

---

```
REPOSITORY MODIFIED: NO
SOURCE CODE MODIFIED: NO
DEPENDENCIES INSTALLED: NO
GIT HISTORY MODIFIED: NO
```
