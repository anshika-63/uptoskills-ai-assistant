"""
Streamlit Application for UpToSkills AI Assistant - RAG Document Ingestion (Phase 2).
Provides an interactive interface for file upload, validation, duplicate detection,
storage inspection, persistent extraction display, and a verification dashboard.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure backend package is discoverable regardless of current working directory
current_file = Path(__file__).resolve()
backend_dir = current_file.parent
project_root = backend_dir.parent

for p in (str(project_root), str(backend_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import streamlit as st

from backend.config import (
    ALLOWED_EXTENSIONS,
    CATEGORY_DIRECTORIES,
    DEFAULT_MAX_SIZES_MB,
    METADATA_DB_PATH,
    STORAGE_BASE_DIR,
)
from backend.ingestion.audio_transcriber import AUDIO_EXTENSIONS
from backend.ingestion.metadata import MetadataManager
from backend.ingestion.models import ValidationStatus
from backend.ingestion.pipeline import IngestionPipeline
from backend.ingestion.storage import StorageManager
from backend.ingestion.text_extractor import EXTRACTABLE_EXTENSIONS
from backend.ingestion.validator import FileValidator

# Page Configuration
st.set_page_config(
    page_title="UpToSkills RAG Ingestion Pipeline",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-badge-valid {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-duplicate {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-invalid {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .metric-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline() -> IngestionPipeline:
    """Initializes and caches the ingestion pipeline components."""
    validator = FileValidator()
    storage = StorageManager(base_dir=STORAGE_BASE_DIR)
    metadata = MetadataManager(db_path=METADATA_DB_PATH)
    return IngestionPipeline(
        validator=validator,
        storage_manager=storage,
        metadata_manager=metadata,
    )


def format_bytes(num_bytes: int) -> str:
    """Formats bytes into human-readable strings (KB, MB, GB)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def render_extraction(
    extracted: dict,
    document_id: Optional[str] = None,
    key_prefix: Optional[str] = None,
) -> None:
    """
    Renders an extraction payload dict into Streamlit widgets.
    Handles PDF, DOCX, TXT, and Audio uniformly with stable, unique widget keys
    to prevent StreamlitDuplicateElementId errors when multiple files are rendered simultaneously.

    Parameters:
        extracted: Dictionary containing extraction metadata, raw text, pages, tables,
                   or audio transcript segments.
        document_id: Optional unique document ID (e.g., 'doc_abc123').
        key_prefix: Optional prefix for widget keys (e.g., 'upload_0' or 'inspect').
    """
    file_type = extracted.get("file_type", "")

    # Establish a stable, unique identifier for widget keys
    stable_id = (
        document_id
        or extracted.get("document_id")
        or extracted.get("doc_id")
        or ""
    )
    if not stable_id:
        content_repr = json.dumps(extracted, sort_keys=True, default=str)
        stable_id = hashlib.sha256(content_repr.encode("utf-8")).hexdigest()[:12]

    base_key = f"{key_prefix}_{stable_id}" if key_prefix else f"ext_{stable_id}"

    # Extraction statistics header
    char_count = extracted.get("char_count", 0)
    word_count = extracted.get("word_count", 0)
    st.caption(
        f"**Extraction stats:** {char_count:,} characters · {word_count:,} words"
        + (f" · {extracted['total_pages']} pages" if file_type == "pdf" and "total_pages" in extracted else "")
    )

    if file_type == "pdf":
        pages = extracted.get("pages", [])
        if not any(p.get("text") for p in pages):
            st.info(
                "No digital text found in this PDF (may be scanned). "
                "OCR support is planned for a future phase."
            )
        else:
            for page in pages:
                label = f"Page {page.get('page_number', '?')}"
                if page.get("char_count", len(page.get("text", ""))):
                    label += f"  ({page.get('char_count', len(page.get('text', '')))} chars)"
                with st.expander(label, expanded=False):
                    st.text(page.get("text") or "No text found on this page.")

    elif file_type == "docx":
        paragraphs = extracted.get("paragraphs", [])
        tables = extracted.get("tables", [])

        if paragraphs:
            st.markdown("**Paragraphs**")
            for para in paragraphs:
                st.write(para)
        else:
            st.info("No paragraphs found in this document.")

        if tables:
            st.markdown("**Tables**")
            for t_idx, table in enumerate(tables, start=1):
                # table is List[List[str]] — rows × cols
                if table:
                    df = pd.DataFrame(table)
                    st.caption(f"Table {t_idx} — {len(table)} rows × {len(table[0])} columns")
                    st.dataframe(df, use_container_width=True, hide_index=True)

    elif file_type == "txt":
        raw = extracted.get("raw_text") or extracted.get("text", "")
        st.text_area(
            "Full text content:",
            raw,
            height=300,
            key=f"{base_key}_txt_content",
        )

    elif extracted.get("category") == "audio" or file_type in ("mp3", "wav", "m4a", "aac", "ogg", "flac"):
        duration = extracted.get("duration", 0.0)
        language = extracted.get("language", "unknown")
        lang_prob = extracted.get("language_probability", 0.0)
        segments = extracted.get("segments", [])
        seg_count = extracted.get("segment_count", len(segments))

        st.caption(
            f"**Audio Transcript:** {char_count:,} chars · {word_count:,} words · {duration:.2f}s duration · "
            f"Language: `{language}` ({lang_prob*100:.1f}%) · {seg_count} segments"
        )

        raw = extracted.get("raw_text", "")
        if raw:
            st.markdown("**Transcript Full Text**")
            st.text_area(
                "Full Transcript:",
                raw,
                height=180,
                label_visibility="collapsed",
                key=f"{base_key}_audio_transcript",
            )
        else:
            st.info("No speech detected in audio file.")

        if segments:
            st.markdown(f"**Timestamped Segments ({len(segments)})**")
            seg_data = []
            for s in segments:
                seg_data.append({
                    "Timestamp": s.get("timestamp", f"[{s.get('start', 0):.2f} -> {s.get('end', 0):.2f}]"),
                    "Text": s.get("text", ""),
                })
            st.dataframe(pd.DataFrame(seg_data), use_container_width=True, hide_index=True)

        # Download JSON button with stable key
        st.download_button(
            label="📥 Download Transcript JSON",
            data=json.dumps(extracted, indent=2, ensure_ascii=False),
            file_name=f"transcript_{stable_id}_{file_type or 'audio'}.json",
            mime="application/json",
            key=f"{base_key}_dl_transcript_json",
        )

    else:
        raw = extracted.get("raw_text", "")
        if raw:
            st.text_area(
                "Extracted text:",
                raw,
                height=300,
                key=f"{base_key}_raw_text",
            )
        else:
            st.info("No extractable text available for this file type.")


pipeline = get_pipeline()
storage_mgr = pipeline.storage
metadata_mgr = pipeline.metadata

# Sidebar
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/streamlit/brand/main/logo/streamlit-mark-color.png",
        width=48,
    )
    st.title("UpToSkills AI")
    st.caption("RAG Ingestion Pipeline • Phase 2")
    st.markdown("---")

    st.subheader("Allowed Extensions & Limits")
    st.markdown(
        f"""
        - **Documents** ({DEFAULT_MAX_SIZES_MB['documents']} MB max):  
          `{', '.join(sorted(ALLOWED_EXTENSIONS['documents']))}`
        - **Images** ({DEFAULT_MAX_SIZES_MB['images']} MB max):  
          `{', '.join(sorted(ALLOWED_EXTENSIONS['images']))}`
        - **Videos** ({DEFAULT_MAX_SIZES_MB['videos']} MB max):  
          `{', '.join(sorted(ALLOWED_EXTENSIONS['videos']))}`
        - **Audio** ({DEFAULT_MAX_SIZES_MB['audio']} MB max):  
          `{', '.join(sorted(ALLOWED_EXTENSIONS['audio']))}`
        """
    )
    st.markdown("---")

    st.subheader("Storage Configuration")
    st.code(f"Uploads: {STORAGE_BASE_DIR.name}\nDB: {METADATA_DB_PATH.name}", language="text")

    # Refresh metrics
    total_docs = metadata_mgr.count_total()
    total_bytes = metadata_mgr.get_total_storage_bytes()
    cat_counts = metadata_mgr.count_by_category()

    st.markdown("---")
    st.write(f"**Total Cataloged Files:** {total_docs}")
    st.write(f"**Total Storage Used:** {format_bytes(total_bytes)}")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()


# Main Header
st.markdown('<div class="main-header">UpToSkills RAG Ingestion Pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Phase 2: Document Upload, Validation, Deduplication, Persistent Text Extraction & Cataloging</div>',
    unsafe_allow_html=True,
)

tab_upload, tab_dashboard, tab_audit = st.tabs(
    ["📤 Document Upload & Ingestion", "📊 Verification Dashboard", "🛡️ Validation & Architecture Audit"]
)

# ══════════════════════════════════════════════════════════════════════
# TAB 1: Document Upload & Ingestion
# ══════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### Upload Documents, Images, Videos, or Audio")
    st.info(
        "Supported formats: **PDF, DOCX, TXT**, **PNG, JPG, JPEG, WEBP**, **MP4, MOV, AVI, MKV, WEBM**, "
        "and **MP3, WAV, M4A, AAC, OGG, FLAC**. "
        "Files undergo strict format, size, header integrity, and SHA-256 deduplication checks before being stored. "
        "Audio files are automatically transcribed with timestamps using Speech-to-Text."
    )

    all_allowed = []
    for ext_set in ALLOWED_EXTENSIONS.values():
        all_allowed.extend([ext.lstrip(".") for ext in ext_set])

    uploaded_files = st.file_uploader(
        "Select or drag and drop files to ingest",
        accept_multiple_files=True,
        type=all_allowed,
        help="Upload multiple files simultaneously. Files are processed individually and validated against corruption.",
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) ready for ingestion.**")

        if st.button("🚀 Process & Ingest Files", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []

            for i, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                content = uploaded_file.getvalue()
                status_text.text(f"Processing ({i+1}/{len(uploaded_files)}): {filename}...")

                result = pipeline.process_file(filename=filename, content=content)
                results.append(result)
                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("Batch processing complete!")

            # Summary KPIs
            new_count       = sum(1 for r in results if r.upload_success)
            duplicate_count = sum(1 for r in results if r.validation_result.is_duplicate)
            invalid_count   = sum(1 for r in results if not r.success and not r.validation_result.is_duplicate)
            extracted_count = sum(1 for r in results if r.extraction_status == "success")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Processed", len(results))
            with col2:
                st.metric("New Files Stored", new_count)
            with col3:
                st.metric("Text Extracted", extracted_count)
            with col4:
                st.metric("Duplicates", duplicate_count)
            with col5:
                st.metric("Validation Failures", invalid_count)

            st.markdown("---")
            st.markdown("### Ingestion Results")

            # Display individual file result cards
            for idx, res in enumerate(results):
                val = res.validation_result
                if res.upload_success:
                    status_badge = "✅ STORED"
                elif val.is_duplicate:
                    status_badge = "⚠️ DUPLICATE"
                else:
                    status_badge = "❌ INVALID"

                # Extraction badge
                ext_badge = {
                    "success":        "🟢 Extracted & Saved",
                    "retrieved":      "🔵 Retrieved from DB",
                    "failed":         "🔴 Extraction Failed",
                    "not_applicable": "—",
                }.get(res.extraction_status, "—")

                with st.expander(
                    f"{status_badge} · {ext_badge} — {val.safe_filename or 'File'}",
                    expanded=True,
                ):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Original Filename:** `{res.metadata.original_filename if res.metadata else val.safe_filename}`")
                        st.write(f"**Document ID:** `{res.document_id or 'N/A'}`")
                        st.write(f"**Category:** `{val.category or 'N/A'}`")
                        st.write(f"**MIME Type:** `{val.mime_type or 'N/A'}`")
                    with col_b:
                        st.write(f"**File Size:** `{format_bytes(val.file_size_bytes)}`")
                        st.write(f"**SHA-256:** `{val.sha256_hash[:16]}...`" if val.sha256_hash else "**SHA-256:** `N/A`")
                        st.write(f"**Upload Status:** {res.message}")
                        st.write(f"**Extraction Status:** `{res.extraction_status}`")

                    if val.sha256_hash:
                        st.caption(f"Full SHA-256 Digest: `{val.sha256_hash}`")

                    # Audio Player if file is audio
                    if val.category == "audio" and res.metadata:
                        try:
                            audio_data = storage_mgr.read_file(res.metadata.storage_path)
                            st.audio(audio_data, format=f"audio/{val.file_type.lstrip('.')}")
                        except Exception:
                            pass

                    # Extracted text / transcript display
                    st.markdown("---")
                    st.markdown("#### 📄 Extracted Content / Transcript")
                    if res.extraction_error:
                        st.error(f"Extraction / transcription failed: {res.extraction_error}")
                    elif res.extracted_content:
                        render_extraction(
                            res.extracted_content,
                            document_id=res.document_id,
                            key_prefix=f"upload_{idx}",
                        )
                    elif res.extraction_status == "not_applicable":
                        st.info("Text extraction is not applicable for this file type.")
                    else:
                        st.warning("No extracted text or transcript available.")

                    # Preprocessed Chunks Display
                    if res.preprocessed_content:
                        st.markdown("---")
                        st.markdown(f"#### 🧩 Preprocessed Chunks ({len(res.preprocessed_content)})")
                        with st.expander(f"View {len(res.preprocessed_content)} chunks with token estimates", expanded=False):
                            for c_idx, chunk in enumerate(res.preprocessed_content):
                                page_info = f" · Page {chunk['page_number']}" if chunk.get("page_number") else ""
                                token_info = f" · ~{chunk.get('estimated_tokens', '?')} tokens"
                                within_limit = "✅ Within limit" if chunk.get("within_token_limit", True) else "⚠️ Exceeds limit"
                                st.markdown(f"**Chunk {c_idx + 1}** ({within_limit}{page_info}{token_info})")
                                st.text(chunk.get("text", ""))


# ══════════════════════════════════════════════════════════════════════
# TAB 2: Verification Dashboard
# ══════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("### Ingested Documents Registry")

    # Real-time KPIs
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    with kpi_col1:
        st.metric("Total Ingested", metadata_mgr.count_total())
    with kpi_col2:
        st.metric("Total Storage", format_bytes(metadata_mgr.get_total_storage_bytes()))
    with kpi_col3:
        st.metric("Documents", cat_counts.get("documents", 0))
    with kpi_col4:
        st.metric("Images", cat_counts.get("images", 0))
    with kpi_col5:
        st.metric("Videos", cat_counts.get("videos", 0))
    with kpi_col6:
        st.metric("Audio", cat_counts.get("audio", 0))

    st.markdown("---")

    # Filters and search
    filt_col1, filt_col2 = st.columns([1, 3])
    with filt_col1:
        category_filter = st.selectbox(
            "Filter by Category",
            options=["All", "Documents", "Images", "Videos", "Audio"],
            index=0,
        )
    with filt_col2:
        search_query = st.text_input(
            "Search Documents",
            placeholder="Search by original filename or Document ID...",
        )

    # Fetch documents from SQLite
    docs = metadata_mgr.list_documents(
        category=category_filter,
        search=search_query,
        limit=200,
    )

    if docs:
        table_data = []
        for d in docs:
            stats = metadata_mgr.get_extraction_stats(d.document_id)
            table_data.append(
                {
                    "Document ID": d.document_id,
                    "Original Filename": d.original_filename,
                    "Category": d.category,
                    "Type": d.file_type,
                    "Size": format_bytes(d.file_size_bytes),
                    "SHA-256": d.sha256_hash[:12] + "...",
                    "Upload Date": d.upload_timestamp[:19].replace("T", " "),
                    "Upload Status": d.status,
                    "Extraction": stats["extraction_status"] if stats else "not extracted",
                    "Words": str(stats["word_count"]) if (stats and stats.get("word_count") is not None) else "—",
                }
            )

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🔍 Single Document Deep Inspector")

        doc_options = {f"{d.original_filename} ({d.document_id})": d for d in docs}
        selected_label = st.selectbox("Select document to inspect:", options=list(doc_options.keys()))

        if selected_label:
            selected_doc = doc_options[selected_label]
            disk_exists = storage_mgr.file_exists(selected_doc.storage_path)

            insp_col1, insp_col2 = st.columns([2, 1])
            with insp_col1:
                st.write(f"**Document ID:** `{selected_doc.document_id}`")
                st.write(f"**Original Filename:** `{selected_doc.original_filename}`")
                st.write(f"**Safe Stored Name:** `{selected_doc.safe_filename}`")
                st.write(f"**Category:** `{selected_doc.category}`")
                st.write(f"**MIME Type:** `{selected_doc.mime_type}`")
                st.write(f"**Size on Disk:** {format_bytes(selected_doc.file_size_bytes)}")
                st.write(f"**Storage Relative Path:** `{selected_doc.storage_path}`")
                st.write(f"**Upload Timestamp (UTC):** `{selected_doc.upload_timestamp}`")
                st.write(f"**SHA-256 Digest:**")
                st.code(selected_doc.sha256_hash, language="text")

            with insp_col2:
                st.markdown("**File Persistence Status:**")
                if disk_exists:
                    st.success("✅ File exists on filesystem")
                else:
                    st.error("❌ Physical file missing on disk")

                # Extraction stats panel
                ext_stats = metadata_mgr.get_extraction_stats(selected_doc.document_id)
                if ext_stats:
                    st.markdown("**Extraction Summary:**")
                    st.write(f"Status: `{ext_stats['extraction_status']}`")
                    st.write(f"Characters: `{ext_stats['char_count']:,}`")
                    st.write(f"Words: `{ext_stats['word_count']:,}`")
                    if ext_stats.get("page_count"):
                        st.write(f"Pages: `{ext_stats['page_count']}`")
                    st.write(f"Extracted at: `{ext_stats['extracted_at'][:19]}`")

                # Preview section based on category
                if disk_exists:
                    try:
                        file_bytes = storage_mgr.read_file(selected_doc.storage_path)
                        if selected_doc.category == "images":
                            st.image(file_bytes, caption=selected_doc.original_filename, use_container_width=True)
                        elif selected_doc.category == "videos":
                            st.video(file_bytes)
                        elif selected_doc.category == "audio":
                            st.audio(file_bytes, format=f"audio/{selected_doc.file_type.lstrip('.')}")
                        elif selected_doc.file_type == ".pdf":
                            st.caption(f"PDF Header: {file_bytes[:16].decode('latin-1', errors='ignore')}")
                    except Exception as preview_err:
                        st.caption(f"Preview unavailable: {preview_err}")

            # Extracted text section — full width below the two-column layout
            st.markdown("---")
            st.markdown("#### 📄 Extracted Text / Transcript")

            is_extractable = (
                selected_doc.file_type in EXTRACTABLE_EXTENSIONS
                or selected_doc.file_type in AUDIO_EXTENSIONS
                or selected_doc.category == "audio"
            )
            if not is_extractable:
                st.info("Text extraction is not applicable for this file type.")
            else:
                # Retrieve from DB; re-extract from disk if missing
                extracted, error = pipeline.get_or_reextract(selected_doc.document_id)
                if error:
                    st.error(f"Could not retrieve or re-extract content: {error}")
                elif extracted:
                    render_extraction(
                        extracted,
                        document_id=selected_doc.document_id,
                        key_prefix="inspect",
                    )
                else:
                    st.warning("No extraction data available.")

    else:
        st.info("No documents found matching the filter criteria. Use the Upload tab to ingest documents.")


# ══════════════════════════════════════════════════════════════════════
# TAB 3: System & Validation Audit
# ══════════════════════════════════════════════════════════════════════
with tab_audit:
    st.markdown("### Phase 2 System Architecture & Validation Rules")

    st.markdown(
        """
        #### 1. Ingestion Boundaries
        - **In Scope (Implemented):**
          - Multi-format file upload and buffering.
          - Category categorization (Documents, Images, Videos, Audio).
          - MIME and magic-bytes header verification.
          - Empty (0-byte) and corrupted file rejection.
          - Configurable file size limits per category.
          - Accidental overwrite prevention and safe filename sanitization.
          - Cryptographic duplicate detection via SHA-256 hashing.
          - Persistent SQLite metadata store with indexed lookup.
          - Text extraction from PDF, DOCX and TXT using PyMuPDF and python-docx.
          - **Audio Speech-to-Text transcription** (MP3, WAV, M4A, AAC, OGG, FLAC) via faster-whisper & FFmpeg.
          - **Timestamped and segmented transcripts** with duration and language detection.
          - **Persistent extraction & transcript storage** in JSON files and `document_extraction` SQLite table.
          - Extraction & transcript retrieval on duplicate upload & Document Inspector.
          - Separate upload status and extraction status tracking.
          - On-demand re-extraction/re-transcription from stored file when DB record is missing.
        - **Out of Scope (Phase 3+):**
          - OCR for scanned PDFs (Tesseract / Amazon Textract).
          - Live microphone voice commands (Query Processing phase).
          - Video audio extraction and transcription (Whisper).
          - Text preprocessing, cleaning, chunking.
          - Embedding generation & ChromaDB vector ingestion.
        """
    )

    st.markdown("#### 2. Configured Validation Rules")
    rule_data = [
        {"Category": "Documents", "Extensions": ", ".join(ALLOWED_EXTENSIONS["documents"]), "Max Size": f"{DEFAULT_MAX_SIZES_MB['documents']} MB", "Magic Check": "%PDF- header, Word ZIP archive structure, UTF-8 text integrity"},
        {"Category": "Images", "Extensions": ", ".join(ALLOWED_EXTENSIONS["images"]), "Max Size": f"{DEFAULT_MAX_SIZES_MB['images']} MB", "Magic Check": "PNG magic signature, JPEG SOI marker, RIFF/WEBP + Pillow verification"},
        {"Category": "Videos", "Extensions": ", ".join(ALLOWED_EXTENSIONS["videos"]), "Max Size": f"{DEFAULT_MAX_SIZES_MB['videos']} MB", "Magic Check": "MP4 ftyp atom, MOV atom, RIFF AVI chunk, MKV/WEBM EBML header"},
        {"Category": "Audio", "Extensions": ", ".join(ALLOWED_EXTENSIONS["audio"]), "Max Size": f"{DEFAULT_MAX_SIZES_MB['audio']} MB", "Magic Check": "ID3/MPEG sync, RIFF/WAVE header, OggS signature, fLaC marker, ftyp atom, ADTS sync word"},
    ]
    st.table(pd.DataFrame(rule_data))

    st.markdown("#### 3. Storage Diagnostics")
    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        st.write(f"**Uploads Directory:** `{STORAGE_BASE_DIR}`")
        st.write(f"**Directory Exists:** {STORAGE_BASE_DIR.exists()}")
        for cat in CATEGORY_DIRECTORIES.values():
            cat_dir = STORAGE_BASE_DIR / cat
            st.write(f"• `{cat}/`: {'Exists' if cat_dir.exists() else 'Missing'}")
    with diag_col2:
        st.write(f"**Metadata Database:** `{METADATA_DB_PATH}`")
        st.write(f"**Database File Exists:** {METADATA_DB_PATH.exists()}")
        st.write(f"**Database Writable:** {os.access(METADATA_DB_PATH.parent, os.W_OK)}")
        # Extraction table stats
        from backend.ingestion.metadata import MetadataManager as _MM
        import sqlite3 as _sq
        try:
            conn = _sq.connect(str(METADATA_DB_PATH))
            cnt = conn.execute("SELECT COUNT(*) FROM document_extraction").fetchone()[0]
            succ = conn.execute("SELECT COUNT(*) FROM document_extraction WHERE extraction_status='success'").fetchone()[0]
            conn.close()
            st.write(f"**Extraction Records:** {cnt} total · {succ} successful")
        except Exception:
            st.write("**Extraction Records:** table initializes on first upload")