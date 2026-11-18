"""
Streamlit Application for UpToSkills AI Assistant - RAG Document Ingestion (Phase 1).
Provides an interactive interface for file upload, validation, duplicate detection,
storage inspection, and a verification dashboard.
"""

import os
import sys
from pathlib import Path

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
from backend.ingestion.metadata import MetadataManager
from backend.ingestion.models import ValidationStatus
from backend.ingestion.pipeline import IngestionPipeline
from backend.ingestion.storage import StorageManager
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
    st.caption("RAG Ingestion Pipeline • Phase 1")
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
    '<div class="sub-header">Phase 1: Multi-Format Document Upload, Deep Validation, Deduplication & Persistent Cataloging</div>',
    unsafe_allow_html=True,
)

tab_upload, tab_dashboard, tab_audit = st.tabs(
    ["📤 Document Upload & Ingestion", "📊 Verification Dashboard", "🛡️ Validation & Architecture Audit"]
)

# TAB 1: Document Upload & Ingestion
with tab_upload:
    st.markdown("### Upload Documents, Images, or Videos")
    st.info(
        "Supported formats: **PDF, DOCX, TXT**, **PNG, JPG, JPEG, WEBP**, and **MP4, MOV, AVI, MKV, WEBM**. "
        "Files undergo strict format, size, header integrity, and SHA-256 deduplication checks before being stored."
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
            success_count = sum(1 for r in results if r.success)
            duplicate_count = sum(1 for r in results if r.validation_result.is_duplicate)
            invalid_count = sum(
                1 for r in results if not r.success and not r.validation_result.is_duplicate
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Processed", len(results))
            with col2:
                st.metric("Successfully Ingested", success_count)
            with col3:
                st.metric("Duplicates Rejected", duplicate_count)
            with col4:
                st.metric("Validation Failures", invalid_count)

            st.markdown("---")
            st.markdown("### Ingestion Results")

            # Display individual file result cards
            for res in results:
                val = res.validation_result
                if res.success:
                    status_badge = "✅ VALID & STORED"
                    box_type = st.success
                elif val.is_duplicate:
                    status_badge = "⚠️ DUPLICATE DETECTED"
                    box_type = st.warning
                else:
                    status_badge = "❌ VALIDATION FAILED"
                    box_type = st.error

                with st.expander(f"{status_badge} — {val.safe_filename or 'File'}", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Original Filename:** `{res.metadata.original_filename if res.metadata else val.safe_filename}`")
                        st.write(f"**Document ID:** `{res.document_id or 'N/A'}`")
                        st.write(f"**Category:** `{val.category or 'N/A'}`")
                        st.write(f"**MIME Type:** `{val.mime_type or 'N/A'}`")
                    with col_b:
                        st.write(f"**File Size:** `{format_bytes(val.file_size_bytes)}`")
                        st.write(f"**SHA-256:** `{val.sha256_hash[:16]}...`" if val.sha256_hash else "**SHA-256:** `N/A`")
                        st.write(f"**Storage Path:** `{res.metadata.storage_path if res.metadata else 'Not saved'}`")
                        st.write(f"**Status Details:** {res.message}")

                    if val.sha256_hash:
                        st.caption(f"Full SHA-256 Digest: `{val.sha256_hash}`")


# TAB 2: Verification Dashboard
with tab_dashboard:
    st.markdown("### Ingested Documents Registry")

    # Real-time KPIs
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
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

    st.markdown("---")

    # Filters and search
    filt_col1, filt_col2 = st.columns([1, 3])
    with filt_col1:
        category_filter = st.selectbox(
            "Filter by Category",
            options=["All", "Documents", "Images", "Videos"],
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
            table_data.append(
                {
                    "Document ID": d.document_id,
                    "Original Filename": d.original_filename,
                    "Category": d.category,
                    "Type": d.file_type,
                    "Size": format_bytes(d.file_size_bytes),
                    "SHA-256": d.sha256_hash[:12] + "...",
                    "Upload Date": d.upload_timestamp[:19].replace("T", " "),
                    "Status": d.status,
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

                # Preview section based on category
                if disk_exists:
                    try:
                        file_bytes = storage_mgr.read_file(selected_doc.storage_path)
                        if selected_doc.category == "images":
                            st.image(file_bytes, caption=selected_doc.original_filename, use_container_width=True)
                        elif selected_doc.file_type == ".txt":
                            st.text_area("File Content Preview (first 1000 chars):", file_bytes[:1000].decode("utf-8", errors="replace"), height=200)
                        elif selected_doc.category == "videos":
                            st.video(file_bytes)
                        elif selected_doc.file_type == ".pdf":
                            st.caption(f"PDF Header: {file_bytes[:16].decode('latin-1', errors='ignore')}")
                    except Exception as preview_err:
                        st.caption(f"Preview unavailable: {preview_err}")

    else:
        st.info("No documents found matching the filter criteria. Use the Upload tab to ingest documents.")


# TAB 3: System & Validation Audit
with tab_audit:
    st.markdown("### Phase 1 System Architecture & Validation Rules")

    st.markdown(
        """
        #### 1. Ingestion Boundaries (Strict Phase 1 Scope)
        This module establishes the secure foundation for the UpToSkills RAG pipeline:
        - **In Scope (Implemented):**
          - Multi-format file upload and buffering.
          - Category categorization (Documents, Images, Videos).
          - MIME and magic-bytes header verification.
          - Empty (0-byte) and corrupted file rejection.
          - Configurable file size limits per category.
          - Accidental overwrite prevention and safe filename sanitization.
          - Cryptographic duplicate detection via SHA-256 hashing.
          - Persistent SQLite metadata store with indexed lookup.
          - Streamlit verification dashboard.
        - **Out of Scope (Reserved for Phase 2):**
          - Text extraction & OCR (PyPDF2, Tesseract, pdfplumber).
          - Video audio extraction and transcription (Whisper).
          - Preprocessing, text cleaning, chunking.
          - Embedding generation & ChromaDB vector ingestion.
        """
    )

    st.markdown("#### 2. Configured Validation Rules")
    rule_data = [
        {"Category": "Documents", "Extensions": ", ".join(ALLOWED_EXTENSIONS["documents"]), "Max Size": f"{DEFAULT_MAX_SIZES_MB['documents']} MB", "Magic Check": "%PDF- header, Word ZIP archive structure, UTF-8 text integrity"},
        {"Category": "Images", "Extensions": ", ".join(ALLOWED_EXTENSIONS["images"]), "Max Size": f"{DEFAULT_MAX_SIZES_MB['images']} MB", "Magic Check": "PNG magic signature, JPEG SOI marker, RIFF/WEBP + Pillow verification"},
        {"Category": "Videos", "Extensions": ", ".join(ALLOWED_EXTENSIONS["videos"]), "Max Size": f"{DEFAULT_MAX_SIZES_MB['videos']} MB", "Magic Check": "MP4 ftyp atom, MOV atom, RIFF AVI chunk, MKV/WEBM EBML header"},
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
