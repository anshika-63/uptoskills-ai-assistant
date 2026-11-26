"""
Pytest configuration, shared fixtures, and sample data generators
for the UpToSkills RAG Ingestion Pipeline tests.
"""

import io
import math
import struct
import sys
import wave
import zipfile
from pathlib import Path
import pytest
from PIL import Image

# Ensure repository root and backend directory are in sys.path
backend_dir = Path(__file__).resolve().parent.parent
project_root = backend_dir.parent

for p in (str(project_root), str(backend_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.ingestion.metadata import MetadataManager
from backend.ingestion.pipeline import IngestionPipeline
from backend.ingestion.storage import StorageManager
from backend.ingestion.validator import FileValidator


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Provides an isolated temporary storage directory."""
    storage_dir = tmp_path / "test_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@pytest.fixture
def temp_db_path(tmp_path):
    """Provides an isolated temporary SQLite database path."""
    return tmp_path / "test_metadata.db"


@pytest.fixture
def storage_manager(temp_storage_dir):
    """Provides a StorageManager instance using temporary directory."""
    return StorageManager(base_dir=temp_storage_dir)


@pytest.fixture
def metadata_manager(temp_db_path):
    """Provides a MetadataManager instance using temporary database."""
    return MetadataManager(db_path=temp_db_path)


@pytest.fixture
def validator():
    """Provides a FileValidator instance with default rules."""
    return FileValidator()


@pytest.fixture
def pipeline(validator, storage_manager, metadata_manager):
    """Provides a full IngestionPipeline instance with isolated storage and DB."""
    return IngestionPipeline(
        validator=validator,
        storage_manager=storage_manager,
        metadata_manager=metadata_manager,
    )


@pytest.fixture
def multimodal_pipeline(tmp_path, validator, storage_manager, metadata_manager):
    """Provides an IngestionPipeline instance with multimodal processing enabled and isolated ChromaDB."""
    from backend.ingestion.embeddings import EmbeddingService
    from backend.ingestion.vector_store import VectorStoreManager

    chroma_dir = tmp_path / "mm_chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    emb_svc = EmbeddingService()
    v_store = VectorStoreManager(
        persist_dir=chroma_dir,
        collection_name="test_multimodal_rag",
        embedding_service=emb_svc,
    )
    return IngestionPipeline(
        validator=validator,
        storage_manager=storage_manager,
        metadata_manager=metadata_manager,
        embedding_service=emb_svc,
        vector_store=v_store,
        enable_multimodal=True,
    )


# Sample content generators
def create_sample_pdf() -> bytes:
    """Generates a valid minimalist PDF file buffer."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n123\n%%EOF"
    )


def create_sample_docx() -> bytes:
    """
    Generates a *minimal* DOCX that passes format validation.
    For extraction tests, use create_real_docx() which builds a fully
    parseable document via python-docx.
    """
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types></Types>')
        zf.writestr("word/document.xml", '<?xml version="1.0" encoding="UTF-8"?><w:document></w:document>')
    return bio.getvalue()


def create_real_docx(paragraphs=None, tables=None) -> bytes:
    """
    Generates a fully parseable DOCX via python-docx.
    paragraphs: list[str]  — text paragraphs to add
    tables: list[list[list[str]]]  — table → rows → cells
    """
    from docx import Document as DocxDoc

    doc = DocxDoc()
    for p in (paragraphs or ["Sample paragraph for RAG extraction test."]):
        doc.add_paragraph(p)
    for tbl in (tables or []):
        if not tbl:
            continue
        rows, cols = len(tbl), len(tbl[0])
        word_table = doc.add_table(rows=rows, cols=cols)
        for r_i, row_data in enumerate(tbl):
            for c_i, cell_text in enumerate(row_data):
                word_table.cell(r_i, c_i).text = cell_text
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


def create_sample_txt() -> bytes:
    """Generates a valid UTF-8 text buffer."""
    return "This is a valid UpToSkills test document for the RAG ingestion pipeline.".encode("utf-8")


def create_sample_png() -> bytes:
    """Generates a valid PNG image buffer using Pillow."""
    bio = io.BytesIO()
    img = Image.new("RGB", (20, 20), color=(73, 109, 137))
    img.save(bio, format="PNG")
    return bio.getvalue()


def create_sample_jpeg() -> bytes:
    """Generates a valid JPEG image buffer using Pillow."""
    bio = io.BytesIO()
    img = Image.new("RGB", (20, 20), color=(255, 0, 0))
    img.save(bio, format="JPEG")
    return bio.getvalue()


def create_sample_webp() -> bytes:
    """Generates a valid WEBP image buffer using Pillow."""
    bio = io.BytesIO()
    img = Image.new("RGB", (20, 20), color=(0, 255, 0))
    img.save(bio, format="WEBP")
    return bio.getvalue()


def create_sample_mp4() -> bytes:
    """Generates a valid MP4 header buffer (ISO BMFF with ftyp atom)."""
    # 4 bytes size (0x1c = 28), 4 bytes 'ftyp', major_brand 'isom', minor_version 0x200, compatible 'isom', 'iso2', 'mp41'
    return b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08mdat"


def create_sample_mov() -> bytes:
    """Generates a valid QuickTime MOV header buffer."""
    return b"\x00\x00\x00\x14ftypqt  \x20\x05\x03\x00qt  \x00\x00\x00\x08mdat"


def create_sample_avi() -> bytes:
    """Generates a valid AVI header buffer."""
    return b"RIFF\x24\x00\x00\x00AVI LIST\x00\x00\x00\x00"


def create_sample_mkv() -> bytes:
    """Generates a valid Matroska MKV EBML header buffer."""
    return b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01\x42\xf2\x81\x04\x42\xf3\x81\x08\x42\x82\x84matroska"


def create_sample_webm() -> bytes:
    """Generates a valid WebM EBML header buffer."""
    return b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01\x42\xf2\x81\x04\x42\xf3\x81\x08\x42\x82\x84webm"


# Audio Sample Generators
def create_sample_wav(duration_s: float = 0.5, freq: float = 440.0, sample_rate: int = 16000) -> bytes:
    """Generates a genuine 16kHz mono 16-bit PCM WAV audio buffer."""
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        total_samples = int(sample_rate * duration_s)
        frames = []
        for i in range(total_samples):
            val = int(32767.0 * 0.3 * math.sin(2.0 * math.pi * freq * i / sample_rate))
            frames.append(struct.pack("<h", val))
        wf.writeframes(b"".join(frames))
    return bio.getvalue()


def create_sample_mp3() -> bytes:
    """Generates a valid MP3 buffer with ID3v2 header and MPEG audio frame sync."""
    # ID3v2 header (10 bytes) + MPEG audio frame sync header (0xFFFB) + padding
    id3 = b"ID3\x04\x00\x00\x00\x00\x00\x00"
    mpeg_frame = b"\xff\xfb\x90\x64" + b"\x00" * 200
    return id3 + mpeg_frame


def create_sample_ogg() -> bytes:
    """Generates a valid Ogg container buffer."""
    return b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00" + b"\x00" * 100


def create_sample_flac() -> bytes:
    """Generates a valid FLAC stream header buffer."""
    return b"fLaC\x00\x00\x00\x22" + b"\x00" * 100


def create_sample_m4a() -> bytes:
    """Generates a valid M4A ISO BMFF header buffer."""
    return b"\x00\x00\x00\x1cftypM4A \x00\x00\x02\x00M4A mp42isom" + b"\x00" * 100


def create_sample_aac() -> bytes:
    """Generates a valid AAC ADTS stream header buffer."""
    # ADTS sync word: 12 bits 0xFFF + layer/profile bits
    return b"\xff\xf1\x50\x80\x01\x20\x00" + b"\x00" * 100


def create_scanned_pdf() -> bytes:
    """
    Generates a genuine scanned PDF: single page containing only a raster image
    and zero digital text.
    """
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    img_bytes = create_sample_png()
    page.insert_image(pymupdf.Rect(0, 0, 200, 200), stream=img_bytes)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


def create_real_video_mp4(duration_s: float = 0.5) -> bytes:
    """
    Generates a real, playable MP4 video file with both video and AAC audio tracks
    using FFmpeg.
    """
    import subprocess
    import tempfile
    import os
    from backend.ingestion.audio_transcriber import get_ffmpeg_path

    ff_bin = get_ffmpeg_path()
    tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tf.close()
    out_path = tf.name

    cmd = [
        ff_bin,
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration_s}:size=160x120:rate=10",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration_s}",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        out_path,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        with open(out_path, "rb") as f:
            data = f.read()
        return data
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)
