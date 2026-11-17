"""
Unit tests for the FileValidator component.
Verifies format checks, safe filename sanitization, size limits,
empty/corrupted file detection, and signature validation.
"""

from backend.ingestion.models import ValidationStatus
from backend.ingestion.validator import FileValidator
from backend.tests.conftest import (
    create_sample_avi,
    create_sample_docx,
    create_sample_jpeg,
    create_sample_mkv,
    create_sample_mov,
    create_sample_mp4,
    create_sample_pdf,
    create_sample_png,
    create_sample_txt,
    create_sample_webm,
    create_sample_webp,
)


def test_sanitize_filename_normal():
    name = "report_2026.pdf"
    assert FileValidator.sanitize_filename(name) == "report_2026.pdf"


def test_sanitize_filename_path_traversal():
    malicious_names = [
        "../../etc/passwd",
        "..\\..\\windows\\system32\\calc.exe",
        "nested/path/to/my_file.docx",
    ]
    for name in malicious_names:
        sanitized = FileValidator.sanitize_filename(name)
        assert "/" not in sanitized
        assert "\\" not in sanitized
        assert ".." not in sanitized


def test_sanitize_filename_illegal_characters():
    dirty = 'my<cool>:file*"name|?.txt'
    sanitized = FileValidator.sanitize_filename(dirty)
    for bad_char in ['<', '>', ':', '"', '|', '?', '*']:
        assert bad_char not in sanitized
    assert sanitized.endswith(".txt")


def test_sanitize_filename_empty_or_spaces():
    assert FileValidator.sanitize_filename("") == "unnamed_file.bin"
    assert FileValidator.sanitize_filename("    ") == "unnamed_file.bin"
    assert FileValidator.sanitize_filename("...").startswith("unnamed_file")


def test_sanitize_filename_max_length():
    long_name = ("a" * 200) + ".pdf"
    sanitized = FileValidator.sanitize_filename(long_name)
    assert len(sanitized) < 150
    assert sanitized.endswith(".pdf")


def test_category_detection(validator):
    assert validator.get_category_for_extension(".pdf") == "documents"
    assert validator.get_category_for_extension(".docx") == "documents"
    assert validator.get_category_for_extension(".txt") == "documents"
    assert validator.get_category_for_extension(".png") == "images"
    assert validator.get_category_for_extension(".jpg") == "images"
    assert validator.get_category_for_extension(".jpeg") == "images"
    assert validator.get_category_for_extension(".webp") == "images"
    assert validator.get_category_for_extension(".mp4") == "videos"
    assert validator.get_category_for_extension(".mov") == "videos"
    assert validator.get_category_for_extension(".avi") == "videos"
    assert validator.get_category_for_extension(".mkv") == "videos"
    assert validator.get_category_for_extension(".webm") == "videos"
    assert validator.get_category_for_extension(".exe") is None


def test_empty_file_rejected(validator):
    result = validator.validate("empty.pdf", b"")
    assert not result.is_valid
    assert result.status == ValidationStatus.INVALID
    assert "empty" in result.error_message.lower()


def test_unsupported_extension_rejected(validator):
    result = validator.validate("malware.exe", b"MZ\x90\x00\x03\x00\x00\x00")
    assert not result.is_valid
    assert "not supported" in result.error_message.lower()


def test_size_limit_exceeded(validator):
    # Documents limit is 25 MB; simulate a 30 MB byte stream
    size_30mb = 30 * 1024 * 1024
    is_valid, err = validator.validate_size(size_30mb, "documents")
    assert not is_valid
    assert "exceeds" in err.lower()

    # Within limit check
    size_5mb = 5 * 1024 * 1024
    is_valid_small, err_small = validator.validate_size(size_5mb, "documents")
    assert is_valid_small
    assert err_small is None


def test_validate_valid_documents(validator):
    # PDF
    pdf_res = validator.validate("test.pdf", create_sample_pdf())
    assert pdf_res.is_valid
    assert pdf_res.category == "documents"
    assert pdf_res.file_type == ".pdf"
    assert pdf_res.sha256_hash

    # DOCX
    docx_res = validator.validate("test.docx", create_sample_docx())
    assert docx_res.is_valid
    assert docx_res.category == "documents"

    # TXT
    txt_res = validator.validate("notes.txt", create_sample_txt())
    assert txt_res.is_valid
    assert txt_res.category == "documents"


def test_validate_corrupted_pdf_rejected(validator):
    # Fake PDF: does not start with %PDF-
    corrupted_data = b"NOT_A_PDF_CONTENT"
    res = validator.validate("fake.pdf", corrupted_data)
    assert not res.is_valid
    assert "header signature" in res.error_message.lower()


def test_validate_corrupted_docx_rejected(validator):
    # Corrupted DOCX: random bytes that are not a zip archive
    res = validator.validate("fake.docx", b"PK\x01\x02CORRUPTED_ZIP_ARCHIVE")
    assert not res.is_valid
    assert "not a valid zip archive" in res.error_message.lower()


def test_validate_corrupted_txt_null_bytes_rejected(validator):
    # Binary file disguised as txt
    binary_content = b"Some valid text\x00\x00\x01binary bytes"
    res = validator.validate("fake.txt", binary_content)
    assert not res.is_valid
    assert "null bytes" in res.error_message.lower()


def test_validate_valid_images(validator):
    # PNG
    png_res = validator.validate("picture.png", create_sample_png())
    assert png_res.is_valid
    assert png_res.category == "images"

    # JPEG
    jpg_res = validator.validate("picture.jpg", create_sample_jpeg())
    assert jpg_res.is_valid
    assert jpg_res.category == "images"

    # WEBP
    webp_res = validator.validate("picture.webp", create_sample_webp())
    assert webp_res.is_valid
    assert webp_res.category == "images"


def test_validate_corrupted_image_rejected(validator):
    # Starts with PNG magic but truncated corrupted data
    corrupted_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
    res = validator.validate("corrupted.png", corrupted_png)
    assert not res.is_valid
    assert "corrupted image file" in res.error_message.lower()


def test_validate_valid_videos(validator):
    # MP4
    mp4_res = validator.validate("clip.mp4", create_sample_mp4())
    assert mp4_res.is_valid
    assert mp4_res.category == "videos"

    # MOV
    mov_res = validator.validate("clip.mov", create_sample_mov())
    assert mov_res.is_valid
    assert mov_res.category == "videos"

    # AVI
    avi_res = validator.validate("clip.avi", create_sample_avi())
    assert avi_res.is_valid
    assert avi_res.category == "videos"

    # MKV
    mkv_res = validator.validate("clip.mkv", create_sample_mkv())
    assert mkv_res.is_valid
    assert mkv_res.category == "videos"

    # WEBM
    webm_res = validator.validate("clip.webm", create_sample_webm())
    assert webm_res.is_valid
    assert webm_res.category == "videos"


def test_validate_corrupted_video_rejected(validator):
    # MP4 with fake bytes
    res = validator.validate("bad.mp4", b"\x00\x00\x00\x10fakevideoheader")
    assert not res.is_valid
    assert "corrupted mp4" in res.error_message.lower()


def test_sha256_computation():
    data = b"UpToSkills AI Ingestion Pipeline Test"
    hash1 = FileValidator.compute_sha256(data)
    hash2 = FileValidator.compute_sha256(data)
    assert hash1 == hash2
    assert len(hash1) == 64
