"""
Unit tests for the StorageManager component.
Verifies document ID generation, directory structures, categorized persistence,
file retrieval, and accidental overwrite prevention.
"""

from backend.ingestion.storage import StorageManager


def test_storage_init_creates_directories(temp_storage_dir):
    manager = StorageManager(base_dir=temp_storage_dir)
    assert (temp_storage_dir / "documents").exists()
    assert (temp_storage_dir / "images").exists()
    assert (temp_storage_dir / "videos").exists()


def test_generate_document_id():
    id1 = StorageManager.generate_document_id()
    id2 = StorageManager.generate_document_id()
    assert id1.startswith("doc_")
    assert id2.startswith("doc_")
    assert id1 != id2


def test_save_file_categorization(storage_manager, temp_storage_dir):
    content = b"Storage categorization test content"
    doc_id = "doc_test123"
    abs_path, rel_path = storage_manager.save_file(
        document_id=doc_id,
        safe_filename="sample.txt",
        category="documents",
        content=content,
    )

    assert abs_path.exists()
    assert abs_path.parent == temp_storage_dir / "documents"
    assert "documents" in rel_path
    assert abs_path.read_bytes() == content


def test_save_file_overwrite_prevention(storage_manager):
    content_v1 = b"Original file version 1"
    content_v2 = b"Attempted overwrite version 2"
    doc_id = "doc_collision_test"

    path1, _ = storage_manager.save_file(
        document_id=doc_id,
        safe_filename="data.txt",
        category="documents",
        content=content_v1,
    )

    # Attempt saving with identical document_id and filename
    path2, _ = storage_manager.save_file(
        document_id=doc_id,
        safe_filename="data.txt",
        category="documents",
        content=content_v2,
    )

    assert path1 != path2
    assert path1.exists()
    assert path2.exists()
    assert path1.read_bytes() == content_v1
    assert path2.read_bytes() == content_v2


def test_file_exists_read_and_delete(storage_manager):
    doc_id = storage_manager.generate_document_id()
    abs_path, rel_path = storage_manager.save_file(
        document_id=doc_id,
        safe_filename="image.png",
        category="images",
        content=b"fake_image_bytes",
    )

    # Test file_exists
    assert storage_manager.file_exists(rel_path)
    assert storage_manager.file_exists(abs_path)
    assert not storage_manager.file_exists("nonexistent.png")

    # Test read_file
    read_data = storage_manager.read_file(rel_path)
    assert read_data == b"fake_image_bytes"

    # Test delete_file
    deleted = storage_manager.delete_file(rel_path)
    assert deleted
    assert not storage_manager.file_exists(rel_path)
