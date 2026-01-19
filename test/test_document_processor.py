import os
import json
import time
from pathlib import Path

import pytest

from document_processor.file_handler import DocumentProcessor


class UploadStub:
    """Mimics Gradio/FastAPI upload object with a .name path."""
    def __init__(self, path: Path):
        self.name = str(path)


@pytest.fixture
def processor(tmp_path, monkeypatch):
    # Patch settings used by the module under test
    from config.settings import settings

    monkeypatch.setattr(settings, "CACHE_DIR", str(tmp_path / "cache"), raising=False)
    monkeypatch.setattr(settings, "CACHE_EXPIRE_DAYS", 7, raising=False)

    # Patch constants to small values for tests
    from config import constants
    monkeypatch.setattr(constants, "ALLOWED_TYPES", [".txt", ".pdf", ".docx", ".md"], raising=False)
    monkeypatch.setattr(constants, "MAX_FILE_SIZE", 50 * 1024 * 1024, raising=False)
    monkeypatch.setattr(constants, "MAX_TOTAL_SIZE", 200 * 1024 * 1024, raising=False)

    return DocumentProcessor()


def test_process_txt_creates_md_cache_and_meta(processor, tmp_path):
    # Arrange
    f = tmp_path / "a.txt"
    f.write_text("Hello\n\n## Section\nWorld", encoding="utf-8")

    # Act
    chunks = processor.process([UploadStub(f)])

    # Assert chunks
    assert len(chunks) >= 1
    assert any("Hello" in (c.page_content or "") for c in chunks)

    # Assert cache files exist
    cache_dir = Path(processor.cache_dir)
    md_files = list(cache_dir.glob("*.md"))
    meta_files = list(cache_dir.glob("*.meta.json"))
    assert len(md_files) == 1
    assert len(meta_files) == 1

    # Assert meta content
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    assert meta["original_filename"] == "a.txt"
    assert meta["file_hash"] in md_files[0].name
    assert meta["cached_markdown_path"] == md_files[0].name
    assert "cached_at_utc" in meta


def test_process_md_reads_without_docling(processor, tmp_path, monkeypatch):
    f = tmp_path / "b.md"
    f.write_text("# Title\n\nSome text", encoding="utf-8")

    def boom(*args, **kwargs):
        raise AssertionError("Docling should not be called for .md")

    monkeypatch.setattr(processor.converter, "convert", boom)

    chunks = processor.process([UploadStub(f)])

    assert len(chunks) >= 1

    # Body should definitely appear in page_content
    assert any("Some text" in (c.page_content or "") for c in chunks)

    # Header is usually stored in metadata by MarkdownHeaderTextSplitter
    assert any(
        any("Title" in str(v) for v in (getattr(c, "metadata", {}) or {}).values())
        for c in chunks
    )



def test_cache_hit_avoids_docling_for_pdf_docx(processor, tmp_path, monkeypatch):
    """
    For pdf/docx we normally call Docling. This test ensures that when a cache
    file already exists and is valid, we don't call Docling again.
    """
    # Arrange: create a fake "pdf" (content doesn't matter; we won't parse it on cache hit)
    f = tmp_path / "c.pdf"
    f.write_bytes(b"%PDF-1.4 fake pdf bytes")

    # Prime the cache by forcing _convert_to_markdown to return markdown without using Docling
    monkeypatch.setattr(processor, "_convert_to_markdown", lambda p: "# Cached\n\nBody")

    # First run creates cache
    chunks1 = processor.process([UploadStub(f)])
    assert len(chunks1) >= 1

    # Now make _convert_to_markdown fail if called again (cache-hit should skip conversion)
    def fail(*args, **kwargs):
        raise AssertionError("_convert_to_markdown should not be called on cache hit")

    monkeypatch.setattr(processor, "_convert_to_markdown", fail)

    # Second run should hit cache and not call conversion
    chunks2 = processor.process([UploadStub(f)])
    assert len(chunks2) >= 1


def test_validate_rejects_unsupported_extension(processor, tmp_path):
    f = tmp_path / "bad.exe"
    f.write_bytes(b"nope")
    with pytest.raises(ValueError) as e:
        processor.process([UploadStub(f)])
    assert "Unsupported file type" in str(e.value)


def test_validate_rejects_file_too_large(processor, tmp_path, monkeypatch):
    from config import constants
    monkeypatch.setattr(constants, "MAX_FILE_SIZE", 10, raising=False)  # 10 bytes max

    f = tmp_path / "big.txt"
    f.write_text("01234567890", encoding="utf-8")  # 11 bytes
    with pytest.raises(ValueError) as e:
        processor.process([UploadStub(f)])
    assert "exceeds" in str(e.value).lower()


def test_validate_rejects_total_size_too_large(processor, tmp_path, monkeypatch):
    from config import constants
    monkeypatch.setattr(constants, "MAX_TOTAL_SIZE", 15, raising=False)  # 15 bytes total

    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.write_text("1234567890", encoding="utf-8")  # 10
    f2.write_text("1234567890", encoding="utf-8")  # 10 => total 20
    with pytest.raises(ValueError) as e:
        processor.process([UploadStub(f1), UploadStub(f2)])
    assert "total size exceeds" in str(e.value).lower()


def test_cleanup_cache_removes_expired_entries(processor, tmp_path, monkeypatch):
    # Arrange: create a cache entry directly
    cache_dir = Path(processor.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    md = cache_dir / "deadbeef.md"
    meta = cache_dir / "deadbeef.meta.json"
    md.write_text("cached markdown", encoding="utf-8")
    meta.write_text('{"ok": true}', encoding="utf-8")

    # Make it "old" by setting mtime far in the past
    old_time = time.time() - (60 * 60 * 24 * 30)  # 30 days ago
    os.utime(md, (old_time, old_time))
    os.utime(meta, (old_time, old_time))

    from config.settings import settings
    monkeypatch.setattr(settings, "CACHE_EXPIRE_DAYS", 7, raising=False)

    # Act
    removed = processor.cleanup_cache()

    # Assert
    assert removed == 1
    assert not md.exists()
    assert not meta.exists()
