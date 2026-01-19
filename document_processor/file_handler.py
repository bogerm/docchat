from __future__ import annotations

import os
import json
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence, Any, Optional, List

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from loguru import logger

from config import constants
from config.settings import settings

logger.info(f"Loading DocumentProcessor with cache directory: {settings.CACHE_DIR}")


class DocumentProcessor:
    """
    Processes documents into LangChain Document chunks using Docling -> Markdown -> MarkdownHeaderTextSplitter.

    Caches extracted Markdown to disk to avoid re-running Docling conversion on the same file content.
      - {sha256}.md
      - {sha256}.meta.json

    Cache validity is based on cache file mtime + CACHE_EXPIRE_DAYS (simple and robust).
    """

    CACHE_VERSION = 1

    def __init__(
        self,
        *,
        headers: Optional[list[tuple[str, str]]] = None,
        max_chars_per_chunk_for_dedup: int = 50_000,
    ):
        self.headers = headers or [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.converter = DocumentConverter()
        self.splitter = MarkdownHeaderTextSplitter(self.headers)

        self.max_chars_per_chunk_for_dedup = max_chars_per_chunk_for_dedup
        self.allowed_exts = {ext.lower() for ext in (constants.ALLOWED_TYPES or [])}

    # ----------------------------
    # Validation / file handling
    # ----------------------------

    def _get_path(self, file: Any) -> Optional[Path]:
        name = getattr(file, "name", None)
        if not name:
            return None
        p = Path(name)
        return p if p.exists() else None

    def _file_size(self, file: Any) -> int:
        p = self._get_path(file)
        if p is not None:
            return p.stat().st_size

        fobj = getattr(file, "file", None) or getattr(file, "stream", None) or file
        if hasattr(fobj, "seek") and hasattr(fobj, "tell"):
            pos = fobj.tell()
            fobj.seek(0, os.SEEK_END)
            size = fobj.tell()
            fobj.seek(pos, os.SEEK_SET)
            return int(size)

        raise ValueError("Cannot determine file size (no path and not a seekable stream).")

    def _ext_ok(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.allowed_exts

    def validate_files(self, files: Sequence[Any]) -> None:
        total_size = 0
        allowed_str = ", ".join(sorted(self.allowed_exts)) if self.allowed_exts else "(none)"

        for f in files:
            file_path = self._get_path(f)
            if file_path is None:
                raise ValueError(f"Cannot access file path for upload: {getattr(f, 'name', 'unknown')}")

            if self.allowed_exts and not self._ext_ok(file_path):
                raise ValueError(f"Unsupported file type: {file_path.name}. Allowed: {allowed_str}")

            size = self._file_size(f)
            total_size += size

            if size > constants.MAX_FILE_SIZE:
                raise ValueError(f"File '{file_path.name}' exceeds {constants.MAX_FILE_SIZE // 1024 // 1024}MB limit")

        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(f"Total size exceeds {constants.MAX_TOTAL_SIZE // 1024 // 1024}MB limit")

    # ----------------------------
    # Core processing
    # ----------------------------

    def process(self, files: Sequence[Any]) -> List[Any]:
        """
        Returns a list of LangChain Document chunks.
        Uses markdown cache for subsequent runs.
        """
        self.validate_files(files)

        all_chunks: list[Any] = []
        seen_hashes: set[str] = set()

        for f in files:
            try:
                file_path = self._get_path(f)
                if file_path is None:
                    logger.warning(f"Skipping file without readable path: {getattr(f, 'name', 'unknown')}")
                    continue

                file_hash = self._hash_file(file_path)
                md_cache_path = self.cache_dir / f"{file_hash}.md"
                meta_cache_path = self.cache_dir / f"{file_hash}.meta.json"

                if self._is_cache_valid(md_cache_path):
                    logger.info(f"Loading markdown from cache: {file_path.name}")
                    markdown = md_cache_path.read_text(encoding="utf-8", errors="ignore")
                else:
                    logger.info(f"Converting and caching markdown: {file_path.name}")
                    markdown = self._convert_to_markdown(file_path)
                    self._save_markdown_cache(md_cache_path, markdown)

                # Always (re)write meta when we process or when cache is missing
                # If cache is valid and meta exists, we keep it as-is.
                if (not meta_cache_path.exists()) or (not self._is_cache_valid(md_cache_path)):
                    self._write_meta(meta_cache_path, file_path=file_path, file_hash=file_hash, md_cache_path=md_cache_path)

                chunks = self.splitter.split_text(markdown)

                # Deduplicate chunks across files
                for chunk in chunks:
                    text = (getattr(chunk, "page_content", "") or "").strip()
                    if not text:
                        continue
                    text_for_hash = text[: self.max_chars_per_chunk_for_dedup]
                    ch = hashlib.sha256(text_for_hash.encode("utf-8", errors="ignore")).hexdigest()
                    if ch not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(ch)

            except Exception as e:
                logger.error(f"Failed to process {getattr(f, 'name', 'unknown')}: {type(e).__name__}: {e}")
                continue

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    def _convert_to_markdown(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        if suffix == ".md":
            return file_path.read_text(encoding="utf-8", errors="ignore")

        if suffix == ".txt":
            return file_path.read_text(encoding="utf-8", errors="ignore")

        # pdf/docx => docling conversion
        return self.converter.convert(str(file_path)).document.export_to_markdown()

    # ----------------------------
    # Cache + hashing + metadata
    # ----------------------------

    def _hash_file(self, file_path: Path, chunk_size: int = 1024 * 1024) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def _save_markdown_cache(self, cache_path: Path, markdown: str) -> None:
        cache_path.write_text(markdown or "", encoding="utf-8", errors="ignore")

    def _write_meta(self, meta_path: Path, *, file_path: Path, file_hash: str, md_cache_path: Path) -> None:
        st = file_path.stat()
        now = datetime.now(timezone.utc).isoformat()

        meta = {
            "cache_version": self.CACHE_VERSION,
            "hash_algo": "sha256",
            "file_hash": file_hash,
            "original_filename": file_path.name,
            "original_suffix": file_path.suffix.lower(),
            "original_size_bytes": st.st_size,
            "original_mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "cached_markdown_path": md_cache_path.name,  # relative within cache_dir
            "cached_at_utc": now,
            "cache_expires_days": settings.CACHE_EXPIRE_DAYS,
            "allowed_types": sorted(self.allowed_exts),
        }

        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)

    def cleanup_cache(self) -> int:
        """
        Remove expired markdown cache entries and their sidecar meta files.

        Returns:
            int: number of markdown cache entries removed
        """
        removed = 0
        now = datetime.now()
        ttl = timedelta(days=settings.CACHE_EXPIRE_DAYS)

        for md_path in self.cache_dir.glob("*.md"):
            try:
                age = now - datetime.fromtimestamp(md_path.stat().st_mtime)
                if age <= ttl:
                    continue

                meta_path = md_path.with_suffix(".meta.json")

                md_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)

                removed += 1
                logger.debug(f"Removed expired cache: {md_path.name}")

            except Exception as e:
                logger.warning(
                    f"Failed to clean cache entry {md_path.name}: "
                    f"{type(e).__name__}: {e}"
                )

        if removed:
            logger.info(f"Cache cleanup complete: removed {removed} expired entries")
        else:
            logger.debug("Cache cleanup complete: no expired entries found")

        return removed