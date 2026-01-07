import os
import pytest
import tempfile
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from document_processor.file_handler import DocumentProcessor


class TestDocumentProcessor:
    """Test suite for DocumentProcessor class"""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance with a temporary cache directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('document_processor.file_handler.settings') as mock_settings:
                mock_settings.CACHE_DIR = tmpdir
                mock_settings.CACHE_EXPIRE_DAYS = 7
                yield DocumentProcessor()

    @pytest.fixture
    def sample_file(self):
        """Create a temporary sample file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Sample content for testing")
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def sample_pdf_file(self):
        """Create a temporary PDF file"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            # Minimal PDF content
            f.write(b'%PDF-1.4\n')
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_initialization(self, processor):
        """Test DocumentProcessor initialization"""
        assert processor.headers == [("#", "Header 1"), ("##", "Header 2")]
        assert processor.cache_dir.exists()
        assert isinstance(processor.cache_dir, Path)

    def test_validate_files_success(self, processor, sample_file):
        """Test file validation with valid file sizes"""
        mock_file = Mock()
        mock_file.name = sample_file
        
        # Should not raise any exception
        processor.validate_files([mock_file])

    def test_validate_files_exceeds_limit(self, processor):
        """Test file validation when total size exceeds limit"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            # Create a file larger than MAX_TOTAL_SIZE
            f.write(b'0' * (210 * 1024 * 1024))  # 210MB
            large_file = f.name
        
        try:
            mock_file = Mock()
            mock_file.name = large_file
            
            with pytest.raises(ValueError, match="Total size exceeds"):
                processor.validate_files([mock_file])
        finally:
            if os.path.exists(large_file):
                os.unlink(large_file)

    def test_validate_files_multiple_files(self, processor):
        """Test file validation with multiple files"""
        files = []
        temp_paths = []
        
        try:
            for i in range(3):
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                    f.write(f"Content {i}")
                    temp_paths.append(f.name)
                    mock_file = Mock()
                    mock_file.name = f.name
                    files.append(mock_file)
            
            # Should not raise exception
            processor.validate_files(files)
        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    os.unlink(path)

    def test_generate_hash(self, processor):
        """Test hash generation"""
        content1 = b"test content"
        content2 = b"test content"
        content3 = b"different content"
        
        hash1 = processor._generate_hash(content1)
        hash2 = processor._generate_hash(content2)
        hash3 = processor._generate_hash(content3)
        
        assert hash1 == hash2  # Same content should produce same hash
        assert hash1 != hash3  # Different content should produce different hash
        assert len(hash1) == 64  # SHA256 produces 64 character hex string

    def test_save_and_load_cache(self, processor):
        """Test saving and loading from cache"""
        # Create simple picklable objects to represent chunks
        from types import SimpleNamespace
        mock_chunk1 = SimpleNamespace(page_content="Content 1", metadata={})
        mock_chunk2 = SimpleNamespace(page_content="Content 2", metadata={})
        chunks = [mock_chunk1, mock_chunk2]
        
        cache_path = processor.cache_dir / "test_cache.pkl"
        
        # Save to cache
        processor._save_to_cache(chunks, cache_path)
        assert cache_path.exists()
        
        # Load from cache
        loaded_chunks = processor._load_from_cache(cache_path)
        assert len(loaded_chunks) == 2
        assert loaded_chunks[0].page_content == "Content 1"
        assert loaded_chunks[1].page_content == "Content 2"

    def test_is_cache_valid_fresh(self, processor):
        """Test cache validation for fresh cache"""
        cache_path = processor.cache_dir / "fresh_cache.pkl"
        
        # Create a fresh cache file
        with open(cache_path, 'wb') as f:
            pickle.dump({
                "timestamp": datetime.now().timestamp(),
                "chunks": []
            }, f)
        
        assert processor._is_cache_valid(cache_path) is True

    def test_is_cache_valid_expired(self, processor):
        """Test cache validation for expired cache"""
        cache_path = processor.cache_dir / "expired_cache.pkl"
        
        # Create an old cache file by modifying its modification time
        with open(cache_path, 'wb') as f:
            pickle.dump({
                "timestamp": datetime.now().timestamp(),
                "chunks": []
            }, f)
        
        # Set modification time to 10 days ago
        old_time = (datetime.now() - timedelta(days=10)).timestamp()
        os.utime(cache_path, (old_time, old_time))
        
        assert processor._is_cache_valid(cache_path) is False

    def test_is_cache_valid_nonexistent(self, processor):
        """Test cache validation for non-existent cache file"""
        cache_path = processor.cache_dir / "nonexistent.pkl"
        assert processor._is_cache_valid(cache_path) is False

    @patch('document_processor.file_handler.DocumentConverter')
    @patch('document_processor.file_handler.MarkdownHeaderTextSplitter')
    def test_process_file_pdf(self, mock_splitter_class, mock_converter_class, processor, sample_pdf_file):
        """Test processing a PDF file"""
        # Setup mocks
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        mock_result = Mock()
        mock_result.document.export_to_markdown.return_value = "# Header\nContent"
        mock_converter.convert.return_value = mock_result
        
        mock_splitter = Mock()
        mock_chunk = Mock()
        mock_chunk.page_content = "Content"
        mock_splitter.split_text.return_value = [mock_chunk]
        mock_splitter_class.return_value = mock_splitter
        
        mock_file = Mock()
        mock_file.name = sample_pdf_file
        
        # Process file
        chunks = processor._process_file(mock_file)
        
        assert len(chunks) == 1
        assert chunks[0].page_content == "Content"
        mock_converter.convert.assert_called_once_with(sample_pdf_file)

    def test_process_file_unsupported_type(self, processor):
        """Test processing an unsupported file type"""
        mock_file = Mock()
        mock_file.name = "test.xyz"
        
        chunks = processor._process_file(mock_file)
        assert chunks == []

    @patch('document_processor.file_handler.DocumentConverter')
    @patch('document_processor.file_handler.MarkdownHeaderTextSplitter')
    def test_process_with_cache_hit(self, mock_splitter_class, mock_converter_class, processor, sample_pdf_file):
        """Test processing files with cache hit"""
        # Create picklable objects to represent chunks
        from types import SimpleNamespace
        mock_chunk = SimpleNamespace(page_content="Cached content", metadata={})
        
        mock_file = Mock()
        mock_file.name = sample_pdf_file
        
        # Pre-populate cache
        with open(sample_pdf_file, 'rb') as f:
            file_hash = processor._generate_hash(f.read())
        cache_path = processor.cache_dir / f"{file_hash}.pkl"
        processor._save_to_cache([mock_chunk], cache_path)
        
        # Process (should hit cache)
        chunks = processor.process([mock_file])
        
        assert len(chunks) == 1
        assert chunks[0].page_content == "Cached content"
        # Converter should not be called when cache is hit
        mock_converter_class.assert_not_called()

    @patch('document_processor.file_handler.DocumentConverter')
    @patch('document_processor.file_handler.MarkdownHeaderTextSplitter')
    def test_process_with_cache_miss(self, mock_splitter_class, mock_converter_class, processor, sample_pdf_file):
        """Test processing files with cache miss"""
        from types import SimpleNamespace
        
        # Setup mocks
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        mock_result = Mock()
        mock_result.document.export_to_markdown.return_value = "# Header\nContent"
        mock_converter.convert.return_value = mock_result
        
        mock_splitter = Mock()
        mock_chunk = SimpleNamespace(page_content="New content", metadata={})
        mock_splitter.split_text.return_value = [mock_chunk]
        mock_splitter_class.return_value = mock_splitter
        
        mock_file = Mock()
        mock_file.name = sample_pdf_file
        
        # Process (cache miss)
        chunks = processor.process([mock_file])
        
        assert len(chunks) == 1
        assert chunks[0].page_content == "New content"
        mock_converter.convert.assert_called_once()

    @patch('document_processor.file_handler.DocumentConverter')
    @patch('document_processor.file_handler.MarkdownHeaderTextSplitter')
    def test_process_deduplication(self, mock_splitter_class, mock_converter_class, processor):
        """Test chunk deduplication across files"""
        from types import SimpleNamespace
        
        # Setup mocks
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        mock_result = Mock()
        mock_result.document.export_to_markdown.return_value = "Content"
        mock_converter.convert.return_value = mock_result
        
        mock_splitter = Mock()
        # Create two chunks with same content using picklable objects
        mock_chunk1 = SimpleNamespace(page_content="Duplicate content", metadata={})
        mock_splitter.split_text.return_value = [mock_chunk1]
        mock_splitter_class.return_value = mock_splitter
        
        # Create two files
        files = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
                f.write(b'%PDF-1.4\n')
                mock_file = Mock()
                mock_file.name = f.name
                files.append(mock_file)
        
        try:
            # Process files
            chunks = processor.process(files)
            
            # Should only have 1 chunk due to deduplication
            assert len(chunks) == 1
        finally:
            for mock_file in files:
                if os.path.exists(mock_file.name):
                    os.unlink(mock_file.name)

    @patch('document_processor.file_handler.DocumentConverter')
    def test_process_handles_errors(self, mock_converter_class, processor, sample_pdf_file):
        """Test that process handles file processing errors gracefully"""
        # Setup mock to raise exception
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        mock_converter.convert.side_effect = Exception("Processing error")
        
        mock_file = Mock()
        mock_file.name = sample_pdf_file
        
        # Should not raise exception, just log and continue
        chunks = processor.process([mock_file])
        assert chunks == []

    @patch('document_processor.file_handler.DocumentConverter')
    @patch('document_processor.file_handler.MarkdownHeaderTextSplitter')
    def test_process_multiple_files(self, mock_splitter_class, mock_converter_class, processor):
        """Test processing multiple files"""
        from types import SimpleNamespace
        
        # Setup mocks
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        mock_result = Mock()
        mock_result.document.export_to_markdown.return_value = "Content"
        mock_converter.convert.return_value = mock_result
        
        mock_splitter = Mock()
        mock_splitter_class.return_value = mock_splitter
        
        # Create different chunks for each file
        files = []
        expected_contents = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
                f.write(f'%PDF-1.4\nContent {i}'.encode())
                mock_file = Mock()
                mock_file.name = f.name
                files.append(mock_file)
                expected_contents.append(f"Content {i}")
        
        # Configure splitter to return different chunks based on call count
        call_count = [0]
        def split_side_effect(text):
            chunk = SimpleNamespace(page_content=f"Content {call_count[0]}", metadata={})
            call_count[0] += 1
            return [chunk]
        
        mock_splitter.split_text.side_effect = split_side_effect
        
        try:
            chunks = processor.process(files)
            
            # Should have 3 unique chunks
            assert len(chunks) == 3
            chunk_contents = [c.page_content for c in chunks]
            for expected in expected_contents:
                assert expected in chunk_contents
        finally:
            for mock_file in files:
                if os.path.exists(mock_file.name):
                    os.unlink(mock_file.name)

    def test_supported_file_types(self, processor):
        """Test that all supported file types are handled correctly"""
        supported_types = ['.pdf', '.docx', '.txt', '.md']
        
        for file_type in supported_types:
            mock_file = Mock()
            mock_file.name = f"test{file_type}"
            
            with patch('document_processor.file_handler.DocumentConverter') as mock_conv:
                with patch('document_processor.file_handler.MarkdownHeaderTextSplitter') as mock_split:
                    mock_converter = Mock()
                    mock_conv.return_value = mock_converter
                    mock_result = Mock()
                    mock_result.document.export_to_markdown.return_value = "Content"
                    mock_converter.convert.return_value = mock_result
                    
                    mock_splitter = Mock()
                    mock_splitter.split_text.return_value = []
                    mock_split.return_value = mock_splitter
                    
                    # Should not return empty list for supported types
                    processor._process_file(mock_file)
                    mock_converter.convert.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
