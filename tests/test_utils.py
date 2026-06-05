"""Tests for sherox.utils module."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sherox.utils import download_file


class TestDownloadFile:
    """Tests for download_file function."""

    def test_downloads_from_scratch_when_no_file_exists(self, tmp_path):
        """Test that download starts from beginning when file doesn't exist."""
        dest = tmp_path / "test.txt"
        url = "http://example.com/test.txt"

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Length": "100"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b"chunk1", b"chunk2", b""]

        with patch("urllib.request.urlopen", return_value=mock_response):
            download_file(url, dest)

        assert dest.exists()
        assert dest.read_text() == "chunk1chunk2"

    def test_accepts_string_destination(self, tmp_path):
        """Test that string paths are accepted as documented."""
        dest = str(tmp_path / "test.txt")
        url = "http://example.com/test.txt"

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Length": "5"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b"hello", b""]

        with patch("urllib.request.urlopen", return_value=mock_response):
            download_file(url, dest)

        assert Path(dest).read_text() == "hello"

    def test_resumes_download_when_partial_file_exists(self, tmp_path):
        """Test that download resumes from existing file position."""
        dest = tmp_path / "test.txt"
        url = "http://example.com/test.txt"

        # Create partial file with existing content
        dest.write_bytes(b"existing")

        mock_response = MagicMock()
        mock_response.status = 206
        mock_response.headers = {"Content-Range": "bytes 8-15/16"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b"chunk", b""]

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            download_file(url, dest)

        # Verify Range header was added
        call_args = mock_urlopen.call_args
        assert call_args[0][0].headers.get("Range") == "bytes=8-"

        # Verify file was appended to, not overwritten
        assert dest.read_bytes() == b"existingchunk"

    def test_handles_missing_content_length(self, tmp_path):
        """Test that download works when Content-Length is missing."""
        dest = tmp_path / "test.txt"
        url = "http://example.com/test.txt"

        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b"chunk1", b""]

        with patch("urllib.request.urlopen", return_value=mock_response):
            download_file(url, dest)

        assert dest.exists()
        assert dest.read_text() == "chunk1"
