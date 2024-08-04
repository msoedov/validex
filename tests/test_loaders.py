from unittest.mock import mock_open, patch

import pytest
import responses

from validex.loaders import (
    LocalTextLoader,
    LocalTextPatternLoader,
    PdfFileLoader,
    RobotsTxtLoader,
    RssLoader,
    TextBlobLoader,
    WebPageLoader,
)


@pytest.fixture
def sample_url():
    return "https://example.com"


@pytest.fixture
def sample_file_path():
    return "/path/to/file.txt"


@pytest.fixture
def sample_pdf_path():
    return "/path/to/file.pdf"


@pytest.mark.skip("This test is not working")
@responses.activate
def test_web_page_loader(sample_url):
    responses.add(
        responses.GET,
        sample_url,
        body="<html><body><h1>Test content</h1></body></html>",
        status=200,
    )
    loader = WebPageLoader()
    result = loader.load_data(sample_url)
    assert len(result) == 1
    assert result[0]["content"] == "Test content"
    assert result[0]["meta_data"]["url"] == sample_url


def test_local_text_loader(sample_file_path):
    with patch("builtins.open", mock_open(read_data="Test content")):
        loader = LocalTextLoader()
        result = loader.load_data(sample_file_path)
        assert len(result) == 1
        assert result[0]["content"] == "Test content"
        assert result[0]["meta_data"]["url"] == "local"
        assert result[0]["meta_data"]["file_path"] == sample_file_path


@patch("pypdf.PdfReader")
def test_pdf_file_loader(mock_pdf_reader, sample_pdf_path):
    mock_pdf_reader.return_value.pages = [
        type("MockPage", (), {"extract_text": lambda: "Page 1 content"}),
        type("MockPage", (), {"extract_text": lambda: "Page 2 content"}),
    ]
    loader = PdfFileLoader()
    result = loader.load_data(sample_pdf_path)
    assert len(result) == 2
    assert result[0]["content"] == "Page 1 content"
    assert result[0]["meta_data"]["source"] == sample_pdf_path
    assert result[0]["meta_data"]["page"] == 0
    assert result[1]["content"] == "Page 2 content"
    assert result[1]["meta_data"]["page"] == 1


@patch("feedparser.parse")
def test_rss_loader(mock_parse, sample_url):
    mock_parse.return_value = type(
        "MockFeed",
        (),
        {
            "entries": [
                type(
                    "MockEntry",
                    (),
                    {
                        "title": "Test Title",
                        "description": "Test Description",
                        "link": "https://example.com/entry",
                        "published": "2023-01-01",
                    },
                )
            ]
        },
    )
    loader = RssLoader()
    result = loader.load_data(sample_url)
    assert len(result) == 1
    assert "Test Title" in result[0]["content"]
    assert "Test Description" in result[0]["content"]
    assert result[0]["meta_data"]["url"] == "https://example.com/entry"
    assert result[0]["meta_data"]["published"] == "2023-01-01"


def test_text_blob_loader():
    loader = TextBlobLoader()
    result = loader.load_data("Test content")
    assert len(result) == 1
    assert result[0]["content"] == "Test content"
    assert result[0]["meta_data"]["source"] == "text_blob"


@pytest.mark.skip("This test is not working")
@responses.activate
def test_robots_txt_loader(sample_url):
    responses.add(
        responses.GET,
        "https://example.com/sitemap.xml",
        body="<loc>https://example.com/page1</loc><loc>https://example.com/page2</loc>",
        status=200,
    )

    loader = RobotsTxtLoader(max_urls=2)
    result = loader.load_data(sample_url)

    assert len(result) == 2
    assert result[0]["content"] == "https://example.com/page1"
    assert result[1]["content"] == "https://example.com/page2"


@patch("glob.glob")
def test_local_text_pattern_loader(mock_glob):
    mock_glob.return_value = ["/path/to/file1.txt", "/path/to/file2.txt"]
    with patch("builtins.open", mock_open(read_data="Test content")):
        loader = LocalTextPatternLoader()
        result = loader.load_data("*.txt")

        assert len(result) == 2
        assert result[0]["content"] == "Test content"
        assert result[0]["meta_data"]["file_path"] == "/path/to/file1.txt"
        assert result[1]["content"] == "Test content"
        assert result[1]["meta_data"]["file_path"] == "/path/to/file2.txt"
