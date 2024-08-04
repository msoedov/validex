from unittest.mock import patch

import pytest
from pydantic import BaseModel

import morph.loaders as loaders
from morph.base import App, DataCleaner


# Example usage
class Superhero(BaseModel):
    name: str
    age: int
    power: str
    enemies: list[str]


class Superhero2(BaseModel):
    name: str
    age: int
    power: str
    enemies: list[str]


@pytest.fixture
def setup_schema_flow():
    schema_flow = App()
    schema_flow.set_budget(1000)
    schema_flow.set_retries(4)
    return schema_flow


def test_data_cleaner():
    raw_text = "Hello\n\nWorld!  This   is a test.\n\n\\ ##"
    cleaned_text = DataCleaner.clean(raw_text)
    assert cleaned_text == "Hello World! This is a test."


def test_rss_loader():
    ...


def test_add_supported_loader(setup_schema_flow):
    setup_schema_flow.add("https://www.example.com", "web_page")
    assert len(setup_schema_flow.data_queue) == 1


def test_add_unsupported_loader(setup_schema_flow):
    with pytest.raises(ValueError):
        setup_schema_flow.add("xml", "unsupported_loader")


@patch.object(loaders.WebPageLoader, "load_data")
def test_web_page_loader(mock_load_data, setup_schema_flow):
    mock_load_data.return_value = [
        {"content": "Example content", "meta_data": {"url": "https://www.example.com"}}
    ]
    setup_schema_flow.add("https://www.example.com", "web_page")
    setup_schema_flow.load_later()
    assert "Example content" in setup_schema_flow.data


@patch.object(loaders.LocalTextLoader, "load_data")
def test_local_text_loader(mock_load_data, setup_schema_flow):
    mock_load_data.return_value = [
        {
            "content": "Local file content",
            "meta_data": {"url": "local", "file_path": "document.txt"},
        }
    ]
    setup_schema_flow.add("document.txt", "text")
    setup_schema_flow.load_later()
    assert "Local file content" in setup_schema_flow.data


@patch.object(loaders.PdfFileLoader, "load_data")
def test_pdf_file_loader(mock_load_data, setup_schema_flow):
    mock_load_data.return_value = [
        {
            "content": "PDF file content",
            "meta_data": {"source": "document.pdf", "page": 0},
        }
    ]
    setup_schema_flow.add("document.pdf", "pdf_file")
    setup_schema_flow.load_later()
    assert "PDF file content" in setup_schema_flow.data


@patch("morph.base._cached_extract")
def test_extract(mock_cached_extract, setup_schema_flow):
    mock_cached_extract.return_value = [
        Superhero(name="Superman", age=30, power="Flying", enemies=["Lex Luthor"])
    ]
    setup_schema_flow.add("https://www.example.com", "web_page")
    setup_schema_flow.load_later()
    superheroes = list(setup_schema_flow.extract(Superhero))
    assert len(superheroes) == 1
    assert superheroes[0][0].name == "Superman"


@patch("morph.base._async_cached_extract")
@pytest.mark.asyncio
async def test_extract_async(mock_async_cached_extract, setup_schema_flow):
    mock_async_cached_extract.return_value = [
        Superhero(name="Superman", age=30, power="Flying", enemies=["Lex Luthor"])
    ]
    setup_schema_flow.add("https://www.example.com", "web_page")
    setup_schema_flow.load_later()
    superheroes = [hero async for hero in setup_schema_flow.extract_async(Superhero)]
    assert len(superheroes) == 1
    assert superheroes[0].name == "Superman"


def test_set_budget(setup_schema_flow):
    setup_schema_flow.set_budget(500)
    assert setup_schema_flow.budget == 500


def test_set_retries(setup_schema_flow):
    setup_schema_flow.set_retries(2)
    assert setup_schema_flow.retries == 2
