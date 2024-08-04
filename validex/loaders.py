import glob
import re
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import justext
import requests
import stamina
from cache_to_disk import cache_to_disk

from .logger import log


class BaseLoader(ABC):
    @abstractmethod
    def load_data(self, source: str) -> list[dict[str, Any]]:
        raise NotImplementedError


class WebPageLoader(BaseLoader):
    @stamina.retry(on=requests.exceptions.RequestException)
    def load_data(self, url: str, proxy=None) -> list[dict[str, Any]]:
        log.info(f"Loading web page: {url}")
        response = cache_to_disk(182)(requests.get)(url, proxies=proxy)
        paragraphs = justext.justext(response.content, justext.get_stoplist("English"))
        paragraphs = [p.text for p in paragraphs if not p.is_boilerplate]
        log.info(f"Loaded {len(paragraphs)} paragraphs from {url}")
        return [{"content": "\n".join(paragraphs), "meta_data": {"url": url}}]


class LocalTextLoader(BaseLoader):
    def load_data(self, file_path: str) -> list[dict[str, Any]]:
        log.info(f"Loading local text file: {file_path}")
        with open(file_path) as file:
            content = file.read()
        log.info(f"Loaded {len(content)} characters from {file_path}")
        return [
            {"content": content, "meta_data": {"url": "local", "file_path": file_path}}
        ]


class PdfFileLoader(BaseLoader):
    def load_data(self, file_path: str) -> list[dict[str, Any]]:
        log.info(f"Loading PDF file: {file_path}")
        import pypdf

        pdf_reader = pypdf.PdfReader(file_path)
        log.info(f"Loaded PDF with {len(pdf_reader.pages)} pages")
        return [
            {
                "content": page.extract_text(),
                "meta_data": {"source": file_path, "page": page_number},
            }
            for page_number, page in enumerate(pdf_reader.pages)
        ]


class RssLoader(BaseLoader):
    def load_data(self, url: str) -> list[dict[str, Any]]:
        import feedparser

        log.info(f"Loading RSS feed: {url}")
        feed = feedparser.parse(url)
        entries = []
        for entry in feed.entries:
            content = f"Title: {entry.title}\n\nDescription: {entry.description}"
            entries.append(
                {
                    "content": content,
                    "meta_data": {"url": entry.link, "published": entry.published},
                }
            )
        log.info(f"Loaded {len(entries)} entries from RSS feed")
        return entries


class TextBlobLoader(BaseLoader):
    def load_data(self, text: str) -> list[dict[str, Any]]:
        log.info("Loading text blob")
        return [{"content": text, "meta_data": {"source": "text_blob"}}]


@cache_to_disk(182)
def read_robot_file(url):
    rp = RobotFileParser()
    rp.set_url(url)
    rp.read()
    return rp


class RobotsTxtLoader(BaseLoader):
    def __init__(self, max_urls: int = 100):
        self.max_urls = max_urls

    def load_data(self, url: str) -> list[dict[str, Any]]:
        log.info(f"Loading robots.txt from: {url}")
        rp = read_robot_file(url)

        base_url = url.rsplit("/", 1)[0]
        sitemaps = rp.site_maps()
        allowed_urls = []

        if sitemaps:
            log.info(f"Found {len(sitemaps)} sitemaps")
            for sitemap in sitemaps:
                response = requests.get(sitemap)
                urls = re.findall(r"<loc>(.*?)</loc>", response.text)
                allowed_urls.extend(urls[: self.max_urls - len(allowed_urls)])
                if len(allowed_urls) >= self.max_urls:
                    break

        if len(allowed_urls) < self.max_urls:
            log.info("Not enough URLs found in sitemaps, checking robots.txt rules")
            for line in rp.default_entry.rulelines:
                if line.allowance and line.path != "/":
                    full_url = urljoin(base_url, line.path)
                    if full_url not in allowed_urls:
                        allowed_urls.append(full_url)
                    if len(allowed_urls) >= self.max_urls:
                        break

        log.info(f"Found {len(allowed_urls)} allowed URLs")
        return [
            {"content": url, "meta_data": {"url": url}}
            for url in allowed_urls[: self.max_urls]
        ]


class LocalTextPatternLoader(BaseLoader):
    def load_data(self, pattern: str) -> list[dict[str, Any]]:
        log.info(f"Loading local text files matching pattern: {pattern}")
        matching_files = glob.glob(pattern)
        log.info(f"Found {len(matching_files)} matching files")
        results = []
        for file_path in matching_files:
            with open(file_path) as file:
                content = file.read()
            log.debug(f"Loaded {len(content)} characters from {file_path}")
            results.append(
                {
                    "content": content,
                    "meta_data": {"url": "local", "file_path": file_path},
                }
            )
        return results
