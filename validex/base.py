import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

import tqdm
from cache_to_disk import cache_to_disk
from magentic import prompt
from pydantic import BaseModel
from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

import validex.loaders as loaders
import validex.training as training
from validex.logger import log
from validex.utils import async_cache_to_disk

T = TypeVar("T", bound=BaseModel)

BAR_KWARGS = dict(
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    colour="blue",
)


class DataCleaner:
    @staticmethod
    def clean(text: str) -> str:
        text = text.replace("\n", " ")
        cleaned_text = re.sub(r"\s+", " ", text.strip())
        cleaned_text = cleaned_text.replace("\\", "")
        cleaned_text = cleaned_text.replace("#", " ")
        cleaned_text = re.sub(r"([^\w\s])\1*", r"\1", cleaned_text)
        return cleaned_text.strip()


@cache_to_disk(182)
def _cached_extract(model: type[T], context: str) -> list[T]:
    log.debug(f"Extracting data for model: {model.__name__}")

    @prompt("Extract data: {context}")
    def magic_helper(
        context: str,
    ) -> list[model]:
        ...  # No function body as this is never executed

    return magic_helper(context)


@async_cache_to_disk(182)
async def _async_cached_extract(model: type[T], context: str) -> list[T]:
    log.debug(f"Asynchronously extracting data for model: {model.__name__}")

    @prompt("Extract data: {context}")
    async def magic_helper(
        context: str,
    ) -> list[model]:
        ...  # No function body as this is never executed

    return await magic_helper(context)


def word_count(text: str) -> int:
    return len(text.split())


class App(training.TrainingMixin):
    # add proxy to load_data method

    def __init__(self, max_workers: int = 10, training_enabled: bool = True):
        self.loaders: dict[str, loaders.BaseLoader] = {
            "pdf_file": loaders.PdfFileLoader(),
            "web_page": loaders.WebPageLoader(),
            "text": loaders.LocalTextLoader(),
            "pattern_text": loaders.LocalTextPatternLoader(),
            "rss": loaders.RssLoader(),
            "robots.txt": loaders.RobotsTxtLoader(),
            "text_blob": loaders.TextBlobLoader(),
        }
        self.data: list[str] = []
        self.data_sources: list[str] = []
        self.budget: float = float("inf")
        self.retries: int = 1
        self.data_queue: list[str] = []
        self.loaded = False
        self.usage = 0
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.dataset = []
        self.training_enabled = training_enabled

        self.console = Console()
        self.num_errors = 0
        self.num_hallucinations = 0
        self.text_size = 0

    @property
    def num_records(self) -> int:
        return len(self.data)

    def set_budget(self, budget: float) -> None:
        log.info(f"Setting budget to {budget}")
        self.budget = budget

    def set_retries(self, retries: int) -> None:
        log.info(f"Setting retries to {retries}")
        self.retries = retries

    def guess_loader_type(self, source: str) -> str:
        log.debug(f"Guessing loader type for source: {source}")
        if source.startswith("http"):
            return "web_page"
        if source.endswith(".pdf"):
            return "pdf_file"
        if source.endswith(".txt"):
            return "text"
        if source.endswith(".rss"):
            return "rss"
        if "*" in source:
            return "pattern_text"
        if len(source) > 50:
            return "text_blob"
        raise ValueError(f"Unsupported source type: {source}")

    def add(self, source: str, *_) -> None:
        log.info(f"Adding source: {source}")
        loader_type = self.guess_loader_type(source)
        loader = self.loaders.get(loader_type)
        if loader is None:
            log.error(f"No loader found for type: {loader_type}")
            raise ValueError(f"No loader found for type: {loader_type}")
        self.data_queue.append((loader, source))

    def load_later(self) -> None:
        if self.loaded:
            log.debug("Data already loaded, skipping load_later()")
            return
        log.info("Loading data")

        def load_source(loader, source):
            try:
                raw_data = loader.load_data(source)
                cleaned_data = [DataCleaner.clean(item["content"]) for item in raw_data]
                return cleaned_data, [item["meta_data"] for item in raw_data]
            except Exception as e:
                log.error(f"Error loading data from {source}: {str(e)}")
                return [], []

        executor = self.thread_pool
        futures = [
            executor.submit(load_source, loader, source)
            for loader, source in self.data_queue
        ]

        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Loading data",
            **BAR_KWARGS,
        ):
            cleaned_data, meta_data = future.result()
            self.data.extend(cleaned_data)
            self.data_sources.extend(meta_data)

        self.loaded = True
        log.info(f"Loaded {len(self.data)} items")

    def extract(
        self, model: type[T], hook=lambda record, meta: None, max_records=None
    ) -> list[T]:
        log.info(f"Extracting data for model: {model.__name__}")
        self.load_later()
        visited = set()
        has_fix = hasattr(model, "fix")
        has_check_hallucinations = hasattr(model, "check_hallucinations")

        def extract_item(item, src_meta):
            if str(src_meta) in visited:
                log.debug(f"Skipping already visited source: {src_meta}")
                return [], None, 0

            visited.add(str(src_meta))
            try:
                data = _cached_extract(model, item)
                usage = word_count(item)
                if self.training_enabled:
                    self.dataset.append((item, data))
                return data, src_meta, usage
            except Exception as e:
                log.error(f"Error extracting data: {str(e)}")
                return [], None, 0

        results = []
        executor = self.thread_pool
        futures = [
            executor.submit(extract_item, item, src_meta)
            for item, src_meta in zip(self.data, self.data_sources)
        ]

        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Extracting data",
            **BAR_KWARGS,
        ):
            data, src_meta, usage = future.result()
            self.usage += usage
            for record in data:
                if has_fix:
                    record.fix()
                if has_check_hallucinations:
                    try:
                        record.check_hallucinations()
                    except ValueError as e:
                        log.error(f"Hallucination detected: {str(e)}")
                        self.num_hallucinations += 1
                        continue
                results.append((record, src_meta))
                self.usage += word_count(str(record.json()))
                hook(record, src_meta)
                if self.usage >= self.budget:
                    log.warning("Budget exceeded, stopping extraction")
                    break

        log.info(f"Extraction complete. Usage: {self.usage}")
        return results

    async def extract_async(self, model: type[T]) -> list[T]:
        log.info(f"Asynchronously extracting data for model: {model.__name__}")
        self.load_later()
        for item in tqdm.tqdm(self.data, desc="Extracting data", **BAR_KWARGS):
            try:
                data = await _async_cached_extract(model, item)
                self.usage += word_count(item)
                for record in data:
                    yield record
                    self.usage += word_count(str(record.json()))
            except Exception as e:
                log.error(f"Error extracting data asynchronously: {str(e)}")
        log.info(f"Async extraction complete. Usage: {self.usage}")

    def multi_extract(self, *models: type[T]) -> list[list[T]]:
        # TODO: implement multi-extraction in a single llm call
        log.info(
            f"Multi-extracting data for models: {[model.__name__ for model in models]}"
        )
        self.load_later()
        results = []
        for model in models:
            model_results = self.extract(model)
            results.append([record for record, _ in model_results])
        return results

    def extract_first(self, model: type[T]) -> T:
        log.info(f"Extracting first result for model: {model.__name__}")
        self.load_later()
        for item, _ in zip(self.data, self.data_sources):
            try:
                data = _cached_extract(model, item)
                if data:
                    return data[0]
            except Exception as e:
                log.error(f"Error extracting data: {str(e)}")
        return None

    def cost(self) -> float:
        cost = round(0.0030 * self.usage / 1_000, 2)
        log.info(f"Calculated cost: ${cost}")
        return cost

    def reset(self) -> None:
        log.info("Resetting App")
        self.data = []
        self.data_sources = []
        self.data_queue = []
        self.loaded = False
        self.usage = 0

    def export_jsonl(self, filename: str) -> None:
        log.info(f"Exporting data to {filename} in JSONL format")
        with open(filename, "w") as f:
            for src, structs in self.dataset:
                json_line = {
                    "messages": [
                        {"role": "user", "content": src},
                        {
                            "role": "assistant",
                            "content": f"""```json\n{json.dumps([s.dict() for s in structs])}\n```""",
                        },
                    ]
                }
                f.write(json.dumps(json_line) + "\n")
        log.info(f"Exported {len(self.data)} items to {filename}")

    def display_stats(self):
        table = Table(title="Extraction Statistics", box=ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Number of Records", str(self.num_records))
        table.add_row("Number of Errors", str(self.num_errors))
        table.add_row("Number of Hallucinations", str(self.num_hallucinations))
        table.add_row("Total Usage (words)", f"{self.usage:,}")
        table.add_row("Text Size (characters)", f"{self.text_size:,}")
        table.add_row("Cost ($)", f"${self.cost():.2f}")

        self.console.print(table)
