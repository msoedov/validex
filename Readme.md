# ValidEx

ValidEx is a Python library that simplifies retrieval, extraction and training of structured data from various unstructured sources.

<p>
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/msoedov/validex" />
<img alt="GitHub Last Commit" src="https://img.shields.io/github/last-commit/msoedov/validex" />
<img alt="" src="https://img.shields.io/github/repo-size/msoedov/validex" />
<img alt="GitHub Issues" src="https://img.shields.io/github/issues/msoedov/validex" />
<img alt="GitHub Pull Requests" src="https://img.shields.io/github/issues-pr/msoedov/validex" />
<img alt="Github License" src="https://img.shields.io/github/license/msoedov/validex" />
</p>

## 🏷 Features

- **Structured Data Extraction**: Parse and extract structured data from various unstructured sources including web pages, text files, PDFs, and more.
- **Heuristic data cleaning**  text normalization (case, whitespace, special characters), deduplication
- **Concurrency Support**: Efficiently process multiple data sources simultaneously.
- **Retry Mechanism**: Implement automatic retries for failed extraction attempts.
- **Hallucination check**: Implement strategies to detect and reduce LLM hallucinations in extracted data.
- **Fine-tuning Dataset Export**: Generate datasets in JSONL format for OpenAI chat fine-tuning.
- **Local Model Creation**: Build custom extraction models combining Named Entity Recognition (NER) and regular expressions.

## 📦 Installation

To get started with ValidEx, simply install the package using pip:

```shell
pip install validex
```

## ⛓️ Quick Start

```python
import validex
from pydantic import BaseModel


class Superhero(BaseModel):
    name: str
    age: int
    power: str
    enemies: list[str]


def main():
    app = validex.App()

    app.add("https://www.britannica.com/topic/list-of-superheroes-2024795")
    app.add("*.txt")
    app.add("*.pdf")
    app.add("*.md")

    superheroes = app.extract(Superhero)
    print(f"Extracted superheroes: {list(superheroes)}")

    first_hero = app.extract_first(Superhero)
    print(f"First extracted hero: {first_hero}")

    print(f"Total cost: ${app.cost()}")
    print(f"Total usage: {app.usage}")


if __name__ == "__main__":
    main()
```

```python
[
    (
        Superhero(
            name="Batman",
            age=81,
            power="Brilliant detective skills, martial arts",
            enemies=["Joker", "Penguin"],
        ),
        {"url": "https://www.britannica.com/topic/list-of-superheroes-2024795"},
    ),
    (
        Superhero(
            name="Wonder Woman",
            age=80,
            power="Superhuman strength, speed, agility",
            enemies=["Ares", "Cheetah"],
        ),
        {"url": "https://www.britannica.com/topic/list-of-superheroes-2024795"},
    ),
    (
        Superhero(
            name="Spider-Man",
            age=59,
            power="Wall-crawling, spider sense",
            enemies=["Green Goblin", "Venom"],
        ),
        {"url": "https://www.britannica.com/topic/list-of-superheroes-2024795"},
    ),
    (
        Superhero(
            name="Captain America",
            age=101,
            power="Super soldier serum, shield",
            enemies=["Red Skull", "Hydra"],
        ),
        {"url": "https://www.britannica.com/topic/list-of-superheroes-2024795"},
    ),
    (
        Superhero(
            name="Superman", age=35, power="Flight", enemies=["Lex Luthor", "Doomsday"]
        ),
        {"url": "https://www.britannica.com/robots.txt"},
    ),
    (
        Superhero(
            name="Wonder Woman",
            age=30,
            power="Super Strength",
            enemies=["Ares", "Cheetah"],
        ),
        {"url": "https://www.britannica.com/robots.txt"},
    ),
    (
        Superhero(
            name="Spider-Man",
            age=25,
            power="Wall-crawling",
            enemies=["Green Goblin", "Venom"],
        ),
        {"url": "https://www.britannica.com/robots.txt"},
    ),
]
```

### Hallucinations and autofix

```python
class Superhero(BaseModel):
    name: str
    age: int
    power: str
    enemies: list[str]

    def fix(self):
        # Logic to auto fix and normalize the generated data
        if self.age < 0:
            self.age = 0

    def check_hallucinations(self):
        # Check name
        if not re.match(r"^[A-Za-z\s-]+$", self.name):
            raise ValueError(f"Name '{self.name}' contains unusual characters")

        # Check age
        if self.age < 0 or self.age > 1000:
            raise ValueError(f"Age {self.age} seems unrealistic")

        # Check power
        if len(self.power) > 50:
            raise ValueError("Power description is unusually long")

        # Check enemies
        if len(self.enemies) > 10:
            raise ValueError("Unusually high number of enemies")

        for enemy in self.enemies:
            if not re.match(r"^[A-Za-z\s-]+$", enemy):
                raise ValueError(f"Enemy name '{enemy}' contains unusual characters")
```

### Experimental: Export and fine tunning

```python
# Use the OpenAI chat fine-tuning format to save data
app.export_jsonl("fine_tune.jsonl")

# Local model training
app.fit()
app.save("state.validex")


app.infer_extract("booob")
```

### Multi-model Extraction

ValidEx supports extracting multiple models at once

```python
class Superhero2(BaseModel):
    name: str
    age: int
    power: str
    enemies: list[str]


multi_results = app.multi_extract(Superhero, Superhero2)
print(f"Multi-extraction results: {multi_results}")
```

### Limitations

TBD

## 🛠️ Roadmap

## 👋 Contributing

Contributions to ValidEx are welcome! If you'd like to contribute, please follow these steps:

- Fork the repository on GitHub
- Create a new branch for your changes
- Commit your changes to the new branch
- Push your changes to the forked repository
- Open a pull request to the main ValidEx repository

Before contributing, please read the contributing guidelines.

## License

ValidEx is released under the MIT License.
