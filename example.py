import re

from pydantic import BaseModel

import morph


# Example usage
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


class Superhero2(BaseModel):
    name: str
    age: int
    power: str
    character: str
    enemies: list[str]


def main(inference=True):
    app = morph.App()
    app.set_budget(1000)
    app.set_retries(4)

    app.add("https://www.britannica.com/topic/list-of-superheroes-2024795")
    app.add("*.txt")
    # app.add("*.py")
    app.add("https://www.britannica.com/robots.txt")
    for _ in range(10):
        app.add("https://www.britannica.com/robots.txt")
    app.add("*.pdf")
    # app.add("*.md")

    # These calls are placeholders and won't work without implementing the extraction logic
    superheroes = app.extract(Superhero)
    print(f"Extracted superheroes: {list(superheroes)}")
    [
        (
            Superhero(
                name="Superman",
                age=35,
                power="Flight",
                enemies=["Lex Luthor", "Doomsday"],
            ),
            {"url": "https://www.britannica.com/topic/list-of-superheroes-2024795"},
        ),
        (
            Superhero(
                name="Wonder Woman",
                age=30,
                power="Super Strength",
                enemies=["Ares", "Cheetah"],
            ),
            {"url": "https://www.britannica.com/topic/list-of-superheroes-2024795"},
        ),
    ]
    multi_results = app.multi_extract(Superhero, Superhero2)
    print(f"Multi-extraction results: {multi_results}")

    first_hero = app.extract_first(Superhero)
    print(f"First extracted hero: {first_hero}")

    print(f"Total cost: ${app.cost()}")
    print(f"Total usage: {app.usage}")

    app.export_jsonl("fine_tune.jsonl")
    app.display_stats()
    if not inference:
        return

    app.fit()
    app.save("state.morph")
    struct = app.infer_extract(
        """
Superhero Name: Quantum Spark

Real Name: Dr. Amelia Quark

Origin: Dr. Amelia Quark was a brilliant physicist working on
cutting-edge quantum mechanics research.
During an experiment gone awry, she was bathed in exotic particles,
fundamentally altering her molecular structure.

    """
    )
    print(f"Inferred structure: {struct}")


if __name__ == "__main__":
    main()
    print("Program execution completed")
